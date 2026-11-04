"""Shared phase budget, replay and controlled recovery of explicit 503 responses only."""

import json

from repro import ROOT
from repro.recovery import safe_json, utcnow
from repro.replay import ReplayTransport
from repro.transport import CostLedger, ExperimentStopped

STATE = ROOT / "results/repro"
CIRCUIT = STATE / "v3-circuit.json"
STOP = STATE / "v3-stop"


class Overloaded(ExperimentStopped):
    pass


class PhaseLedger(CostLedger):
    def _check(self, amount):
        super()._check(amount)
        rows = [row for row in self._records() if row["run_id"].startswith("v3-")]
        if sum(row["accounted_cny"] for row in rows)+amount > 30:
            raise ExperimentStopped("V3 phase budget reached")
        if sum(row["accounted_cny"] for row in rows
               if row["state"] in ("pending", "failed_billing_unknown"))+amount > 5:
            raise ExperimentStopped("V3 failure/in-flight reservation limit reached")


class PhaseReplay(ReplayTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = PhaseLedger(self.ledger.path, self.run_id, self.config)

    def call(self, *args, **kwargs):
        if STOP.exists():
            raise ExperimentStopped("V3 stop requested")
        if CIRCUIT.exists():
            raise Overloaded("Shared V3 circuit open")
        return super().call(*args, **kwargs)

    def _recover(self, fingerprint):
        records = [row for row in self.ledger.records()
                   if row["run_id"] == self.run_id and row.get("logical_index") == self.position]
        if records and all(row["state"] == "failed_billing_unknown" and row.get("status_code") == 503
                           for row in records):
            if any(row.get("prompt_fingerprint") != fingerprint for row in records):
                raise ExperimentStopped("503 checkpoint fingerprint changed")
            directory = self.output / "recovery"
            safe_json(directory / f"retry-{len(list(directory.glob('*.json')))+1:04}.json",
                      {"at_utc": utcnow(), "logical_index": self.position,
                       "prior_requests": [row["request_id"] for row in records],
                       "prior_reserves_retained_cny": sum(row["accounted_cny"] for row in records)})
            return None
        return super()._recover(fingerprint)

    def _failure(self, record, exc, started):
        super()._failure(record, exc, started)
        if getattr(exc, "status_code", None) == 503:
            safe_json(CIRCUIT, {"opened_at_utc": utcnow(), "request_id": record["request_id"]})
            raise Overloaded("HTTP 503; reservation retained until scheduled retry") from None
        raise ExperimentStopped(f"Non-503 request failure: {type(exc).__name__}; no automatic restart") from None
