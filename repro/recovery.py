"""User-authorized operational recovery of explicit HTTP 503 responses; keep all cost reserves."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from repro import ROOT
from repro.holdout import LOCK
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest, verify_upstream
from repro.resume_balance import BalanceAwareReplay
from repro.transport import CostLedger, ExperimentStopped

STATE = ROOT / "results/repro"
CIRCUIT = STATE / "stage3-recovery-circuit.json"
STOP = STATE / "stage3-auto-resume.stop"
POLICY = ARTIFACTS / "stage3_recovery_policy.json"
CONTRACT = ARTIFACTS / "stage3_recovery_contract.json"
SOURCES = ("repro/recovery.py", "repro/resume_holdout.py", "repro/auto_resume.py",
           "repro/artifacts/stage3_recovery_policy.json")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def safe_json(path, value):
    """Unique temporary files also permit simultaneous circuit trips in separate processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def verify_original():
    verify_upstream()
    freeze = json.loads(LOCK.read_text())
    for path, expected in freeze["file_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ExperimentStopped(f"Frozen source/input changed: {path}")
    return freeze


def seal_recovery():
    verify_original()
    if CONTRACT.exists():
        return validate_recovery()
    policy = json.loads(POLICY.read_text())
    files = list(STATE.glob("holdout-*/items/*.json")) + list(STATE.glob("holdout-*/replay/*.json"))
    ledger = CostLedger(STATE / "live_usage.jsonl", "recovery-seal", load_config()).records()
    if any(row["run_id"].startswith("holdout-") and row["state"] == "pending" for row in ledger):
        raise ExperimentStopped("Existing request is still in flight; do not overlap recovery")
    contract = {
        "created_at_utc": utcnow(), "authorization": "User requested periodic detection and checkpoint resume",
        "original_contract_sha256": digest(LOCK.read_bytes()), "policy": policy,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {name: digest((ROOT / name).read_bytes()) for name in SOURCES},
        "preserved_files": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in files},
        "successful_predictions_before_recovery": sum("/items/" in str(path) for path in files),
        "existing_unknown_billing_reserve_cny": sum(row["accounted_cny"] for row in ledger
                                                    if row["state"] == "failed_billing_unknown"),
        "deviation": "Operational 503 cooldown/retry only; original method/inputs/scoring remain frozen. "
                     "Unknown charges are NOT reconciled to zero. No test scores opened.",
    }
    safe_json(CONTRACT, contract)
    return contract


def validate_recovery():
    verify_original()
    contract = json.loads(CONTRACT.read_text())
    if contract["original_contract_sha256"] != digest(LOCK.read_bytes()):
        raise ExperimentStopped("Original contract changed")
    for name, expected in {**contract["source_hashes"], **contract["preserved_files"]}.items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ExperimentStopped(f"Recovery source or preserved response changed: {name}")
    return contract


def failure_exposure(rows, since):
    return sum(row["accounted_cny"] for row in rows if row["run_id"].startswith("holdout-")
               and row["timestamp"] >= since and row["state"] in ("pending", "failed_billing_unknown"))


class RecoveryLedger(CostLedger):
    def __init__(self, path, run_id, config, contract):
        super().__init__(path, run_id, config)
        self.contract = contract

    def _check(self, amount):
        super()._check(amount)
        exposure = failure_exposure(self._records(), self.contract["created_at_utc"])
        if exposure + amount > self.contract["policy"]["max_extra_failure_reserve_cny"]:
            raise ExperimentStopped("Additional failure/in-flight reservation limit reached")


class ProbeFinished(ExperimentStopped):
    pass


class ServiceUnavailable(ExperimentStopped):
    pass


class OverloadAwareReplay(BalanceAwareReplay):
    def __init__(self, *args, probe=False, recovery=None, **kwargs):
        self.recovery = recovery or validate_recovery()
        self.probe = probe
        super().__init__(*args, **kwargs)
        self.ledger = RecoveryLedger(self.ledger.path, self.run_id, self.config, self.recovery)

    def call(self, *args, **kwargs):
        if STOP.exists():
            raise ExperimentStopped("Automatic recovery stopped by user")
        if CIRCUIT.exists():
            raise ServiceUnavailable("Shared 503 circuit is open; no new request")
        if self.probe and self.calls >= 1:
            raise ProbeFinished("One real pending prediction completed; no additional probe")
        return super().call(*args, **kwargs)

    def _recover(self, fingerprint):
        records = [row for row in self.ledger.records()
                   if row["run_id"] == self.run_id and row.get("logical_index") == self.position]
        explicit_503 = records and all(row["state"] == "failed_billing_unknown" and row.get("status_code") == 503
                                       for row in records)
        if not explicit_503:
            return super()._recover(fingerprint)
        if any(row.get("prompt_fingerprint") != fingerprint for row in records):
            raise ExperimentStopped("503 retry fingerprint differs; no request")
        directory = self.output / "recovery"
        sequence = len(list(directory.glob("retry-*.json"))) + 1
        safe_json(directory / f"retry-{sequence:05}.json", {
            "authorized_at_utc": utcnow(), "logical_index": self.position,
            "prompt_fingerprint": fingerprint, "prior_requests": [row["request_id"] for row in records],
            "retained_unknown_cost_reserve_cny": sum(row["accounted_cny"] for row in records),
            "recovery_contract_sha256": digest(CONTRACT.read_bytes()),
            "reason": "Explicit HTTP 503 retry authorized by user; billing remains unresolved",
        })
        return None

    def _failure(self, record, exc, started):
        super()._failure(record, exc, started)
        if getattr(exc, "status_code", None) == 503:
            safe_json(CIRCUIT, {"opened_at_utc": utcnow(), "run_id": self.run_id,
                                "request_id": record["request_id"], "http_status": 503})
            raise ServiceUnavailable("HTTP 503: preserve cost reserve and wait for scheduled retry") from None
        # Unknown timeouts, auth/balance failures and all other statuses require attention, not automatic restarts.
        raise ExperimentStopped(f"Non-503 failure: {type(exc).__name__}; automatic retry blocked") from None
