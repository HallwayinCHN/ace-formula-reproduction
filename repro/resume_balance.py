"""Resume explicitly rejected HTTP-402 slots after a fresh read-only balance check."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re

from repro import ROOT
from repro.balance import get_balance
from repro.pilot import load_config
from repro.replay import ReplayTransport, atomic_json
from repro.transport import CostLedger, ExperimentStopped


def reconcile_rejections(ledger):
    count = 0
    for row in ledger.records():
        if row.get("status_code") == 402 and row["state"] == "failed_billing_unknown":
            row.update(state="rejected_402", accounted_cny=0.0,
                       reconciliation="Explicit HTTP rejection; no completion usage; not a timed-out success",
                       reconciled_at_utc=datetime.now(timezone.utc).isoformat())
            ledger.append(row)
            count += 1
    return count


class BalanceAwareReplay(ReplayTransport):
    def _recover(self, fingerprint):
        records = [row for row in self.ledger.records()
                   if row["run_id"] == self.run_id and row.get("logical_index") == self.position]
        if records and all(row["state"] == "rejected_402" for row in records):
            if any(row.get("prompt_fingerprint") != fingerprint for row in records):
                raise ExperimentStopped("Rejected-slot fingerprint changed")
            return None
        return super()._recover(fingerprint)

    def _failure(self, record, exc, started):
        super()._failure(record, exc, started)
        if getattr(exc, "status_code", None) == 402:
            body = getattr(exc, "body", {}) or {}
            error = body.get("error", body) if isinstance(body, dict) else {}
            message = str(error.get("message", "")) if isinstance(error, dict) else ""
            key = os.environ.get(self.config["key_env"], "")
            if key:
                message = message.replace(key, "[REDACTED]")
            message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
            record.update(state="rejected_402", accounted_cny=0.0, provider_message=message[:300])
            self.ledger.append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    balance = get_balance()
    if balance["is_available"] is not True:
        raise ExperimentStopped("Provider reports no usable balance; no model request sent")
    config = load_config()
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "reconcile", config)
    reconciled = reconcile_rejections(ledger)
    output = ROOT / "results/repro" / f"formal-selective-seed{args.seed}"
    atomic_json(output / "balance_resume.json", {
        "balance_checked_at_utc": balance["checked_at_utc"], "is_available": True,
        "resume_adapter_sha256": hashlib.sha256((ROOT / "repro/resume_balance.py").read_bytes()).hexdigest(),
        "reconciled_402_requests": reconciled, "method_and_prompt_config_changed": False,
    })
    from repro import formal
    formal.ReplayTransport = BalanceAwareReplay
    formal.run_job("selective", args.seed)


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
