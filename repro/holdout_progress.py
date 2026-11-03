"""Blinded progress: counts and estimated expenses only; never load predictions or labels."""

import json

from repro import ROOT
from repro.holdout import VARIANTS
from repro.pilot import load_config
from repro.transport import CostLedger


def main():
    jobs = []
    for variant in VARIANTS:
        for seed in (42, 43, 44):
            output = ROOT / "results/repro" / f"holdout-{variant}-seed{seed}"
            item = {"variant": variant, "seed": seed, "items": 0, "state": "not_started"}
            if (output / "progress.json").exists():
                item.update(items=json.loads((output / "progress.json").read_text())["items_completed"], state="running")
            if (output / "status.json").exists():
                item["state"] = json.loads((output / "status.json").read_text())["status"]
            jobs.append(item)
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "monitor", load_config()).records()
    print(json.dumps({"jobs": jobs, "predictions_completed": sum(row["items"] for row in jobs),
                      "completed_conditions": sum(row["state"] == "completed" for row in jobs),
                      "all_stages_accounted_cny": sum(row["accounted_cny"] for row in ledger),
                      "blinded": True}, indent=2))


if __name__ == "__main__":
    main()
