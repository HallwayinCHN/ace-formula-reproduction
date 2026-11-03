"""Read aggregate job progress without exposing prompts, answers or credentials."""

import json
import re

from repro import ROOT
from repro.pilot import load_config
from repro.transport import CostLedger


def main():
    records = CostLedger(ROOT / "results/repro/live_usage.jsonl", "progress", load_config()).records()
    jobs = []
    for variant in ("ace", "selective"):
        for seed in (42, 43, 44):
            name = f"formal-{variant}-seed{seed}"
            rows = [row for row in records if row["run_id"] == name]
            path = ROOT / "results/repro" / name
            summary = path / "summary.json"
            steps = [int(match.group(1)) for row in rows
                     if (match := re.search(r"train_e_1_s_(\d+)", row["call_id"]))]
            item = {"job": name, "step": max(steps, default=0), "requests": len(rows),
                    "accounted_cny": round(sum(row["accounted_cny"] for row in rows), 4),
                    "state": "not_started"}
            if rows:
                item.update(state="running", last_call=rows[-1]["call_id"], last_role=rows[-1]["role"])
            if summary.exists():
                data = json.loads(summary.read_text())
                item.update(state="completed", best_validation_accuracy=data["best_validation_accuracy"])
            elif (path / "status.json").exists():
                try:
                    item["state"] = json.loads((path / "status.json").read_text())["status"]
                except json.JSONDecodeError:
                    pass
            jobs.append(item)
    print(json.dumps({"jobs": jobs, "all_stages_accounted_cny": round(sum(row["accounted_cny"] for row in records), 4)},
                     indent=2))


if __name__ == "__main__":
    main()
