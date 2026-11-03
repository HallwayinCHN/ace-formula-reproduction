"""Seal stage-three sources, books, IDs and conservative budget before any held-out model call."""

import json
from datetime import datetime, timezone
import subprocess
from pathlib import Path

from repro import ROOT
from repro.holdout import LOCK
from repro.holdout_stats import standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest, verify_upstream
from repro.replay import atomic_json
from repro.transport import CostLedger

SOURCES = (
    "repro/holdout.py", "repro/holdout_batch.py", "repro/holdout_stats.py", "repro/holdout_analysis.py",
    "repro/holdout_freeze.py", "repro/methods.py", "repro/transport.py", "repro/replay.py",
    "repro/holdout_progress.py",
    "repro/resume_balance.py", "repro/balance.py", "repro/pilot.py", "repro/prepare.py",
    "repro/configs/pilot.json", "repro/artifacts/sample_manifest.json", "repro/artifacts/frozen_books.json",
    "repro/artifacts/STAGE3_PLAN.md", "eval/finance/data/formula_test.jsonl",
)
SCHEDULE = {42: ["base", "ace", "selective", "retrieval", "combined"],
            43: ["selective", "retrieval", "combined", "base", "ace"],
            44: ["combined", "base", "ace", "selective", "retrieval"]}


def freeze():
    verify_upstream()
    protocol = json.loads((ROOT / "repro/configs/protocol.json").read_text())
    if protocol["formal_run_status"] != "training_complete":
        raise ValueError("Formal learning has not completed")
    books = json.loads((ARTIFACTS / "frozen_books.json").read_text())["books"]
    paths = list(SOURCES)
    for book in books:
        if digest((ROOT / book["path"]).read_bytes()) != book["sha256"]:
            raise ValueError("Book hash changed")
        paths.append(book["path"])
    config = load_config()
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "freeze", config)
    if any(row["run_id"].startswith("holdout-") for row in ledger.records()) and not LOCK.exists():
        raise ValueError("Held-out calls already exist without a frozen contract")
    spent = ledger.totals()[0]
    # Pessimistic forecast: 700 native overhead tokens, 1.2x local book tokens, 1,000 output tokens.
    trained = json.loads((ARTIFACTS / "stage2_analysis.json").read_text())["runs"]
    input_tokens = 1500 * 700 + 100 * 1.2 * sum(row["full_book_tokens"] for row in trained)
    input_tokens += 100 * 1.2 * sum(min(1500, row["full_book_tokens"]) for row in trained)
    additional = standardized_cost(input_tokens, 1500 * 1000, config) * 1.3
    if spent + additional > config["project_limit_cny"]:
        raise ValueError("Conservative forecast exceeds project budget; cohort remains unchanged")
    contract = {
        "protocol": "FTEC5660-ACE-Formula-heldout-v1", "question_count": 100, "condition_count": 5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_count": 3, "logical_predictions": 1500, "primary_comparison": "combined_minus_ace",
        "condition_schedule": SCHEDULE, "bootstrap_seed": 2026, "bootstrap_repetitions": 10000,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "file_hashes": {path: digest((ROOT / path).read_bytes()) for path in paths},
        "spent_before_test_cny": spent, "additional_forecast_with_30pct_margin_cny": additional,
        "forecast_assumptions": "No cache discount; 700 overhead tokens, 1.2x local book tokens, 1000 output tokens",
        "blinding": "No scoring until fifteen inference jobs complete; target removed from generator inputs",
    }
    if LOCK.exists():
        previous = json.loads(LOCK.read_text())
        if previous["file_hashes"] != contract["file_hashes"]:
            raise ValueError("Refusing to replace an existing held-out source contract")
    else:
        atomic_json(LOCK, contract)
    print(json.dumps({"sealed": True, "predictions": 1500, "spent_cny": spent,
                      "additional_forecast_with_margin_cny": additional}, indent=2))


if __name__ == "__main__":
    freeze()
