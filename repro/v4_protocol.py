"""Freeze a single prospective comparison of the six already-selected V3 books."""

import json
import subprocess

from repro import ROOT
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger, ExperimentStopped
from repro.v3_protocol import verify as verify_v3

CONTRACT = ARTIFACTS / "v4_contract.json"
DATA = ROOT / "eval/finance/data" / FILES["validation"]
SOURCES = ("repro/v4_protocol.py", "repro/v4_transport.py", "repro/v4_job.py",
           "repro/v4_analysis.py", "repro/v4_batch.py", "ace/core/generator.py", "ace/prompts/generator.py",
           "eval/finance/data_processor.py", "utils.py", "repro/holdout_stats.py",
           "repro/transport.py", "repro/replay.py", "repro/pilot.py")


def previous_evidence():
    old = verify_v3()
    manifest = ARTIFACTS / "v3_exports.json"
    for path, expected in json.loads(manifest.read_text())["sha256"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ExperimentStopped(f"V3 evidence changed: {path}")
    if json.loads((ARTIFACTS / "v3_router_lock.json").read_text())["passed"]:
        raise ExperimentStopped("Expected the rejected V3 routing condition")
    if list((ROOT / "results/repro").glob("v3-eval-*/predictions.json")):
        raise ExperimentStopped("Project evaluation cohort already has V3 predictions")
    return old


def freeze():
    if CONTRACT.exists():
        return verify()
    old = previous_evidence()
    ids = old["evaluation_ids"]
    if len(ids) != 160 or len(set(ids)) != 160 or set(ids) & set(old["dev_ids"]):
        raise ExperimentStopped("Unexpected evaluation cohort")
    if list((ROOT / "results/repro").glob("v4-*/items/*.json")):
        raise ExperimentStopped("V4 predictions predate its contract")
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    validation_cost = sum(group["all_workflow_accounted_cny"]-group["learning_accounted_cny"]
                          for group in training["groups"].values())
    forecast = validation_cost*960/360*1.3
    config = load_config()
    spent = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v4-freeze", config).totals()[0]
    if forecast > 10 or spent+forecast > 180:
        raise ExperimentStopped("Forecast exceeds authorized phase/project budget")
    books = {}
    for variant in ("original", "nodiag"):
        for seed in (42, 43, 44):
            path = ARTIFACTS / "v3_training" / f"{variant}-seed{seed}-best_playbook.txt"
            books[f"{variant}-seed{seed}"] = {"path": str(path.relative_to(ROOT)),
                                              "sha256": digest(path.read_bytes())}
    contract = {
        "created_at_utc": utcnow(), "evaluation_ids": ids, "questions": 160,
        "variants": ["original", "nodiag"], "seeds": [42, 43, 44], "predictions": 960,
        "origin": "Validation-derived project holdout, not a new official test set",
        "prior_boundary_deviation": "V3 formal_data bulk-parsed the validation file but selected only old 30; "
                                    "no heldout inference/scoring or use in method selection occurred",
        "books": books, "book_selection": "V3 earliest best of steps30/60 on old 30 validation questions",
        "schedule": {"42": ["original", "nodiag"], "43": ["nodiag", "original"],
                     "44": ["original", "nodiag"]},
        "inference_config": {**config, "max_calls": 160, "run_limit_cny": 10},
        "inference": "Original full-book generator prompt, one call per item; no routing, answer retry or retraining",
        "primary": "nodiag minus original; question-cluster paired bootstrap 10000 repeats, RNG2026",
        "operational_accuracy_target": {"minimum_delta_pp": -1, "minimum_ci95_lower_pp": -2},
        "interpretation": "Fixed-six-book accuracy audit; not formal noninferiority or causal proof. "
                          "An inconclusive result ends this protocol without tuning/retesting.",
        "phase_limit_cny": 10, "project_limit_cny": 180, "failure_and_inflight_limit_cny": 2,
        "503_cooldown_seconds": 300, "maximum_recovery_rounds": 12,
        "spent_before_phase_cny": spent, "forecast_plus_30pct_cny": forecast,
        "validation_sha256": old["validation_sha256"],
        "v3_exports_sha256": digest((ARTIFACTS / "v3_exports.json").read_bytes()),
        "source_hashes": {path: digest((ROOT / path).read_bytes()) for path in SOURCES},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "pricing_checked_client_date": "2026-11-06", "pricing": "Frozen peak USD prices and factor8 buffer; not bills",
    }
    safe_json(CONTRACT, contract)
    return contract


def verify():
    previous_evidence()
    contract = json.loads(CONTRACT.read_text())
    for path, expected in contract["source_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ExperimentStopped(f"V4 experimental source changed: {path}")
    if digest(DATA.read_bytes()) != contract["validation_sha256"]:
        raise ExperimentStopped("Validation data changed")
    if digest((ARTIFACTS / "v3_exports.json").read_bytes()) != contract["v3_exports_sha256"]:
        raise ExperimentStopped("V3 source evidence manifest changed")
    for book in contract["books"].values():
        if digest((ROOT / book["path"]).read_bytes()) != book["sha256"]:
            raise ExperimentStopped("A selected V4 book changed")
    return contract


def read_rows(ids, targets=False):
    """Select physical line IDs BEFORE JSON decoding. Agent inputs contain no target field."""
    from eval.finance.data_processor import DataProcessor
    if len(ids) != len(set(ids)):
        raise ExperimentStopped("Duplicate row IDs")
    wanted, selected = set(ids), {}
    with DATA.open() as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                raw = json.loads(line)
                selected[index] = {"context": raw["context"], "target": raw["target"] if targets else ""}
    if set(selected) != wanted:
        raise ExperimentStopped("Missing selected data rows")
    processed = DataProcessor("formula").process_task_data([selected[index] for index in ids])
    return [{"sample_id": index, "question": row["question"], "context": row["context"],
             **({"target": row["target"]} if targets else {})} for index, row in zip(ids, processed)]


def book_for(contract, variant, seed):
    book = contract["books"][f"{variant}-seed{seed}"]
    content = (ROOT / book["path"]).read_bytes()
    if digest(content) != book["sha256"]:
        raise ExperimentStopped("Book content changed")
    return content.decode()


if __name__ == "__main__":
    data = freeze()
    print(json.dumps({"sealed": True, "predictions": data["predictions"],
                      "forecast_cny": data["forecast_plus_30pct_cny"]}, indent=2))
