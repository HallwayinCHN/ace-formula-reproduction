"""Seal unused validation IDs, a limited router search and a phase-wide budget before paid calls."""

import json
import subprocess

from repro import ROOT
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger, ExperimentStopped
from repro.v2_prepare import verify as verify_v2

CONTRACT = ARTIFACTS / "v3_contract.json"
RULE = ARTIFACTS / "v3_router_lock.json"
SOURCES = ("repro/v3_protocol.py", "repro/v3_router.py", "repro/v3_transport.py", "repro/v3_jobs.py",
           "repro/v3_analysis.py", "repro/v3_batch.py", "repro/diagnostic_free_replay.py")


def freeze():
    verify_v2()
    if CONTRACT.exists():
        return verify()
    old = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    v2 = json.loads((ARTIFACTS / "v2_screen_contract.json").read_text())
    used = (set(old["selected_indices"]["validation"]) | set(old["splits"]["validation"]["pilot_indices"])
            | set(v2["screen_validation_ids"]) | set(v2["confirmation_validation_ids"]))
    available = [index for index in old["splits"]["validation"]["formal_order"] if index not in used]
    if len(available) != 200:
        raise ExperimentStopped("Unexpected fresh validation pool")
    stage2 = json.loads((ARTIFACTS / "stage2_analysis.json").read_text())["groups"]["ace"]
    confirm = json.loads((ARTIFACTS / "v2_confirm_analysis.json").read_text())["groups"]
    final = json.loads((ARTIFACTS / "v2_final_analysis.json").read_text())["groups"]
    # Assume no cached training savings, and adaptive inference costs as much as original.
    train_forecast = 2*stage2["all_calls_accounted_cny"]
    dev_forecast = confirm["original"]["accounted_cny"] + confirm["brief_guard"]["accounted_cny"]
    eval_forecast = 4.8*(2*final["original"]["accounted_cny_per_100"]
                         + final["brief_guard"]["accounted_cny_per_100"])
    forecast = (train_forecast+dev_forecast+eval_forecast)*1.3
    config = load_config()
    spent = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v3-freeze", config).totals()[0]
    if forecast > 30 or forecast+spent > 180:
        raise ExperimentStopped("Empirical forecast with 30% margin exceeds authorized limits")
    contract = {
        "created_at_utc": utcnow(), "dev_ids": available[:40], "evaluation_ids": available[40:],
        "evaluation_origin": "160 unused validation-pool questions, NOT a new official test set",
        "seeds": [42, 43, 44], "dev_variants": ["original", "brief"],
        "eval_variants": ["original", "brief", "adaptive"], "dev_predictions": 240,
        "conditional_eval_predictions": 1440, "learning_train_count": 60, "learning_validation_count": 30,
        "router_support_candidates": [3, 5, 8], "cross_validation": "4 question-grouped folds, RNG2026",
        "bucket_qualification": "minimum distinct questions; zero original-correct/brief-wrong answers in fit data",
        "selection_gate": "out-of-fold accuracy delta >=-1pp AND estimated cached cost reduction >=15%; "
                          "same gates on the full-development fit; choose CV accuracy then CV cost then larger support",
        "gate_failure_action": "do not load the 160 evaluation questions; complete live training checks only",
        "primary_eval": "adaptive minus original; paired question-cluster bootstrap seed2026 10000 repeats",
        "learning_schedule": {"42": ["original", "nodiag"], "43": ["nodiag", "original"],
                              "44": ["original", "nodiag"]},
        "training_pilot_gate": "seed42 pair must finish 60 steps and 2 complete 30-item validations; "
                               "nodiag has zero diagnostic model calls and missing post-train accuracy is null",
        "phase_limit_cny": 30, "project_limit_cny": 180, "failure_exposure_limit_cny": 5,
        "503_cooldown_seconds": 300, "maximum_recovery_rounds": 12,
        "forecast_with_30pct_margin_cny": forecast, "spent_before_phase_cny": spent,
        "forecast_components_cny": {"six_trainings": train_forecast, "development": dev_forecast,
                                     "conditional_evaluation": eval_forecast},
        "validation_sha256": old["split_hashes"]["validation"],
        "source_hashes": {path: digest((ROOT / path).read_bytes()) for path in SOURCES},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reuse": "unchanged original ACE frozen books for routing; fresh training books used only for cost checks",
    }
    safe_json(CONTRACT, contract)
    return contract


def verify():
    verify_v2()
    contract = json.loads(CONTRACT.read_text())
    for path, expected in contract["source_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ExperimentStopped(f"V3 source changed after freeze: {path}")
    path = ROOT / "eval/finance/data" / FILES["validation"]
    if digest(path.read_bytes()) != contract["validation_sha256"]:
        raise ExperimentStopped("Validation data changed")
    return contract


def selected_rows(ids, targets=False):
    from eval.finance.data_processor import DataProcessor
    wanted = set(ids)
    rows = {}
    with (ROOT / "eval/finance/data" / FILES["validation"]).open() as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                raw = json.loads(line)
                rows[index] = {"context": raw["context"], "target": raw["target"] if targets else ""}
    processed = DataProcessor("formula").process_task_data([rows[index] for index in ids])
    return [{"sample_id": index, **row} for index, row in zip(ids, processed)]


if __name__ == "__main__":
    result = freeze()
    print(json.dumps({"sealed": True, "dev": len(result["dev_ids"]), "heldout": len(result["evaluation_ids"]),
                      "forecast_cny": result["forecast_with_30pct_margin_cny"]}, indent=2))
