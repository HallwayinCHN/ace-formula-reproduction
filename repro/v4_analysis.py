"""Unblind only after six complete jobs, then run the predeclared paired fixed-book analysis."""

import json
import statistics

import numpy as np

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.holdout_stats import paired_interval, standardized_cost
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json
from repro.transport import CostLedger, ExperimentStopped
from repro.v2_methods import numeric_answer
from repro.v4_protocol import CONTRACT, read_rows, verify

STATE = ROOT / "results/repro"


def collect():
    contract = verify()
    jobs, hashes = [], {}
    ids = contract["evaluation_ids"]
    for variant in contract["variants"]:
        for seed in contract["seeds"]:
            folder = STATE / f"v4-{variant}-seed{seed}"
            if not (folder / "summary.json").exists():
                raise ExperimentStopped("All six V4 conditions must complete before loading scoring labels")
            summary = json.loads((folder / "summary.json").read_text())
            raw = (folder / "predictions.json").read_bytes()
            if (summary["state"] != "completed" or summary["items"] != len(ids)
                    or summary["contract_sha256"] != digest(CONTRACT.read_bytes())
                    or summary["book_sha256"] != contract["books"][f"{variant}-seed{seed}"]["sha256"]
                    or summary["targets_supplied_to_agent"] is not False):
                raise ExperimentStopped("V4 condition metadata changed or incomplete")
            if digest(raw) != summary["predictions_sha256"]:
                raise ExperimentStopped("V4 predictions changed")
            items = json.loads(raw)
            if len(items) != len(ids) or {row["sample_id"] for row in items} != set(ids):
                raise ExperimentStopped("V4 prediction cohort changed")
            jobs.append((variant, seed, items))
            for name in ("summary.json", "predictions.json"):
                path = folder / name
                hashes[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    all_requests = [item["request_id"] for _, _, items in jobs for item in items]
    if len(set(all_requests)) != contract["predictions"]:
        raise ExperimentStopped("Duplicate or missing V4 response IDs")
    lock = {"contract_sha256": digest(CONTRACT.read_bytes()), "complete_job_hashes": hashes,
            "prediction_count": len(all_requests), "labels_loaded_only_after_all_hash_checks": True}
    lock_path = ARTIFACTS / "v4_prediction_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ExperimentStopped("Refusing to change the unblinding lock")
    safe_json(lock_path, lock)
    # No scoring target is supplied to the scorer until all six jobs and their hashes were checked.
    targets = {row["sample_id"]: row["target"] for row in read_rows(ids, targets=True)}
    processor = DataProcessor("formula")
    return [{**item, "variant": variant, "seed": seed, "target": targets[item["sample_id"]],
             "is_correct": processor.answer_is_correct(item["final_answer"], targets[item["sample_id"]])}
            for variant, seed, items in jobs for item in items]


def compute(rows, usage, contract, training):
    ids, seeds = contract["evaluation_ids"], contract["seeds"]
    expected = {(variant, seed, index) for variant in contract["variants"] for seed in seeds for index in ids}
    if len(rows) != contract["predictions"] or {(r["variant"], r["seed"], r["sample_id"]) for r in rows} != expected:
        raise ValueError("Incomplete fixed denominator")
    groups, matrices = {}, {}
    for variant in contract["variants"]:
        subset = [row for row in rows if row["variant"] == variant]
        matrix = np.zeros((len(ids), 3))
        for row in subset:
            matrix[ids.index(row["sample_id"]), seeds.index(row["seed"])] = row["is_correct"]
        matrices[variant] = matrix
        prompt = sum(row["prompt_tokens"] for row in subset)
        output = sum(row["output_tokens"] for row in subset)
        records = [row for row in usage if row["run_id"].startswith(f"v4-{variant}-seed")]
        inference_cost = sum(row["accounted_cny"] for row in records)
        train_cost = training["groups"][variant]["all_workflow_accounted_cny"]
        durations = [row["end_to_end_seconds"] for row in subset if row["end_to_end_seconds"] is not None]
        groups[variant] = {
            "accuracy": float(matrix.mean()), "correct": int(matrix.sum()), "predictions": len(subset),
            "seed_accuracy": matrix.mean(axis=0).tolist(),
            "accuracy_sd_across_three_books": statistics.stdev(matrix.mean(axis=0).tolist()),
            "inference_accounted_cny": inference_cost,
            "inference_accounted_cny_per_100": inference_cost/len(subset)*100,
            "standardized_cny_per_100": standardized_cost(prompt, output, contract["inference_config"])
                                        /len(subset)*100,
            "prompt_tokens": prompt, "output_tokens": output,
            "cache_hit_ratio": sum(row["cached_tokens"] for row in subset)/prompt,
            "truncated_predictions": sum(row["length_response"] for row in subset),
            "finite_numeric_output_rate": statistics.mean(numeric_answer(row["final_answer"]) for row in subset),
            "missing_answers": sum(row["final_answer"] == "No final answer found" for row in subset),
            "api_median_seconds": statistics.median(row["api_seconds"] for row in subset),
            "api_p95_seconds": float(np.quantile([row["api_seconds"] for row in subset], .95)),
            "end_to_end_measured_predictions": len(durations),
            "end_to_end_median_seconds": statistics.median(durations) if durations else None,
            "end_to_end_p95_seconds": float(np.quantile(durations, .95)) if durations else None,
            "historical_v3_training_plus_validation_cny": train_cost,
            "assembled_cost_three_books_plus_160_answers_each_cny": train_cost+inference_cost,
        }
    comparison = paired_interval(matrices["nodiag"], matrices["original"])
    limits = contract["operational_accuracy_target"]
    point_ok = comparison["delta_percentage_points"] >= limits["minimum_delta_pp"]
    interval_ok = comparison["ci95_percentage_points"][0] >= limits["minimum_ci95_lower_pp"]
    return {"status": "fixed_six_book_evaluation_complete", "origin": contract["origin"],
            "questions": len(ids), "predictions": len(rows), "groups": groups, "primary_comparison": comparison,
            "seed_delta_pp": ((matrices["nodiag"]-matrices["original"]).mean(axis=0)*100).tolist(),
            "point_target_passed": point_ok, "interval_target_passed": interval_ok,
            "operational_accuracy_target_passed": point_ok and interval_ok,
            "formal_noninferiority_proved": False,
            "scope": "Comparison of six fixed V3-selected books, not causal proof of diagnostic removal. "
                     "Historical training plus fresh inference costs are assembled estimates, not a continuous bill."}


def analyze():
    rows = collect()
    contract = verify()
    usage = [row for row in CostLedger(STATE / "live_usage.jsonl", "v4-analysis",
                                      contract["inference_config"]).records() if row["run_id"].startswith("v4-")]
    if any(row["state"] == "pending" for row in usage):
        raise ExperimentStopped("Pending V4 billing reservation requires reconciliation")
    accepted = {row["request_id"] for row in usage if row["state"] == "completed"}
    if accepted != {row["request_id"] for row in rows}:
        raise ExperimentStopped("Prediction IDs and completed API usage differ")
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    result = compute(rows, usage, contract, training)
    safe_json(ARTIFACTS / "v4_analysis.json", result)
    with (ARTIFACTS / "v4_scored.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    return result


if __name__ == "__main__":
    analyze()
