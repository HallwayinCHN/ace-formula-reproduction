"""Unblind only after all fifteen frozen inference jobs finish; no model calls."""

import json
import statistics
from datetime import datetime

import numpy as np

from repro import ROOT
from repro.holdout import LOCK, VARIANTS
from repro.holdout_stats import paired_interval, standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest, load_rows, write_json
from repro.transport import CostLedger


def collect_predictions():
    jobs = []
    # Complete the entire check before loading any answer labels.
    for variant in VARIANTS:
        for seed in (42, 43, 44):
            output = ROOT / "results/repro" / f"holdout-{variant}-seed{seed}"
            summary = json.loads((output / "summary.json").read_text())
            if summary["status"] != "completed" or summary["mock"] or summary["items"] != 100:
                raise ValueError("All fifteen real jobs must be complete before unblinding")
            if summary["variant"] != variant or summary["seed"] != seed:
                raise ValueError("Condition labels do not match their frozen job")
            if summary["contract_sha256"] != digest(LOCK.read_bytes()):
                raise ValueError("Inference jobs used different contracts")
            raw = (output / "predictions.json").read_bytes()
            if digest(raw) != summary["predictions_sha256"]:
                raise ValueError("Predictions changed")
            jobs.append((summary, json.loads(raw)))
    return jobs


def score(jobs, ids):
    from eval.finance.data_processor import DataProcessor
    targets = load_rows("test")
    processor = DataProcessor("formula")
    rows, matrices = [], {variant: np.zeros((len(ids), 3)) for variant in VARIANTS}
    index_map = {sample_id: index for index, sample_id in enumerate(ids)}
    for summary, items in jobs:
        if len(items) != len(ids) or {item["sample_id"] for item in items} != set(ids):
            raise ValueError("Prediction IDs do not match the frozen cohort")
        for item in items:
            target = targets[item["sample_id"]]["target"]
            correct = not item["error"] and processor.answer_is_correct(item["final_answer"], target)
            row = {**item, "variant": summary["variant"], "seed": summary["seed"],
                   "target": target, "is_correct": bool(correct)}
            matrices[row["variant"]][index_map[row["sample_id"]], row["seed"] - 42] = correct
            rows.append(row)
    return rows, matrices


def summarize_variants(rows, matrices, ledger, config):
    training = json.loads((ARTIFACTS / "stage2_analysis.json").read_text())["groups"]
    groups = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        requests = [r for r in ledger if r["run_id"] in {f"holdout-{variant}-seed{s}" for s in (42, 43, 44)}]
        seed_accuracy = matrices[variant].mean(axis=0).tolist()
        source = "ace" if variant in ("ace", "retrieval") else "selective"
        learned = training[source] if variant != "base" else None
        prompt_tokens = sum(r.get("prompt_tokens", 0) for r in requests)
        output_tokens = sum(r.get("completion_tokens", 0) for r in requests)
        cold = standardized_cost(prompt_tokens, output_tokens, config)
        accounted = sum(r["accounted_cny"] for r in requests)
        groups[variant] = {
            "seed_accuracy": seed_accuracy, "accuracy_mean": statistics.mean(seed_accuracy),
            "accuracy_sd": statistics.stdev(seed_accuracy), "accuracy_variance": statistics.variance(seed_accuracy),
            "correct": sum(row["is_correct"] for row in subset), "denominator": len(subset),
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "cached_tokens": sum(r.get("cached_tokens", 0) for r in requests),
            "inference_accounted_cny_per_100": accounted / 3,
            "inference_standardized_cny_per_100": cold / 3,
            "lifecycle_accounted_cny_per_100": accounted / 3 + (
                learned["learning_accounted_cny"] / 3 if learned else 0),
            "lifecycle_standardized_cny_per_100": cold / 3 + (
                learned["learning_full_cache_miss_cny"] / 3 if learned else 0),
            "truncated": sum(row.get("finish_reason") == "length" for row in subset),
            "no_answer": sum(row["final_answer"] == "No final answer found" for row in subset),
            "errors": sum(row["error"] is not None for row in subset),
            "latency_reconstructed": sum(row.get("latency_reconstructed", False) for row in subset),
            "end_to_end_seconds_median": float(np.median([r.get("end_to_end_seconds", 0) for r in subset])),
            "end_to_end_seconds_p95": float(np.quantile([r.get("end_to_end_seconds", 0) for r in subset], .95)),
            "retrieval_seconds_mean": statistics.mean(row.get("retrieval_seconds", 0) for row in subset),
            "mean_selected_book_tokens": statistics.mean(row.get("selected_book_tokens", 0) for row in subset),
        }
    return groups


def contrast_cases(rows, matrices, ids):
    differences = (matrices["combined"] - matrices["ace"]).mean(axis=1)
    sources = load_rows("test")
    cases = {}
    for name, sign in (("regressions", 1), ("gains", -1)):
        order = sorted(range(len(ids)), key=lambda i: (sign * differences[i], ids[i]))
        chosen = [i for i in order if sign * differences[i] < 0][:5]
        cases[name] = [{"sample_id": ids[i], "context": sources[ids[i]]["context"],
                        "delta_seed_mean": float(differences[i]),
                        "outputs": [{k: row[k] for k in ("variant", "seed", "final_answer", "target", "is_correct")}
                                    for row in rows if row["sample_id"] == ids[i]]} for i in chosen]
    return cases


def analyze():
    freeze = json.loads(LOCK.read_text())
    for path, expected in freeze["file_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Frozen source or input changed: {path}")
    jobs = collect_predictions()
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    rows, matrices = score(jobs, manifest["selected_indices"]["test"])
    config = load_config()
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "analysis", config).records()
    heldout = [row for row in ledger if row["run_id"].startswith("holdout-")]
    if any(row["role"] != "generator" for row in heldout):
        raise ValueError("A non-generator call occurred during held-out testing")
    if any(datetime.fromisoformat(row["timestamp"]) < datetime.fromisoformat(freeze["created_at_utc"])
           for row in heldout):
        raise ValueError("A held-out model request predates the frozen contract")
    groups = summarize_variants(rows, matrices, ledger, config)
    comparisons = {f"{left}_minus_{right}": paired_interval(matrices[left], matrices[right])
                   for left, right in (("ace", "base"), ("selective", "ace"), ("retrieval", "ace"),
                                       ("combined", "ace"), ("combined", "selective"))}
    primary = comparisons["combined_minus_ace"]
    training = json.loads((ARTIFACTS / "stage2_analysis.json").read_text())
    learning_reduction = training["comparison"]["standardized_learning_cost_reduction_percent"]
    inference_reduction = 100 * (1 - groups["combined"]["inference_standardized_cny_per_100"]
                                  / groups["ace"]["inference_standardized_cny_per_100"])
    report = {"status": "all_frozen_tests_scored", "groups": groups, "comparisons": comparisons,
              "primary_comparison": "combined_minus_ace", "test_questions": 100, "seed_count": 3,
              "logical_predictions": len(rows), "contract_sha256": digest(LOCK.read_bytes()),
              "learning_standardized_cost_reduction_percent": learning_reduction,
              "inference_standardized_cost_reduction_percent": inference_reduction,
              "point_estimate_goal_met": primary["delta_percentage_points"] >= -2
              and (learning_reduction >= 20 or inference_reduction >= 20),
              "formal_noninferiority_test_performed": False,
              "all_stages_accounted_cny": sum(r["accounted_cny"] for r in ledger),
              "stage3_accounted_cny": sum(r["accounted_cny"] for r in ledger if r["run_id"].startswith("holdout-")),
              "unknown_billing_requests": sum(r["state"] in ("pending", "failed_billing_unknown") for r in ledger),
              "actual_provider_bill_cny": None,
              "limitations": ["Different backbone and reduced data versus the paper",
                              "Base is the upstream ACE Generator with an empty book, not a separately rebuilt baseline",
                              "Bootstrap conditional on three trained-book orders, not general model randomness",
                              "Primary labels unchanged; label quality is not established"]}
    write_json(ARTIFACTS / "stage3_analysis.json", report)
    cases = contrast_cases(rows, matrices, manifest["selected_indices"]["test"])
    write_json(ARTIFACTS / "stage3_cases.json", cases)
    with (ARTIFACTS / "stage3_scored.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "groups": groups, "primary": primary}, indent=2))


if __name__ == "__main__":
    analyze()
