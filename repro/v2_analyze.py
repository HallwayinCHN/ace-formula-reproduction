"""Development-only comparisons; never opens the reserved 100 test questions."""

import argparse
import json
import math
import statistics

import numpy as np

from repro import ROOT
from repro.prepare import ARTIFACTS, digest, load_rows
from repro.pilot import load_config
from repro.recovery import safe_json
from repro.transport import CostLedger
from repro.holdout_stats import paired_interval, standardized_cost
from repro.v2_prepare import CONTRACT, confirmation_candidate, verify


def analyze(stage):
    contract = verify()
    if stage == "screen":
        variants = contract["variants"]
        ids = contract["screen_validation_ids"]
    else:
        selected = confirmation_candidate()
        variants = ["original", selected]
        ids = contract["confirmation_validation_ids"]
    jobs = []
    # Check all prediction files first; targets are only loaded afterward.
    for variant in variants:
        for seed in (42, 43, 44):
            output = ROOT / "results/repro" / f"v2-{stage}-{variant}-seed{seed}"
            summary = json.loads((output / "summary.json").read_text())
            items = json.loads((output / "predictions.json").read_text())
            if summary["state"] != "completed" or len(items) != len(ids) or {x["sample_id"] for x in items} != set(ids):
                raise ValueError("Incomplete development cohort")
            jobs.append((variant, seed, items))
    from eval.finance.data_processor import DataProcessor
    processor = DataProcessor("formula")
    labels = load_rows("validation")
    matrices = {variant: np.zeros((len(ids), 3)) for variant in variants}
    rows = []
    for variant, seed, items in jobs:
        for item in items:
            target = labels[item["sample_id"]]["target"]
            correct = processor.answer_is_correct(item["final_answer"], target)
            matrices[variant][ids.index(item["sample_id"]), seed-42] = correct
            rows.append({**item, "variant": variant, "seed": seed, "target": target, "is_correct": bool(correct)})
    config = load_config()
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v2-analysis", config).records()
    groups = {}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        requests = [r for r in ledger if r["run_id"].startswith(f"v2-{stage}-{variant}-seed")]
        prompt = sum(r.get("prompt_tokens", 0) for r in requests)
        output = sum(r.get("completion_tokens", 0) for r in requests)
        seed_accuracy = matrices[variant].mean(axis=0).tolist()
        groups[variant] = {
            "correct": int(matrices[variant].sum()), "total": len(subset), "seed_accuracy": seed_accuracy,
            "accuracy": statistics.mean(seed_accuracy), "accuracy_sd": statistics.stdev(seed_accuracy),
            "accounted_cny": sum(r["accounted_cny"] for r in requests),
            "standardized_cny": standardized_cost(prompt, output, config),
            "output_tokens": output, "prompt_tokens": prompt,
            "cache_hit_ratio": sum(r.get("cached_tokens", 0) for r in requests)/prompt if prompt else 0,
            "fallback_items": sum(row["fallback_used"] for row in subset),
            "truncated_responses": sum(row["length_responses"] for row in subset),
            "no_answer": sum(row["final_answer"] == "No final answer found" for row in subset),
            "mean_book_tokens": statistics.mean(row["selected_book_tokens"] for row in subset),
            "median_api_seconds": statistics.median(row["api_seconds"] for row in subset),
        }
    original = groups["original"]
    eligible = []
    comparisons = {}
    for variant in variants:
        if variant == "original":
            continue
        group = groups[variant]
        group["cached_cost_reduction_percent"] = 100 * (1-group["accounted_cny"]/original["accounted_cny"])
        group["output_token_reduction_percent"] = 100 * (1-group["output_tokens"]/original["output_tokens"])
        group["screen_gate_met"] = group["correct"] >= original["correct"] and (
            group["cached_cost_reduction_percent"] >= 10 or group["output_token_reduction_percent"] >= 25)
        if group["screen_gate_met"]:
            eligible.append(variant)
        comparisons[variant] = paired_interval(matrices[variant], matrices["original"])
    selected = sorted(eligible, key=lambda name: (-groups[name]["correct"], groups[name]["accounted_cny"], name))
    report = {"stage": stage, "status": "development_only_not_heldout", "questions": len(ids),
              "groups": groups, "comparisons_vs_original": comparisons,
              "selected_candidate": selected[0] if selected else None,
              "reserved_test_loaded": False, "formal_noninferiority_proved": False,
              "v2_accounted_cny": sum(r["accounted_cny"] for r in ledger if r["run_id"].startswith("v2-")),
              "project_accounted_cny": sum(r["accounted_cny"] for r in ledger)}
    safe_json(ARTIFACTS / f"v2_{stage}_analysis.json", report)
    if stage == "screen" and report["selected_candidate"]:
        choice = {"candidate": report["selected_candidate"], "screen_contract_sha256": digest(CONTRACT.read_bytes()),
                  "screen_prediction_hashes": {
                      str(path.relative_to(ROOT)): digest(path.read_bytes())
                      for path in (ROOT / "results/repro").glob("v2-screen-*/predictions.json")}}
        path = ARTIFACTS / "v2_confirmation_lock.json"
        if path.exists() and json.loads(path.read_text()) != choice:
            raise ValueError("Refusing to change the confirmation candidate after screening")
        safe_json(path, choice)
    with (ARTIFACTS / f"v2_{stage}_scored.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    analyze(parser.parse_args().stage)
