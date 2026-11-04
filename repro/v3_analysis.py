"""Predeclared grouped development selection, sealed evaluation and live-training accounting."""

import json
import statistics

import numpy as np

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.holdout_stats import paired_interval, standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json
from repro.transport import CostLedger, ExperimentStopped
from repro.v2_methods import numeric_answer
from repro.v3_protocol import CONTRACT, RULE, selected_rows, verify
from repro.v3_router import SUPPORTS, choose, fit, profile

STATE = ROOT / "results/repro"


def collect(stage):
    contract = verify()
    variants = contract[f"{stage}_variants"]
    ids = contract["dev_ids" if stage == "dev" else "evaluation_ids"]
    jobs = []
    for variant in variants:
        for seed in (42, 43, 44):
            output = STATE / f"v3-{stage}-{variant}-seed{seed}"
            summary = json.loads((output / "summary.json").read_text())
            raw = (output / "predictions.json").read_bytes()
            if summary["state"] != "completed" or summary["items"] != len(ids):
                raise ExperimentStopped("All conditions must complete before scoring")
            if summary["contract_sha256"] != digest(CONTRACT.read_bytes()):
                raise ExperimentStopped("Inference contracts differ")
            if digest(raw) != summary["predictions_sha256"]:
                raise ExperimentStopped("Predictions changed")
            if stage == "eval" and summary["router_lock_sha256"] != digest(RULE.read_bytes()):
                raise ExperimentStopped("Evaluation router changed")
            items = json.loads(raw)
            if len(items) != len(ids) or {row["sample_id"] for row in items} != set(ids):
                raise ExperimentStopped("Wrong question cohort")
            jobs.append((variant, seed, items))
    # Labels are loaded only after every completion/hash/cohort check succeeded.
    targets = {row["sample_id"]: row for row in selected_rows(ids, targets=True)}
    rows = []
    processor = DataProcessor("formula")
    for variant, seed, items in jobs:
        for item in items:
            target = targets[item["sample_id"]]["target"]
            rows.append({**item, "variant": variant, "seed": seed, "target": target,
                         "is_correct": processor.answer_is_correct(item["final_answer"], target)})
    return rows, targets


def simulated(pairs, decisions):
    total = len(pairs)*3
    correct = sum(sum(pair["brief_correct"] if decisions[pair["sample_id"]] else pair["original_correct"])
                  for pair in pairs)
    cost = sum(sum(pair["brief_cost"] if decisions[pair["sample_id"]] else pair["original_cost"])
               for pair in pairs)
    original_correct = sum(sum(pair["original_correct"]) for pair in pairs)
    original_cost = sum(sum(pair["original_cost"]) for pair in pairs)
    delta = (correct-original_correct)/total*100
    reduction = 100*(1-cost/original_cost) if original_cost else 0
    return {"correct": correct, "total": total, "accuracy": correct/total,
            "delta_pp": delta, "estimated_cached_cost_cny": cost, "estimated_cost_reduction_percent": reduction,
            "brief_questions": sum(decisions[pair["sample_id"]] for pair in pairs),
            "passed": delta >= -1 and reduction >= 15}


def select_router():
    rows, targets = collect("dev")
    ids = verify()["dev_ids"]
    lookup = {(row["variant"], row["seed"], row["sample_id"]): row for row in rows}
    pairs = []
    for index in ids:
        pairs.append({"sample_id": index, "question": targets[index]["question"],
                      "profile": profile(targets[index]["question"]),
                      "original_correct": [lookup[("original", seed, index)]["is_correct"] for seed in (42, 43, 44)],
                      "brief_correct": [lookup[("brief", seed, index)]["is_correct"] for seed in (42, 43, 44)],
                      "original_cost": [lookup[("original", seed, index)]["accounted_cny"] for seed in (42, 43, 44)],
                      "brief_cost": [lookup[("brief", seed, index)]["accounted_cny"] for seed in (42, 43, 44)]})
    permutation = np.random.default_rng(2026).permutation(len(pairs))
    folds = {pairs[index]["sample_id"]: rank % 4 for rank, index in enumerate(permutation)}
    candidates = []
    for support in SUPPORTS:
        decisions, fold_rules = {}, []
        for fold in range(4):
            training = [pair for pair in pairs if folds[pair["sample_id"]] != fold]
            validation = [pair for pair in pairs if folds[pair["sample_id"]] == fold]
            rule = fit(training, support)
            fold_rules.append({"fold": fold, "rule": rule,
                               "fit_ids": [pair["sample_id"] for pair in training],
                               "validation_ids": [pair["sample_id"] for pair in validation]})
            decisions.update({pair["sample_id"]: choose(pair["question"], rule)[0] for pair in validation})
        cv = simulated(pairs, decisions)
        rule = fit(pairs, support)
        all_decisions = {pair["sample_id"]: choose(pair["question"], rule)[0] for pair in pairs}
        full = simulated(pairs, all_decisions)
        candidates.append({"support": support, "cv": cv, "full_fit": full, "rule": rule,
                           "folds": fold_rules, "passed": cv["passed"] and full["passed"]})
    passed = sorted([item for item in candidates if item["passed"]],
                    key=lambda item: (-item["cv"]["correct"], item["cv"]["estimated_cached_cost_cny"],
                                      -item["support"]))
    winner = passed[0] if passed else None
    lock = {"passed": bool(winner), "rule": winner["rule"] if winner else None,
            "contract_sha256": digest(CONTRACT.read_bytes()), "candidates": candidates,
            "dev_prediction_hashes": {str(path.relative_to(ROOT)): digest(path.read_bytes())
                                       for path in STATE.glob("v3-dev-*/predictions.json")},
            "evaluation_loaded": False,
            "cost_proxy_limit": "Mixed routing is simulated from separate dev streams; "
                                "cache history can differ at evaluation"}
    if RULE.exists() and json.loads(RULE.read_text()) != lock:
        raise ExperimentStopped("Refusing to change a locked router")
    safe_json(RULE, lock)
    safe_json(ARTIFACTS / "v3_dev_summary.json", {"status": "development_only", "gate_passed": bool(winner),
                                                "selected": winner, "candidates": candidates,
                                                "scored_predictions": len(rows), "evaluation_loaded": False})
    save_rows("v3_dev_scored.jsonl", rows)
    return lock


def save_rows(name, rows):
    with (ARTIFACTS / name).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")


def summarize_evaluation():
    rows, _ = collect("eval")
    contract, config = verify(), load_config()
    ids = contract["evaluation_ids"]
    groups, matrices = {}, {}
    for variant in contract["eval_variants"]:
        subset = [row for row in rows if row["variant"] == variant]
        matrix = np.zeros((len(ids), 3))
        for row in subset:
            matrix[ids.index(row["sample_id"]), row["seed"]-42] = row["is_correct"]
        matrices[variant] = matrix
        accuracies = matrix.mean(axis=0).tolist()
        prompt = sum(row["prompt_tokens"] for row in subset)
        output = sum(row["output_tokens"] for row in subset)
        groups[variant] = {
            "accuracy": statistics.mean(accuracies), "accuracy_sd": statistics.stdev(accuracies),
            "seed_accuracy": accuracies, "correct": int(matrix.sum()), "total": len(subset),
            "accounted_cny_per_100": sum(row["accounted_cny"] for row in subset)/len(subset)*100,
            "standardized_cny_per_100": standardized_cost(prompt, output, config)/len(subset)*100,
            "output_tokens": output, "prompt_tokens": prompt,
            "cache_hit_ratio": sum(row["cached_tokens"] for row in subset)/prompt,
            "brief_fraction": sum(row["brief"] for row in subset)/len(subset),
            "truncated": sum(row["length_response"] for row in subset),
            "missing": sum(row["final_answer"] == "No final answer found" for row in subset),
            "finite_numeric_output_rate": statistics.mean(numeric_answer(row["final_answer"]) for row in subset),
            "api_median_seconds": statistics.median(row["api_seconds"] for row in subset),
            "api_p95_seconds": float(np.quantile([row["api_seconds"] for row in subset], .95)),
            "end_to_end_median_seconds": statistics.median(row["end_to_end_seconds"] for row in subset),
            "end_to_end_p95_seconds": float(np.quantile([row["end_to_end_seconds"] for row in subset], .95)),
            "routing_mean_seconds": statistics.mean(row["routing_seconds"] for row in subset),
        }
    comparisons = {variant: paired_interval(matrix, matrices["original"]) for variant, matrix in matrices.items()
                   if variant != "original"}
    reduction = 100*(1-groups["adaptive"]["accounted_cny_per_100"]/groups["original"]["accounted_cny_per_100"])
    report = {"status": "sealed_evaluation_complete", "origin": contract["evaluation_origin"],
              "questions": len(ids), "predictions": len(rows), "groups": groups, "comparisons": comparisons,
              "primary": "adaptive_minus_original", "cached_cost_reduction_percent": reduction,
              "point_goal_met": comparisons["adaptive"]["delta_percentage_points"] >= -1 and reduction >= 15,
              "formal_noninferiority_proved": False}
    safe_json(ARTIFACTS / "v3_eval_analysis.json", report)
    save_rows("v3_eval_scored.jsonl", rows)
    return report


def training_checks(seeds):
    results = []
    for seed in seeds:
        pair = {}
        for variant in ("original", "nodiag"):
            directory = STATE / f"v3-train-{variant}-seed{seed}"
            summary = json.loads((directory / "summary.json").read_text())
            traces = json.loads((directory / "trace.json").read_text())
            if summary["steps"] != 60 or len(traces) != 60 or len(summary["checkpoints"]) != 2:
                raise ExperimentStopped("Training pilot has missing steps or checkpoints")
            if any(row["total"] != 30 or row["errors"] for row in summary["checkpoints"]):
                raise ExperimentStopped("Training validation denominator changed")
            expected = 60 if variant == "original" else 0
            if summary["diagnostic_calls"] != expected:
                raise ExperimentStopped("Diagnostic call count differs from the registered execution strategy")
            if variant == "nodiag" and any(row["post_train_result"]["is_correct"] is not None for row in traces):
                raise ExperimentStopped("Skipped diagnostic accuracy was fabricated")
            pair[variant] = summary
        results.append({"seed": seed, **pair})
    return results


def summarize_training():
    pairs = training_checks((42, 43, 44))
    ledger = CostLedger(STATE / "live_usage.jsonl", "v3-train-analysis", load_config()).records()
    groups = {}
    for variant in ("original", "nodiag"):
        runs = [pair[variant] for pair in pairs]
        records = [row for row in ledger if row["run_id"].startswith(f"v3-train-{variant}-seed")]
        accepted = [row for row in records if row["state"] == "completed"]
        learning = [row for row in accepted if row["call_id"].startswith("train_")]
        groups[variant] = {"learning_calls": sum(row["learning_calls"] for row in runs),
                           "diagnostic_calls": sum(row["diagnostic_calls"] for row in runs),
                           "learning_accounted_cny": sum(row["learning_accounted_cny"] for row in runs),
                           "learning_standardized_cny": sum(row["learning_standardized_cny"] for row in runs),
                           "all_workflow_accounted_cny": sum(row["accounted_cny"] for row in runs),
                           "wall_seconds_per_seed": [row["wall_seconds"] for row in runs],
                           "api_seconds_per_seed": [row["api_seconds"] for row in runs],
                           "validation_best_per_seed": [row["best_validation_accuracy"] for row in runs],
                           "learning_cache_hit_ratio": sum(row.get("cached_tokens", 0) for row in learning)
                           /sum(row["prompt_tokens"] for row in learning),
                           "accepted_validation_calls": len(accepted)-len(learning),
                           "retained_learning_calls": sum(not row["call_id"].endswith("_post_curate")
                                                          for row in learning)}
    report = {"status": "three_live_pairs_complete", "groups": groups, "pairs": pairs,
              "scope": "Fresh model responses can differ; compare live costs separately from fixed-response replay. "
                       "Validation selects books and is not an independent accuracy test."}
    safe_json(ARTIFACTS / "v3_training_analysis.json", report)
    return report
