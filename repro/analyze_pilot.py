"""Summarize a completed live pilot and choose formal size using costs, never test accuracy."""

import argparse
from collections import Counter
import json
import statistics

import tiktoken

from repro import ROOT
from repro.prepare import ARTIFACTS, write_json


def expense(input_tokens, output_tokens, config):
    usd = (input_tokens * config["input_usd_per_million"]
           + output_tokens * config["output_usd_per_million"]) / 1_000_000
    return usd * config["usd_to_cny_budget_factor"]


def size_forecasts(observed, config, spent):
    forecasts = []
    for n_train, n_val, n_test in ((100, 40, 200), (60, 30, 100), (30, 20, 60)):
        scale = n_train / observed["train_count"]
        full_playbook = observed["final_playbook_tokens"] * scale * observed["token_ratio"]
        additional_input = observed["mean_playbook_tokens"] * (scale - 1) * observed["token_ratio"]
        train_calls = observed["train_calls"] * scale
        one_train = expense(observed["train_input"] * scale + train_calls * max(0, additional_input),
                            observed["train_output"] * scale, config)
        # Two learned books x three order seeds; assume no unmeasured saving from selective reflection.
        learning = 6 * one_train
        validation = 6 * expense(n_val * (2 * observed["base_input"] + 1.5 * full_playbook),
                                 2 * n_val * observed["generator_output"], config)
        retrieval_book = min(full_playbook, 1500 * observed["token_ratio"])
        # Per seed: base + two full-book variants + two retrieval variants, all on the same test IDs.
        evaluation = 3 * expense(n_test * (5 * observed["base_input"] + 2 * full_playbook + 2 * retrieval_book),
                                 5 * n_test * observed["generator_output"], config)
        forecasts.append({
            "train": n_train, "validation": n_val, "test": n_test,
            "learning_cny": learning, "validation_cny": validation, "test_cny": evaluation,
            "total_including_spent_and_30pct_margin_cny": spent + 1.3 * (learning + validation + evaluation),
        })
    return forecasts


def observation(run):
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if summary["mock"] or summary["status"] != "pilot_completed":
        raise ValueError("A completed real pilot is required")
    logs = [json.loads(path.read_text()) for path in run.rglob("*.json")
            if path.parent.name in ("baseline_calls", "detailed_llm_logs")]
    train = [row for row in logs if row["call_id"].startswith("train_")]
    baseline = [json.loads(path.read_text()) for path in (run / "baseline_calls").glob("*.json")]
    encoder = tiktoken.get_encoding("cl100k_base")
    ratios = [row["prompt_num_tokens"] / max(1, len(encoder.encode(row["prompt"]))) for row in logs]
    trace_path = next((run / "upstream").glob("*/pre_train_post_train_results.json"))
    trace = json.loads(trace_path.read_text())
    final_book = next((run / "upstream").glob("*/final_playbook.txt")).read_text()
    observed = {
        "train_count": summary["train_count"], "train_calls": len(train),
        "train_input": sum(row["prompt_num_tokens"] for row in train),
        "train_output": sum(row["response_num_tokens"] for row in train),
        "base_input": statistics.mean(row["prompt_num_tokens"] for row in baseline),
        "generator_output": statistics.mean(row["response_num_tokens"] for row in logs
                                            if row["role"] == "generator"),
        "token_ratio": max(1.0, max(ratios)),
        "final_playbook_tokens": len(encoder.encode(final_book)),
        "mean_playbook_tokens": statistics.mean(row["pre_train_result"]["playbook_num_tokens"] for row in trace),
    }
    training_log = next((run / "upstream").glob("*/train_results.json"))
    checkpoints = json.loads(training_log.read_text())["results"]
    return summary, config, logs, observed, checkpoints, trace


def analyze(run, lock):
    summary, config, logs, observed, checkpoints, trace = observation(run)
    latest = {}
    ledger = ROOT / "results/repro/live_usage.jsonl"
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        latest[row["request_id"]] = row
    spent = sum(row["accounted_cny"] for row in latest.values())
    forecasts = size_forecasts(observed, config, spent)
    selected = next((row for row in forecasts
                     if row["total_including_spent_and_30pct_margin_cny"] <= config["project_limit_cny"]), None)
    report = {
        "source_pilot": run.name, "model": config["model"], "baseline_validation": summary["baseline"],
        "checkpoint_validation": [{"step": row["step"], **row["val_result"]} for row in checkpoints],
        "best_validation_accuracy": summary["training_results"]["best_validation_accuracy"],
        "initial_training_correct": sum(row["pre_train_result"]["is_correct"] for row in trace),
        "logical_calls": summary["logical_calls"], "request_counts_by_role": dict(Counter(row["role"] for row in logs)),
        "response_models": sorted({row["response_model"] for row in logs}),
        "prompt_tokens": sum(row["prompt_num_tokens"] for row in logs),
        "completion_tokens": sum(row["response_num_tokens"] for row in logs),
        "truncated_responses": sum(row.get("response_finish_reason") == "length" for row in logs),
        "truncations_by_role": dict(Counter(row["role"] for row in logs
                                           if row.get("response_finish_reason") == "length")),
        "api_call_seconds": sum(row["call_time"] for row in logs),
        "pilot_estimated_cny": summary["accounted_run_cny"], "all_runs_accounted_cny": spent,
        "actual_provider_bill_cny": None, "observed_cost_inputs": observed, "size_forecasts": forecasts,
        "selected_size": selected, "held_out_test_used_for_inference": False,
        "forecast_assumptions": ["Peak pricing, no cache discount, FX accounting factor 8",
                                 "Linear playbook growth from 10 training items; this is uncertain",
                                 "Both learning methods cost as much as original ACE, plus 30 percent margin",
                                 "No test accuracy or test labels are used for size choice"],
        "quality_limitations": ["10 validation items and best-of-two checkpoint selection; not a held-out result",
                                "Pilot is a feasibility check, not evidence of method superiority"],
    }
    write_json(ARTIFACTS / "pilot_analysis.json", report)
    if lock:
        if selected is None:
            raise ValueError("No predefined size fits the budget; no formal protocol was changed")
        lock_size(selected, run.name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def lock_size(selected, run_name):
    manifest_path = ARTIFACTS / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    counts = {key: selected[key] for key in ("train", "validation", "test")}
    if manifest.get("formal_size_locked"):
        if manifest.get("selected_counts") != counts:
            raise ValueError("Refusing to change previously frozen formal sample counts")
        return
    manifest.update(formal_size_locked=True, selected_counts=counts, cost_pilot=run_name)
    manifest["selected_indices"] = {
        split: manifest["splits"][split]["formal_order"][:count] for split, count in counts.items()
    }
    protocol_path = ROOT / "repro/configs/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(formal_size_locked=True, selected_counts=counts, cost_pilot=run_name,
                    json_mode=False, allow_truncated_nonempty=True)
    write_json(manifest_path, manifest)
    write_json(protocol_path, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-size", action="store_true")
    args = parser.parse_args()
    candidates = sorted((ROOT / "results/repro").glob("pilot-*/summary.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("No completed live pilot exists")
    analyze(candidates[-1].parent, args.lock_size)


if __name__ == "__main__":
    main()
