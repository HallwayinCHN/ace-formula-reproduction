"""Freeze the six selected books and report training/validation evidence before any held-out evaluation."""

import json
import statistics

from repro import ROOT
from repro.methods import parse_entries
from repro.prepare import ARTIFACTS, digest, write_json
from repro.transport import CostLedger
from repro.pilot import load_config


def collect():
    rows, books = [], []
    for variant in ("ace", "selective"):
        for seed in (42, 43, 44):
            output = ROOT / "results/repro" / f"formal-{variant}-seed{seed}"
            row = json.loads((output / "summary.json").read_text())
            if row["status"] != "completed" or row["mock"] or row["held_out_test_loaded"]:
                raise ValueError("Formal learning job is incomplete or violated test isolation")
            raw = (output / "best_playbook.txt").read_bytes()
            if digest(raw) != row["best_playbook_sha256"]:
                raise ValueError("Selected playbook changed")
            row["entries"] = len(parse_entries(raw.decode()))
            rows.append(row)
            books.append({"variant": variant, "seed": seed,
                          "path": str((output / "best_playbook.txt").relative_to(ROOT)),
                          "sha256": digest(raw)})
    return rows, books


def grouped(rows):
    groups = {}
    for variant in ("ace", "selective"):
        selected = [row for row in rows if row["variant"] == variant]
        accuracy = [row["best_validation_accuracy"] for row in selected]
        groups[variant] = {
            "validation_accuracy_mean": statistics.mean(accuracy),
            "validation_accuracy_sd": statistics.stdev(accuracy),
            "validation_accuracy_variance": statistics.variance(accuracy),
            "learning_calls": sum(row["learning_calls"] for row in selected),
            "learning_accounted_cny": sum(row["learning_accounted_cny"] for row in selected),
            "learning_full_cache_miss_cny": sum(row["learning_full_cache_miss_cny"] for row in selected),
            "all_calls_accounted_cny": sum(row["accounted_cny"] for row in selected),
            "skipped_correct": sum(row["skipped_correct"] for row in selected),
        }
    return groups


def main():
    rows, books = collect()
    groups = grouped(rows)
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "report", load_config())
    all_records = ledger.records()
    stage2_names = {f"formal-{row['variant']}-seed{row['seed']}" for row in rows}
    stage2_records = [row for row in all_records if row["run_id"] in stage2_names]
    comparison = {
        "validation_accuracy_change_percentage_points": 100 * (
            groups["selective"]["validation_accuracy_mean"] - groups["ace"]["validation_accuracy_mean"]),
        "learning_call_reduction_percent": 100 * (1 - groups["selective"]["learning_calls"]
                                                  / groups["ace"]["learning_calls"]),
        "standardized_learning_cost_reduction_percent": 100 * (
            1 - groups["selective"]["learning_full_cache_miss_cny"] / groups["ace"]["learning_full_cache_miss_cny"]),
    }
    report = {
        "status": "training_and_validation_complete", "sample_counts": {"train": 60, "validation": 30},
        "groups": groups, "comparison": comparison, "runs": rows,
        "stage2_accounted_cny": sum(row["accounted_cny"] for row in stage2_records),
        "all_stages_accounted_cny": ledger.totals()[0],
        "actual_provider_bill_cny": None, "held_out_test_evaluated": False,
        "retrieval_quality_measured": False,
        "caveat": "Validation selected the books; these are not unbiased held-out performance estimates.",
    }
    write_json(ARTIFACTS / "stage2_analysis.json", report)
    write_json(ARTIFACTS / "frozen_books.json", {"books": books, "held_out_test_evaluated": False})
    print(json.dumps({"groups": groups, "comparison": comparison,
                      "stage2_accounted_cny": report["stage2_accounted_cny"]}, indent=2))


if __name__ == "__main__":
    main()
