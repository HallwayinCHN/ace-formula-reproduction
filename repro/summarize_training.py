"""Freeze the six selected books and report training/validation evidence before any held-out evaluation."""

import json
import statistics
import shutil

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
                          "path": f"repro/artifacts/books/{variant}-seed{seed}.txt",
                          "source_path": str((output / "best_playbook.txt").relative_to(ROOT)),
                          "sha256": digest(raw)})
    return rows, books


def freeze_books(books):
    for book in books:
        target = ROOT / book["path"]
        if target.exists() and digest(target.read_bytes()) != book["sha256"]:
            raise ValueError("Refusing to overwrite an already frozen playbook")
    for book in books:
        target = ROOT / book["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / book["source_path"], target)


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
            "learning_without_diagnostic_calls": sum(row["learning_without_diagnostic_calls"] for row in selected),
            "learning_without_diagnostic_full_miss_cny": sum(
                row["learning_without_diagnostic_full_miss_cny"] for row in selected),
            "all_calls_accounted_cny": sum(row["accounted_cny"] for row in selected),
            "skipped_correct": sum(row["skipped_correct"] for row in selected),
        }
    return groups


def main():
    rows, books = collect()
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "report", load_config())
    all_records = ledger.records()
    for row in rows:
        requests = [r for r in all_records if r["run_id"] == f"formal-{row['variant']}-seed{row['seed']}"]
        row["cached_input_tokens"] = sum(r.get("cached_tokens", 0) for r in requests)
        row["cache_hit_fraction"] = row["cached_input_tokens"] / max(1, row["prompt_tokens"])
        row["unknown_billing_requests"] = sum(r["state"] in ("pending", "failed_billing_unknown") for r in requests)
        row["rejected_402_requests"] = sum(r["state"] == "rejected_402" for r in requests)
        row["learning_request_attempts"] = row["learning_calls"]
        row["learning_calls"] = sum(r["state"] == "completed" and r["call_id"].startswith("train_") for r in requests)
        row["validation_calls"] = sum(r["state"] == "completed" and r["call_id"].startswith("test_") for r in requests)
        row["learning_full_cache_miss_cny"] += sum(
            r["accounted_cny"] for r in requests if r["state"] != "completed" and r["call_id"].startswith("train_")
        )
        core = [r for r in requests if r["state"] == "completed" and r["call_id"].startswith("train_")
                and not r["call_id"].endswith("_post_curate")]
        config = load_config()
        row["learning_without_diagnostic_calls"] = len(core)
        row["learning_without_diagnostic_full_miss_cny"] = sum(
            (r.get("prompt_tokens", 0) * config["input_usd_per_million"]
             + r.get("completion_tokens", 0) * config["output_usd_per_million"])
            / 1_000_000 * config["usd_to_cny_budget_factor"] for r in core)
    groups = grouped(rows)
    stage2_names = {f"formal-{row['variant']}-seed{row['seed']}" for row in rows}
    stage2_records = [row for row in all_records if row["run_id"] in stage2_names]
    comparison = {
        "validation_accuracy_change_percentage_points": 100 * (
            groups["selective"]["validation_accuracy_mean"] - groups["ace"]["validation_accuracy_mean"]),
        "learning_call_reduction_percent": 100 * (1 - groups["selective"]["learning_calls"]
                                                  / groups["ace"]["learning_calls"]),
        "standardized_learning_cost_reduction_percent": 100 * (
            1 - groups["selective"]["learning_full_cache_miss_cny"] / groups["ace"]["learning_full_cache_miss_cny"]),
        "learning_call_reduction_excluding_diagnostics_percent": 100 * (
            1 - groups["selective"]["learning_without_diagnostic_calls"]
            / groups["ace"]["learning_without_diagnostic_calls"]),
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
    freeze_books(books)
    write_json(ARTIFACTS / "frozen_books.json", {"books": books, "held_out_test_evaluated": False})
    print(json.dumps({"groups": groups, "comparison": comparison,
                      "stage2_accounted_cny": report["stage2_accounted_cny"]}, indent=2))


if __name__ == "__main__":
    main()
