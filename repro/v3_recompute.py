"""Verify exported V3 accounting, learning results and dev accuracy without private logs or an API key."""

import json
import math
import statistics

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.holdout_stats import standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest
from repro.v3_protocol import selected_rows


def close(a, b):
    if not math.isclose(a, b, abs_tol=1e-9):
        raise ValueError(f"Recomputed result differs: {a} vs {b}")


def main():
    manifest = json.loads((ARTIFACTS / "v3_exports.json").read_text())
    contract = ARTIFACTS / "v3_contract.json"
    if digest(contract.read_bytes()) != manifest["contract_sha256"]:
        raise ValueError("V3 contract changed")
    for path, expected in manifest["sha256"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Export changed: {path}")
    rows = [json.loads(line) for line in (ARTIFACTS / "v3_usage.jsonl").read_text().splitlines()]
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Usage export contains duplicate request states")
    completion = json.loads((ARTIFACTS / "v3_completion.json").read_text())
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    close(sum(row["accounted_cny"] for row in rows), completion["phase_accounted_cny"])
    close(completion["project_accounted_cny"], json.loads(contract.read_text())["spent_before_phase_cny"]
          + completion["phase_accounted_cny"])
    original, nodiag = training["groups"]["original"], training["groups"]["nodiag"]
    for key, field in (("learning_calls", "learning_calls"), ("learning_accounted_cost", "learning_accounted_cny"),
                       ("learning_standardized_cost", "learning_standardized_cny")):
        close(100*(1-nodiag[field]/original[field]), completion["training_reductions_percent"][key])
    close(100*(1-statistics.median(nodiag["wall_seconds_per_seed"])
               /statistics.median(original["wall_seconds_per_seed"])),
          completion["training_reductions_percent"]["median_workflow_wall_time"])
    for variant in ("original", "nodiag"):
        learning = [row for row in rows if row["run_id"].startswith(f"v3-train-{variant}-seed")
                    and row["call_id"].startswith("train_")]
        accepted = [row for row in learning if row["state"] == "completed"]
        group = training["groups"][variant]
        close(len(accepted), group["learning_calls"])
        close(sum(row["call_id"].endswith("_post_curate") for row in accepted), group["diagnostic_calls"])
        close(sum(row["accounted_cny"] for row in learning), group["learning_accounted_cny"])
        close(standardized_cost(sum(row["prompt_tokens"] for row in accepted),
                                sum(row["completion_tokens"] for row in accepted), load_config()),
              group["learning_standardized_cny"])
        for seed in (42, 43, 44):
            prefix = ARTIFACTS / "v3_training" / f"{variant}-seed{seed}-"
            summary = json.loads(prefix.with_name(prefix.name+"summary.json").read_text())
            trace = json.loads(prefix.with_name(prefix.name+"trace.json").read_text())
            book = prefix.with_name(prefix.name+"best_playbook.txt")
            if len(trace) != 60 or digest(book.read_bytes()) != summary["best_book_sha256"]:
                raise ValueError("Exported trace or selected book inconsistent")
            checkpoints = summary["checkpoints"]
            if [row["step"] for row in checkpoints] != [30, 60]:
                raise ValueError("Validation schedule changed")
            for row in checkpoints:
                if row["total"] != 30 or row["book_sha256"] != trace[row["step"]-1]["book_sha256"]:
                    raise ValueError("Validation cohort or book trajectory inconsistent")
            winner = max(checkpoints, key=lambda row: row["accuracy"])
            close(winner["accuracy"], summary["best_validation_accuracy"])
            if winner["book_sha256"] != summary["best_book_sha256"]:
                raise ValueError("Selected book is not the earliest best validation checkpoint")
            if variant == "nodiag" and any(row["post_train_result"]["is_correct"] is not None for row in trace):
                raise ValueError("Skipped accuracy was not null")
    dev = [json.loads(line) for line in (ARTIFACTS / "v3_dev_scored.jsonl").read_text().splitlines()]
    processor = DataProcessor("formula")
    if len(dev) != 240:
        raise ValueError("Incomplete dev predictions")
    registered = json.loads(contract.read_text())
    raw_validation = ROOT / "eval/finance/data" / FILES["validation"]
    if digest(raw_validation.read_bytes()) != registered["validation_sha256"]:
        raise ValueError("Original validation data changed")
    targets = {row["sample_id"]: row["target"] for row in selected_rows(registered["dev_ids"], targets=True)}
    cohorts = {(row["variant"], row["seed"], row["sample_id"]) for row in dev}
    expected = {(variant, seed, index) for variant in ("original", "brief") for seed in (42, 43, 44)
                for index in registered["dev_ids"]}
    if cohorts != expected:
        raise ValueError("Wrong development cohort")
    for row in dev:
        if row["target"] != targets[row["sample_id"]]:
            raise ValueError("Stored development target differs from original data")
        if processor.answer_is_correct(row["final_answer"], row["target"]) != row["is_correct"]:
            raise ValueError("Public development score differs")
    print(json.dumps({"verified": True, "api_calls": 0, "requires_private_responses": False,
                      "exported_usage_records": len(rows), "development_scores": len(dev),
                      "phase_accounted_cny": completion["phase_accounted_cny"]}, indent=2))


if __name__ == "__main__":
    main()
