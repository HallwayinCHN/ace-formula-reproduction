"""Recompute the entire V4 comparison from public local artifacts, without credentials/private logs."""

import json
import math

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.prepare import ARTIFACTS, digest
from repro.v4_analysis import compute
from repro.v4_protocol import DATA, read_rows


def equal(left, right):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Result fields differ")
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("Result lengths differ")
        for a, b in zip(left, right):
            equal(a, b)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        if not math.isclose(left, right, abs_tol=1e-9):
            raise ValueError("Recomputed numeric result differs")
    elif left != right:
        raise ValueError("Recomputed result differs")


def main():
    manifest = json.loads((ARTIFACTS / "v4_exports.json").read_text())
    for path, expected in manifest["sha256"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Export changed: {path}")
    contract = json.loads((ARTIFACTS / "v4_contract.json").read_text())
    for path, expected in contract["source_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Frozen V4 source changed: {path}")
    if digest(DATA.read_bytes()) != contract["validation_sha256"]:
        raise ValueError("Original dataset changed")
    targets = {row["sample_id"]: row["target"] for row in read_rows(contract["evaluation_ids"], targets=True)}
    rows = [json.loads(line) for line in (ARTIFACTS / "v4_scored.jsonl").read_text().splitlines()]
    processor = DataProcessor("formula")
    for row in rows:
        if row["target"] != targets[row["sample_id"]]:
            raise ValueError("Target differs from original data")
        if row["is_correct"] != processor.answer_is_correct(row["final_answer"], row["target"]):
            raise ValueError("Stored correctness differs")
    usage = [json.loads(line) for line in (ARTIFACTS / "v4_usage.jsonl").read_text().splitlines()]
    if len({r["request_id"] for r in usage}) != len(usage):
        raise ValueError("Duplicated request states")
    if {r["request_id"] for r in rows} != {r["request_id"] for r in usage if r["state"] == "completed"}:
        raise ValueError("Predictions and accepted usage differ")
    by_id = {row["request_id"]: row for row in usage if row["state"] == "completed"}
    for row in rows:
        record = by_id[row["request_id"]]
        for item_key, usage_key in (("prompt_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                                    ("cached_tokens", "cached_tokens"), ("accounted_cny", "accounted_cny"),
                                    ("api_seconds", "elapsed_seconds")):
            equal(row[item_key], record[usage_key])
        if record["run_id"] != f"v4-{row['variant']}-seed{row['seed']}":
            raise ValueError("Response belongs to another condition")
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    result = compute(rows, usage, contract, training)
    equal(result, json.loads((ARTIFACTS / "v4_analysis.json").read_text()))
    completion = json.loads((ARTIFACTS / "v4_completion.json").read_text())
    equal(sum(row["accounted_cny"] for row in usage), completion["phase_accounted_cny"])
    equal(contract["spent_before_phase_cny"]+completion["phase_accounted_cny"], completion["project_accounted_cny"])
    print(json.dumps({"verified": True, "api_calls": 0, "public_predictions": len(rows),
                      "requires_private_logs": False, "primary_comparison": result["primary_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
