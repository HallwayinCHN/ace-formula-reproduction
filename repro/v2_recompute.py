"""Recompute public V2 test metrics from committed rows; no private API logs or model calls required."""

import json

import numpy as np

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.holdout_stats import paired_interval, standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest, load_rows, verify_upstream
from repro.recovery import safe_json


def main():
    verify_upstream()
    contract = json.loads((ARTIFACTS / "v2_final_contract.json").read_text())
    if digest((ROOT / "eval/finance/data" / FILES["test"]).read_bytes()) != contract["test_sha256"]:
        raise ValueError("Original test data changed")
    report = json.loads((ARTIFACTS / "v2_final_analysis.json").read_text())
    rows = [json.loads(line) for line in (ARTIFACTS / "v2_final_scored.jsonl").read_text().splitlines()]
    ids, candidate = contract["test_ids"], contract["candidate"]
    expected = {(variant, seed, sample_id) for variant in ("original", candidate)
                for seed in (42, 43, 44) for sample_id in ids}
    keys = [(row["variant"], row["seed"], row["sample_id"]) for row in rows]
    if len(keys) != 600 or set(keys) != expected:
        raise ValueError("Wrong or duplicated public result rows")
    targets, processor, config = load_rows("test"), DataProcessor("formula"), load_config()
    matrices = {variant: np.zeros((100, 3)) for variant in ("original", candidate)}
    metrics = {}
    for row in rows:
        target = targets[row["sample_id"]]["target"]
        if row["target"] != target:
            raise ValueError("Saved ground truth differs from original label")
        correct = processor.answer_is_correct(row["final_answer"], target)
        if bool(correct) != row["is_correct"]:
            raise ValueError("Saved correctness differs from the upstream scorer")
        matrices[row["variant"]][ids.index(row["sample_id"]), row["seed"]-42] = correct
    for variant in matrices:
        group = [row for row in rows if row["variant"] == variant]
        prompt = sum(row["prompt_tokens"] for row in group)
        output = sum(row["output_tokens"] for row in group)
        cached = sum(row["cached_tokens"] for row in group)
        cost = ((prompt-cached)*config["input_usd_per_million"] + cached*config["cached_input_usd_per_million"]
                + output*config["output_usd_per_million"])/1_000_000*config["usd_to_cny_budget_factor"]/3
        saved = report["groups"][variant]
        if abs(cost-saved["accounted_cny_per_100"]) > 1e-10 or int(matrices[variant].sum()) != saved["correct"]:
            raise ValueError("Recomputed accuracy or cost differs from report")
        metrics[variant] = {"correct": int(matrices[variant].sum()), "accounted_cny_per_100": cost,
                            "standardized_cny_per_100": standardized_cost(prompt, output, config)/3}
    comparison = paired_interval(matrices[candidate], matrices["original"])
    if comparison != report["primary_comparison"]:
        raise ValueError("Paired interval differs from report")
    result = {"status": "public_rows_recompute_passed", "metrics": metrics, "comparison": comparison,
              "private_logs_required": False, "new_api_calls": 0}
    safe_json(ARTIFACTS / "v2_public_recompute.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
