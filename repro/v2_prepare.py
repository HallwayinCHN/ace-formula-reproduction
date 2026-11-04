"""Seal a new validation-only screen; reserve the other 100 original test IDs without opening them."""

from collections import Counter
from datetime import datetime, timezone
import json
import subprocess

from repro import ROOT
from repro.prepare import ARTIFACTS, FILES, digest
from repro.recovery import safe_json, validate_recovery
from repro.v2_methods import VARIANTS

CONTRACT = ARTIFACTS / "v2_screen_contract.json"
SOURCES = ("repro/v2_methods.py", "repro/v2_prepare.py", "repro/v2_screen.py", "repro/v2_analyze.py")


def diagnose():
    rows = [json.loads(line) for line in (ARTIFACTS / "stage3_scored.jsonl").read_text().splitlines()]
    groups = {}
    for variant in ("base", "ace", "selective", "retrieval", "combined"):
        subset = [row for row in rows if row["variant"] == variant]
        output = sum(row["output_tokens"] for row in subset)
        truncated = sum(row["output_tokens"] for row in subset if row.get("finish_reason") == "length")
        groups[variant] = {"incorrect": sum(not row["is_correct"] for row in subset),
                           "missing_answer": sum(row["final_answer"] == "No final answer found" for row in subset),
                           "truncated": sum(row.get("finish_reason") == "length" for row in subset),
                           "output_tokens": output, "truncated_output_token_fraction": truncated/output}
    by_key = {(row["variant"], row["seed"], row["sample_id"]): row for row in rows}
    changes = {}
    for variant in ("selective", "retrieval", "combined"):
        pairs = [(row, by_key[("ace", row["seed"], row["sample_id"])])
                 for row in rows if row["variant"] == variant]
        changes[variant] = {"gained": sum(a["is_correct"] and not b["is_correct"] for a, b in pairs),
                            "lost": sum(not a["is_correct"] and b["is_correct"] for a, b in pairs)}
    result = {"status": "posthoc_diagnostic_only", "groups": groups, "discordant_pairs_vs_ace": changes,
              "interpretation": "Truncation and missing answers are observed associations, not proven causes. "
                                "Changing the prompt can also affect correctly answered questions."}
    safe_json(ARTIFACTS / "v2_diagnostics.json", result)
    return result


def freeze():
    validate_recovery()
    if CONTRACT.exists():
        verify()
        return
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    prior_val = set(manifest["selected_indices"]["validation"]) | set(manifest["splits"]["validation"]["pilot_indices"])
    available = [index for index in manifest["splits"]["validation"]["formal_order"] if index not in prior_val]
    old_test = set(manifest["selected_indices"]["test"])
    new_test = [index for index in manifest["splits"]["test"]["formal_order"] if index not in old_test]
    assert len(new_test) == 100 and len(available) == 260
    # Hashing only: no new test context or target is loaded, shown or scored here.
    assert digest((ROOT / "eval/finance/data" / FILES["test"]).read_bytes()) == manifest["split_hashes"]["test"]
    contract = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exploratory accuracy-first inference development; not a fresh test result",
        "variants": VARIANTS, "seeds": [42, 43, 44], "screen_validation_ids": available[:20],
        "confirmation_validation_ids": available[20:60], "reserved_unopened_test_ids": new_test,
        "screen_predictions": 300, "screen_limit_cny": 20, "project_limit_cny": 180,
        "selection": "Candidate must match or exceed original screen correct count and reduce cached accounted "
                     "cost >=10% or output tokens >=25%; rank by correct count, then cost, then name. "
                     "No qualifying candidate means no automatic confirmation or held-out testing.",
        "confirmation": "Rerun selected method and original on the other 40 validation IDs with three books; "
                        "these remain development results. No automatic opening of the reserved test IDs.",
        "guard": "Retry original full-book prompt once only if extracted answer is nonnumeric or nonfinite; "
                 "never inspect target or retry because an answer is wrong",
        "context_guard": {"core": "all non-formula entries", "top_formula_entries": 6,
                          "token_budget": 4300, "weak_match_threshold": .08,
                          "dependency_closure": True, "overflow_or_weak_match": "full_book"},
        "brief_reasoning_words": 60, "model_and_sampling": "same frozen pilot config; output cap stays 4096",
        "file_hashes": {path: digest((ROOT / path).read_bytes()) for path in SOURCES},
        "validation_sha256": manifest["split_hashes"]["validation"],
        "original_stage3_analysis_sha256": digest((ARTIFACTS / "stage3_analysis.json").read_bytes()),
        "books_manifest_sha256": digest((ARTIFACTS / "frozen_books.json").read_bytes()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    safe_json(CONTRACT, contract)
    print(json.dumps({"sealed": True, "screen_predictions": 300, "unused_test_ids_reserved": len(new_test)}))


def verify():
    validate_recovery()
    contract = json.loads(CONTRACT.read_text())
    for path, expected in contract["file_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"V2 screening source changed: {path}")
    if digest((ARTIFACTS / "stage3_analysis.json").read_bytes()) != contract["original_stage3_analysis_sha256"]:
        raise ValueError("Original results changed")
    if digest((ARTIFACTS / "frozen_books.json").read_bytes()) != contract["books_manifest_sha256"]:
        raise ValueError("Trained books changed")
    return contract


def confirmation_candidate():
    verify()
    choice = json.loads((ARTIFACTS / "v2_confirmation_lock.json").read_text())
    if choice["screen_contract_sha256"] != digest(CONTRACT.read_bytes()):
        raise ValueError("Confirmation contract differs from screening")
    for path, expected in choice["screen_prediction_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError("Screening predictions changed after method selection")
    return choice["candidate"]


if __name__ == "__main__":
    diagnose()
    freeze()
