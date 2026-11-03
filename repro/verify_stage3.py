"""Audit completed stage-three artifacts without making model calls."""

from collections import Counter
from datetime import datetime, timezone
import json

from repro import ROOT
from repro.holdout import LOCK, VARIANTS
from repro.holdout_analysis import collect_predictions
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest, verify_upstream, write_json
from repro.transport import CostLedger


def verify():
    upstream = verify_upstream()
    freeze = json.loads(LOCK.read_text())
    for path, expected in freeze["file_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    jobs = collect_predictions()
    rows = [json.loads(line) for line in (ARTIFACTS / "stage3_scored.jsonl").read_text().splitlines()]
    ids = json.loads((ARTIFACTS / "sample_manifest.json").read_text())["selected_indices"]["test"]
    expected = {(variant, seed, sample_id) for variant in VARIANTS for seed in (42, 43, 44) for sample_id in ids}
    keys = [(row["variant"], row["seed"], row["sample_id"]) for row in rows]
    if len(keys) != 1500 or len(set(keys)) != 1500 or set(keys) != expected:
        raise ValueError("Incorrect or duplicated scored cohort")
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "audit", load_config()).records()
    by_request = {row["request_id"]: row for row in ledger}
    for row in rows:
        if row.get("request_id"):
            record = by_request[row["request_id"]]
            if record["run_id"] != f"holdout-{row['variant']}-seed{row['seed']}" or record["state"] != "completed":
                raise ValueError("Scored response does not match a completed physical request")
            if record["prompt_tokens"] != row["prompt_tokens"] or record["completion_tokens"] != row["output_tokens"]:
                raise ValueError("Token accounting differs from saved response")
    heldout = [row for row in ledger if row["run_id"].startswith("holdout-")]
    if {row["role"] for row in heldout} != {"generator"}:
        raise ValueError("Held-out testing made a learning request")
    for summary, _ in jobs:
        state = json.loads((ROOT / "results/repro" / f"holdout-{summary['variant']}-seed{summary['seed']}"
                            / "status.json").read_text())
        if state["labels_used_for_generation"] or state["status"] != "completed":
            raise ValueError("Invalid inference completion or blinding state")
    files = ("stage3_analysis.json", "stage3_scored.jsonl", "stage3_cases.json", "stage3_lock.json")
    audit = {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_rows": len(rows), "distinct_questions": len(ids), "conditions": 15,
        "frozen_hashes_verified": len(freeze["file_hashes"]), "upstream_files_verified": len(upstream),
        "heldout_roles": sorted({row["role"] for row in heldout}),
        "heldout_request_states": dict(Counter(row["state"] for row in heldout)),
        "all_phases_request_states": dict(Counter(row["state"] for row in ledger)),
        "heldout_response_models": dict(Counter(row.get("response_model") for row in heldout
                                                if row["state"] == "completed")),
        "artifact_hashes": {name: digest((ARTIFACTS / name).read_bytes()) for name in files},
        "accounted_cny": sum(row["accounted_cny"] for row in ledger),
        "unknown_billing_requests": sum(row["state"] in ("pending", "failed_billing_unknown") for row in ledger),
    }
    from repro.recovery import CONTRACT, validate_recovery
    if CONTRACT.exists():
        recovery = validate_recovery()
        audit["recovery"] = {"contract_sha256": digest(CONTRACT.read_bytes()),
                             "preserved_predictions_verified": recovery["successful_predictions_before_recovery"],
                             "preserved_files_verified": len(recovery["preserved_files"]),
                             "original_unknown_reserve_cny": recovery["existing_unknown_billing_reserve_cny"],
                             "policy": recovery["policy"]}
    write_json(ARTIFACTS / "stage3_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
