"""Whitelist every completed live learning prompt without loading sealed evaluation questions."""

import json
import random
import re

from eval.finance.data_processor import DataProcessor
from repro import ROOT
from repro.prepare import ARTIFACTS, FILES, digest
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger
from repro.pilot import load_config
from repro.v3_protocol import CONTRACT, verify

STATE = ROOT / "results/repro"


def allowed_data(split, ids):
    selected = {}
    wanted = set(ids)
    for index, line in enumerate((ROOT / "eval/finance/data" / FILES[split]).open()):
        if index in wanted:
            selected[index] = json.loads(line)
    return DataProcessor("formula").process_task_data([selected[index] for index in ids])


def audit():
    verify()
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    validation_ids = manifest["selected_indices"]["validation"]
    validations = allowed_data("validation", validation_ids)
    ledger = CostLedger(STATE / "live_usage.jsonl", "v3-boundary-audit", load_config()).records()
    jobs = []
    for variant in ("original", "nodiag"):
        for seed in (42, 43, 44):
            folder = STATE / f"v3-train-{variant}-seed{seed}"
            summary = json.loads((folder / "summary.json").read_text())
            if summary["contract_sha256"] != digest(CONTRACT.read_bytes()):
                raise ValueError("Training contract changed")
            if summary["best_book_sha256"] != digest((folder / "best_playbook.txt").read_bytes()):
                raise ValueError("Selected learning book changed")
            train_ids = list(manifest["selected_indices"]["train"])
            random.Random(seed).shuffle(train_ids)
            training = allowed_data("train", train_ids)
            rows = [row for row in ledger if row["run_id"] == folder.name and row["state"] == "completed"]
            files = {path.stem: path for path in folder.glob("attempts/**/*.json")}
            checked = {"generator": 0, "reflector": 0, "curator": 0}
            for row in rows:
                if row["request_id"] not in files:
                    raise ValueError("Missing durable API prompt")
                info = json.loads(files[row["request_id"]].read_text())
                prompt = info["prompt"]
                if row["call_id"].startswith("test_eval_"):
                    sample = validations[int(row["call_id"].removeprefix("test_eval_"))]
                else:
                    step = int(re.match(r"train_e_1_s_(\d+)", row["call_id"])[1])
                    sample = training[step-1]
                if row["role"] == "generator":
                    if f"**Question:**\n{sample['question']}\n\n**Context:**\n{sample['context']}" not in prompt:
                        raise ValueError("Generator question/context differs from the registered 60/30 cohort")
                elif row["role"] == "reflector":
                    if f"**Question:**\n{sample['question']}\n\n**Model's Reasoning Trace:**" not in prompt:
                        raise ValueError("Reflector question outside the registered training order")
                    if f"**Ground Truth Answer:**\n{sample['target']}\n\n**Environment Feedback:**" not in prompt:
                        raise ValueError("Reflector ground truth differs from the registered training item")
                elif row["role"] == "curator":
                    if f"**Question Context:**\n{sample['context']}\n" not in prompt:
                        raise ValueError("Curator context differs from the registered training item")
                else:
                    raise ValueError("Unexpected learning role")
                checked[row["role"]] += 1
            jobs.append({"variant": variant, "seed": seed, "accepted_prompts_checked": len(rows),
                         "checked_by_role": checked, "train_ids": train_ids, "validation_ids": validation_ids,
                         "best_book_hash_verified": True})
    result = {"status": "all_live_learning_prompts_match_registered_60_30", "audited_at_utc": utcnow(),
              "jobs": jobs, "new_evaluation_questions_used_by_learning_agents": False,
              "sealed_evaluation_inputs_loaded_by_this_audit": False,
              "implementation_deviation": "Frozen formal_data bulk-parses the entire validation JSONL "
                                          "before selecting old 30 validation records. Raw file parsing occurred; "
                                          "no sealed question was selected for these learning requests.",
              "limitation": "Whitelist checks the explicit question/context/ground-truth fields; "
                            "it is not a proof about all possible semantics of model-generated text.",
              "contract_sha256": digest(CONTRACT.read_bytes())}
    safe_json(ARTIFACTS / "v3_boundary_audit.json", result)
    print(json.dumps({"status": result["status"], "accepted_prompts": sum(
        row["accepted_prompts_checked"] for row in jobs)}, indent=2))


if __name__ == "__main__":
    audit()
