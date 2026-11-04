"""Offline accounting detail and development-only failure analysis after V3 completion."""

from collections import Counter
import json
import shutil
import statistics

from repro import ROOT
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json
from repro.transport import CostLedger
from repro.v2_methods import numeric_answer
from repro.v3_protocol import CONTRACT, RULE, verify


def usage(rows):
    accepted = [row for row in rows if row["state"] == "completed"]
    prompt = sum(row["prompt_tokens"] for row in accepted)
    return {"accepted_calls": len(accepted), "states": dict(Counter(row["state"] for row in rows)),
            "prompt_tokens": prompt, "output_tokens": sum(row["completion_tokens"] for row in accepted),
            "cached_tokens": sum(row.get("cached_tokens", 0) for row in accepted),
            "cached_input_ratio": sum(row.get("cached_tokens", 0) for row in accepted)/prompt if prompt else 0,
            "accounted_cny": sum(row["accounted_cny"] for row in rows),
            "api_seconds": sum(row.get("elapsed_seconds", 0) for row in rows),
            "truncated_calls": sum(row.get("response_finish_reason") == "length" for row in accepted)}


def main():
    verify()
    completion = json.loads((ARTIFACTS / "v3_completion.json").read_text())
    if not completion["status"].startswith("completed_"):
        raise ValueError("V3 must be complete before supplemental analysis")
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v3-supplement", load_config()).records()
    public = ARTIFACTS / "v3_training"
    public.mkdir(exist_ok=True)
    exports = {}
    for variant in ("original", "nodiag"):
        for seed in (42, 43, 44):
            folder = ROOT / "results/repro" / f"v3-train-{variant}-seed{seed}"
            for filename in ("summary.json", "trace.json", "best_playbook.txt"):
                target = public / f"{variant}-seed{seed}-{filename}"
                shutil.copyfile(folder / filename, target)
                exports[str(target.relative_to(ROOT))] = digest(target.read_bytes())
    fields = ("request_id", "run_id", "role", "call_id", "state", "accounted_cny", "prompt_tokens",
              "completion_tokens", "cached_tokens", "elapsed_seconds", "response_model", "response_finish_reason",
              "status_code", "error_type", "logical_index", "timestamp")
    usage_path = ARTIFACTS / "v3_usage.jsonl"
    with usage_path.open("w") as handle:
        for row in ledger:
            if row["run_id"].startswith("v3-"):
                handle.write(json.dumps({key: row[key] for key in fields if key in row})+"\n")
    exports[str(usage_path.relative_to(ROOT))] = digest(usage_path.read_bytes())
    for name in ("v3_dev_scored.jsonl", "v3_router_lock.json", "v3_training_analysis.json", "v3_completion.json"):
        path = ARTIFACTS / name
        exports[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    safe_json(ARTIFACTS / "v3_exports.json", {"sha256": exports, "contains_secrets": False,
                                             "contract_sha256": digest(CONTRACT.read_bytes())})
    details = {}
    for variant in ("original", "nodiag"):
        rows = [row for row in ledger if row["run_id"].startswith(f"v3-train-{variant}-seed")]
        learning = [row for row in rows if row["call_id"].startswith("train_")]
        details[variant] = {
            "learning": usage(learning),
            "validation": usage([row for row in rows if row["call_id"].startswith("test_eval_")]),
            "diagnostics": usage([row for row in learning if row["call_id"].endswith("_post_curate")]),
            "retained_by_role": {role: usage([row for row in learning if row["role"] == role
                                             and not row["call_id"].endswith("_post_curate")])
                                 for role in sorted({row["role"] for row in learning})},
        }
    scored = [json.loads(line) for line in (ARTIFACTS / "v3_dev_scored.jsonl").read_text().splitlines()]
    dev = {}
    for variant in ("original", "brief"):
        rows = [row for row in scored if row["variant"] == variant]
        dev[variant] = {"correct": sum(row["is_correct"] for row in rows), "total": len(rows),
                        "accuracy": statistics.mean(row["is_correct"] for row in rows),
                        "finite_numeric_output_rate": statistics.mean(numeric_answer(row["final_answer"])
                                                                       for row in rows),
                        "output_tokens": sum(row["output_tokens"] for row in rows),
                        "accounted_cny": sum(row["accounted_cny"] for row in rows)}
    lookup = {(row["variant"], row["seed"], row["sample_id"]): row for row in scored}
    transitions = []
    for row in scored:
        if row["variant"] != "original":
            continue
        other = lookup[("brief", row["seed"], row["sample_id"])]
        if row["is_correct"] != other["is_correct"]:
            transitions.append({"sample_id": row["sample_id"], "seed": row["seed"],
                                "transition": "harm" if row["is_correct"] else "rescue",
                                "target": row["target"], "original_answer": row["final_answer"],
                                "brief_answer": other["final_answer"]})
    # Evidence is development-only; a failed gate does not permit loading sealed evaluation inputs.
    result = {"scope": "Post-completion descriptive analysis; no router changes or new model requests",
              "learning_usage": details, "development_groups": dev,
              "development_correctness_transitions": transitions,
              "contract_sha256": digest(CONTRACT.read_bytes()), "router_lock_sha256": digest(RULE.read_bytes())}
    safe_json(ARTIFACTS / "v3_supplement.json", result)
    report = ARTIFACTS / "V3_RESULTS.md"
    text = ["", "## 开发集观察与学习用量明细", "",
            f"开发集原始输出 {dev['original']['correct']}/{dev['original']['total']}，"
            f"短输出 {dev['brief']['correct']}/{dev['brief']['total']}。"
            "这些数据用于挑选路由，不能解释为新方案的独立准确率提升。", "",
            "| 学习用量（三次合计） | 原始训练 | 去诊断训练 |", "|---|---:|---:|"]
    for title, key in (("输入 tokens", "prompt_tokens"), ("输出 tokens", "output_tokens"),
                       ("缓存命中输入 tokens", "cached_tokens"), ("截断调用数", "truncated_calls")):
        text.append(f"| {title} | {details['original']['learning'][key]} | {details['nodiag']['learning'][key]} |")
    text += ["", "原始条件中诊断调用的可观测开销为 "
             f"{details['original']['diagnostics']['accepted_calls']} 次、"
             f"{details['original']['diagnostics']['accounted_cny']:.4f} 元。"
             "这是原始运行的费用归因；不能假定删掉后其他请求的缓存、响应和耗时完全不变。", "",
             "逐题开发正确性变化与按角色/验证拆分的 tokens、缓存、费用和 API 时间保存在 `v3_supplement.json`。",
             "`v3_training/` 保留六次训练的摘要、逐步轨迹与选定手册；`v3_usage.jsonl` 仅导出调用元数据，"
             "不含密钥或请求头。`v3_recompute` 可使用这些公开本地产物离线核对主结果，无需私有响应缓存。",
             "图表：`v3_results.png` / `v3_results.svg`。", ""]
    marker = "\n## 开发集观察与学习用量明细\n"
    report.write_text(report.read_text().split(marker, 1)[0]+"\n".join(text))


if __name__ == "__main__":
    main()
