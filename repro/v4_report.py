"""Audit every actual V4 prompt, export key-free evidence and report the registered outcome."""

from collections import Counter
import json

from ace.prompts.generator import GENERATOR_PROMPT
from repro import ROOT
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger
from repro.v4_protocol import CONTRACT, book_for, read_rows, verify
from utils import extract_answer

STATE = ROOT / "results/repro"


def audit_prompts(contract):
    inputs = {row["sample_id"]: row for row in read_rows(contract["evaluation_ids"])}
    jobs = []
    for variant in contract["variants"]:
        for seed in contract["seeds"]:
            folder = STATE / f"v4-{variant}-seed{seed}"
            infos = {path.stem: path for path in folder.glob("attempts/**/*.json")}
            items = json.loads((folder / "predictions.json").read_text())
            book = book_for(contract, variant, seed)
            for item in items:
                info = json.loads(infos[item["request_id"]].read_text())
                sample = inputs[item["sample_id"]]
                expected = GENERATOR_PROMPT.format(book, "(empty)", sample["question"], sample["context"])
                if info["prompt"] != expected or extract_answer(info["response"]) != item["final_answer"]:
                    raise ValueError("Actual V4 prompt or extracted answer differs from registered inference")
            jobs.append({"variant": variant, "seed": seed, "exact_prompts_verified": len(items)})
    audit = {"status": "all_actual_prompts_and_answers_verified", "jobs": jobs,
             "total_exact_prompts": sum(row["exact_prompts_verified"] for row in jobs),
             "prompt_contains_only": "original template, fixed book, empty reflection, selected question and context",
             "targets_added_to_model_prompt": False, "prior_bulk_parse_deviation": contract["prior_boundary_deviation"]}
    safe_json(ARTIFACTS / "v4_audit.json", audit)


def finish():
    contract = verify()
    data = json.loads((ARTIFACTS / "v4_analysis.json").read_text())
    if data["status"] != "fixed_six_book_evaluation_complete" or data["predictions"] != 960:
        raise ValueError("Require complete registered evaluation")
    audit_prompts(contract)
    ledger = CostLedger(STATE / "live_usage.jsonl", "v4-report", contract["inference_config"]).records()
    usage = [row for row in ledger if row["run_id"].startswith("v4-")]
    if any(row["state"] == "pending" for row in usage):
        raise ValueError("Pending request at finalization")
    completion = {"status": "complete", "finished_at_utc": utcnow(),
                  "phase_accounted_cny": sum(row["accounted_cny"] for row in usage),
                  "project_accounted_cny": sum(row["accounted_cny"] for row in ledger),
                  "request_states": dict(Counter(row["state"] for row in usage)),
                  "response_models": dict(Counter(row.get("response_model") for row in usage
                                                    if row["state"] == "completed")),
                  "truncated_responses": sum(row.get("response_finish_reason") == "length" for row in usage),
                  "operational_accuracy_target_passed": data["operational_accuracy_target_passed"],
                  "contract_sha256": digest(CONTRACT.read_bytes())}
    safe_json(ARTIFACTS / "v4_completion.json", completion)
    fields = ("request_id", "run_id", "role", "call_id", "state", "accounted_cny", "prompt_tokens",
              "completion_tokens", "cached_tokens", "elapsed_seconds", "response_model", "response_finish_reason",
              "status_code", "error_type", "logical_index", "timestamp")
    with (ARTIFACTS / "v4_usage.jsonl").open("w") as handle:
        for row in usage:
            handle.write(json.dumps({key: row[key] for key in fields if key in row})+"\n")
    paths = ["v4_contract.json", "v4_analysis.json", "v4_scored.jsonl", "v4_prediction_lock.json", "v4_audit.json",
             "v4_completion.json", "v4_usage.jsonl", "v3_training_analysis.json"]
    hashes = {str((ARTIFACTS / name).relative_to(ROOT)): digest((ARTIFACTS / name).read_bytes()) for name in paths}
    hashes.update({book["path"]: book["sha256"] for book in contract["books"].values()})
    safe_json(ARTIFACTS / "v4_exports.json", {"sha256": hashes, "contains_api_keys_or_headers": False})
    comparison = data["primary_comparison"]
    lower, upper = comparison["ci95_percentage_points"]
    text = ["# V4：去诊断训练手册的项目留出评估", "",
            "六份手册由 V3 的原 30 题验证集提前选定，分别来自原始训练和去诊断训练，各三个顺序种子。"
            "本轮不重新训练、不替换手册，使用相同原始提示、完整手册和模型参数。", "",
            "160 个验证池留出题 × 两组 × 三份手册 = 960 条预测。六个条件全部完成并锁定哈希后才统一评分。"
            "独立的问题单位是 160 个，不能将 480 条组内回答解释为 480 个独立题目。", "",
            "| 条件 | 正确/回答数 | 准确率 | 三份手册准确率 | 核算费用/100回答 |",
            "|---|---:|---:|---|---:|"]
    for variant, group in data["groups"].items():
        seeds = " / ".join(f"{value*100:.2f}%" for value in group["seed_accuracy"])
        text.append(f"| {variant} | {group['correct']}/{group['predictions']} | {group['accuracy']*100:.2f}% | "
                    f"{seeds} | {group['inference_accounted_cny_per_100']:.4f} 元 |")
    text += ["", f"主要差值 nodiag−original：**{comparison['delta_percentage_points']:+.2f} pp**，"
             f"按题聚类的 95% bootstrap 区间 **[{lower:+.2f}, {upper:+.2f}] pp**。"
             "随机种子 2026，重复 10000 次，每次保留题目对应的全部三个手册配对。", "",
             "预设操作性标准：点估计至少 −1 pp，且区间下界至少 −2 pp。"
             f"本轮**{'达到' if data['operational_accuracy_target_passed'] else '未达到'}**该标准。"
             "这不是正式非劣效证明，也不能保证任意模型、手册或数据上的准确率。", "",
             "## 推理用量与总流程费用口径", "",
             "| 指标 | original | nodiag |", "|---|---:|---:|"]
    metrics = (("输入 tokens", "prompt_tokens", ".0f"), ("输出 tokens", "output_tokens", ".0f"),
               ("缓存命中比例", "cache_hit_ratio", ".3f"), ("有限数值输出率", "finite_numeric_output_rate", ".3f"),
               ("截断回答数", "truncated_predictions", ".0f"), ("API中位秒数", "api_median_seconds", ".3f"),
               ("API P95秒数", "api_p95_seconds", ".3f"),
               ("推理核算/100回答（元）", "inference_accounted_cny_per_100", ".4f"),
               ("统一无缓存成本/100回答（元）", "standardized_cny_per_100", ".4f"),
               ("历史训练含验证＋本次三本各答160题（元）", "assembled_cost_three_books_plus_160_answers_each_cny", ".4f"))
    for title, key, fmt in metrics:
        text.append(f"| {title} | {data['groups']['original'][key]:{fmt}} | {data['groups']['nodiag'][key]:{fmt}} |")
    text += ["", "总流程费用是 V3 的训练及选模验证核算，加上本次推理核算的组合；"
             "不是一次连续部署的实付账单。训练节省与推理节省不能直接相加。", "",
             "## 边界与局限", "",
             "这些题来自原验证池，不是新的官方测试集。V3 旧训练加载器曾整文件解析验证文件，"
             "但只选原 30 条验证记录供模型使用；该实现偏差继续披露。"
             "V4 按物理行 ID 选择后才 JSON 解析，推理输入不含 target 字段；"
             "判分标签在全部条件完成后才提供给判分程序。960 条真实提示与原始模板、指定手册及题目逐字审计。", "",
             "手册生成过程存在在线响应差异，因此这是六份既定产物的准确率对照，"
             "不能把观察到的所有差异都归因于跳过诊断。置信区间也不覆盖任意重新训练手册的全部不确定性。"
             "本轮结果无论通过或证据不足，都不追加改规则或挑题重测。", "",
             f"阶段核算 {completion['phase_accounted_cny']:.4f} 元，全项目累计 "
             f"{completion['project_accounted_cny']:.4f} 元；请求状态 {completion['request_states']}，"
             f"非空截断 {completion['truncated_responses']} 条，按冻结上游行为计入固定分母。"
             "采用已核对的官方峰时美元价及冻结的 factor8 记账系数，核算含未知计费预留，不等于账单实付。", "",
             "公开逐题结果、去密钥用量和导出哈希分别为 `v4_scored.jsonl`、`v4_usage.jsonl`、`v4_exports.json`。"
             "离线复算：`.venv/bin/python -m repro.v4_recompute`；图表：`v4_results.png` / `v4_results.svg`。", ""]
    (ARTIFACTS / "V4_RESULTS.md").write_text("\n".join(text))


if __name__ == "__main__":
    finish()
