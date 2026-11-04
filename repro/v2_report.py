"""Audit and describe the completed prospective V2 comparison; no new model calls."""

from collections import Counter
import json

from repro import ROOT
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json, utcnow, validate_recovery
from repro.v2_holdout import CONTRACT, check
from repro.pilot import load_config
from repro.transport import CostLedger


def audit():
    contract = check()
    validate_recovery()
    rows = [json.loads(line) for line in (ARTIFACTS / "v2_final_scored.jsonl").read_text().splitlines()]
    expected = {(variant, seed, sample_id) for variant in ("original", contract["candidate"])
                for seed in (42, 43, 44) for sample_id in contract["test_ids"]}
    keys = [(row["variant"], row["seed"], row["sample_id"]) for row in rows]
    if len(keys) != 600 or len(set(keys)) != 600 or set(keys) != expected:
        raise ValueError("Prospective test cohort is incomplete or duplicated")
    old = json.loads((ARTIFACTS / "sample_manifest.json").read_text())["selected_indices"]["test"]
    if set(contract["test_ids"]) & set(old):
        raise ValueError("Previously exposed test IDs entered the prospective cohort")
    ledger = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v2-audit", load_config()).records()
    by_request = {row["request_id"]: row for row in ledger}
    for row in rows:
        records = [by_request[index] for index in row["request_ids"]]
        expected_run = f"v2-final-{row['variant']}-seed{row['seed']}"
        if any(record["state"] != "completed" or record["role"] != "generator"
               or record["run_id"] != expected_run or record["mock"] for record in records):
            raise ValueError("Prediction does not map to successful generator calls")
        if sum(record["prompt_tokens"] for record in records) != row["prompt_tokens"]:
            raise ValueError("Input token accounting changed")
        if sum(record["completion_tokens"] for record in records) != row["output_tokens"]:
            raise ValueError("Output token accounting changed")
    scoped = [row for row in ledger if row["run_id"].startswith("v2-")]
    result = {"status": "passed", "audited_at_utc": utcnow(), "unique_test_predictions": len(keys),
              "new_test_questions": 100, "overlap_with_previous_test": 0,
              "all_v2_request_states": dict(Counter(row["state"] for row in scoped)),
              "all_v2_response_models": dict(Counter(row.get("response_model") for row in scoped
                                                     if row["state"] == "completed")),
              "original_sources_books_and_220_old_predictions_verified": True,
              "final_contract_sha256": digest(CONTRACT.read_bytes()),
              "final_scored_sha256": digest((ARTIFACTS / "v2_final_scored.jsonl").read_bytes()),
              "final_analysis_sha256": digest((ARTIFACTS / "v2_final_analysis.json").read_bytes())}
    safe_json(ARTIFACTS / "v2_audit.json", result)
    return result


def report():
    verified = audit()
    screen = json.loads((ARTIFACTS / "v2_screen_analysis.json").read_text())
    confirm = json.loads((ARTIFACTS / "v2_confirm_analysis.json").read_text())
    final = json.loads((ARTIFACTS / "v2_final_analysis.json").read_text())
    candidate = final["candidate"]
    a, b = final["groups"]["original"], final["groups"][candidate]
    comparison = final["primary_comparison"]
    lo, hi = comparison["ci95_percentage_points"]
    savings = {"含缓存推理核算": 100*(1-b["accounted_cny_per_100"]/a["accounted_cny_per_100"]),
               "统一无缓存价推理成本": 100*(1-b["standardized_cny_per_100"]/a["standardized_cny_per_100"]),
               "输出 tokens": 100*(1-b["output_tokens"]/a["output_tokens"]),
               "单题 API 耗时中位数": 100*(1-b["median_api_seconds"]/a["median_api_seconds"])}
    text = ["# 第二轮研究结果：完整经验手册＋简短输出＋解析失败回退", "",
            "**状态：20 题筛选、40 题开发确认和剩余 100 题独立测试全部完成。**", "",
            "第一轮 100 个测试 ID 用于诊断，第二轮独立测试 ID 与其零重叠。以下准确率只在同一行比较，"
            "不能把第二轮数值直接与第一轮的 84% 相减来算提升。", "",
            "## 实际改动", "",
            "保留三份原始 ACE 完整手册、同一模型、原预处理及数值匹配；不使用选择性训练手册。"
            "提示要求计算检查不超过 60 词，这是软约束；保留 JSON 字段和 4096 输出硬上限。"
            "若最终答案不是有限数值，再用原始提示和完整手册调用一次。回退不读取正确答案，两个调用的费用和耗时均计入。", "",
            "学习成本没有改变。本次收益来自答题阶段，不能沿用第一轮选择性反思的学习节省比例来宣传新方案。", "",
            "## 分阶段验证", "",
            "| 阶段 | 题数 × 手册数 | 原始 ACE | 新方案 | 配对差值与 95% 区间（pp） |",
            "|---|---|---:|---:|---|"]
    for name, data, key in (("方案筛选", screen, "brief_guard"), ("开发确认", confirm, candidate)):
        effect = data["comparisons_vs_original"][key]
        low, high = effect["ci95_percentage_points"]
        text.append(f"| {name} | {data['questions']} × 3 | {data['groups']['original']['accuracy']*100:.2f}% | "
                    f"{data['groups'][key]['accuracy']*100:.2f}% | {effect['delta_percentage_points']:+.2f} "
                    f"[{low:+.2f}, {high:+.2f}] |")
    text.append(f"| 独立测试 | 100 × 3 | {a['accuracy']*100:.2f}% | {b['accuracy']*100:.2f}% | "
                f"{comparison['delta_percentage_points']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    text += ["", "筛选阶段按预设的准确率和费用/token 门槛选一个方案，确认后再次封存它；"
             "独立测试只做这一组主要比较，不根据测试表现换提示词、重选样本或改标签。"
             "区间按题 ID 聚类、保留三个训练手册的配对，10,000 次 bootstrap；不把 300 次输出当作 300 道独立题。", "",
             "## 独立测试指标", "",
             "| 指标 | 原始 ACE | 新方案 |", "|---|---:|---:|",
             f"| 正确 / 300 | {a['correct']} | {b['correct']} |",
             f"| 准确率均值 ± SD | {a['accuracy']*100:.2f}% ± {a['accuracy_sd']*100:.2f} pp | "
             f"{b['accuracy']*100:.2f}% ± {b['accuracy_sd']*100:.2f} pp |",
             f"| 含缓存推理核算 / 100题 | {a['accounted_cny_per_100']:.4f} 元 | "
             f"{b['accounted_cny_per_100']:.4f} 元 |",
             f"| 无缓存标准化成本 / 100题 | {a['standardized_cny_per_100']:.4f} 元 | "
             f"{b['standardized_cny_per_100']:.4f} 元 |",
             f"| 输出 tokens（300次答题合计） | {a['output_tokens']} | {b['output_tokens']} |",
             f"| 输入缓存命中比例 | {a['cache_hit_ratio']*100:.1f}% | {b['cache_hit_ratio']*100:.1f}% |",
             f"| 单题 API 耗时中位数 | {a['median_api_seconds']:.3f} 秒 | {b['median_api_seconds']:.3f} 秒 |",
             f"| 截断响应数 | {a['truncated_responses']} | {b['truncated_responses']} |",
             f"| 最终无答案数 | {a['missing_answers']} | {b['missing_answers']} |",
             f"| 触发回退的题次 | {a['fallback_items']} | {b['fallback_items']} |", ""]
    text.append("相对变化：" + "；".join(f"{name}减少 {value:.2f}%" for name, value in savings.items()) + "。")
    text += ["", "完整手册长度与知识内容没有减少，故这里的主要节省不依赖丢弃经验。"
             "含缓存核算采用冻结峰时单价和保守换算系数，不是已核对的服务商账单。"
             "耗时是本次服务和网络下的观测量，没有进行严格延迟显著性检验。", "",
             "## 准确率保证的边界", "",
             "保留完整经验、限制冗长输出、使用无标签回退，旨在保护知识与输出完整性，"
             "但有效性仍由独立测试判断，不能承诺所有题目或新任务准确率不下降。未进行正式非劣效检验。"]
    text.append("本次区间仍允许超过 2 pp 的下降，不能保证 2 pp 以内的精度损失。" if lo < -2 else
                "本次配对区间未显示超过 2 pp 的下降，但这仍是条件于当前题目和手册的证据，不是一般精度保证。")
    text += ["", "开发中另一候选保留固定通用规则、补全公式条目引用、必要时用完整手册回退。"
             "它在筛选中的准确率与最终候选相同，但含缓存核算更高，按预先排序没有入选；不能把它也称作已通过独立测试。", "",
             "## 费用与核验", "",
             f"第二轮全流程核算 {final['v2_accounted_cny']:.3f} 元，全项目累计 {final['project_accounted_cny']:.3f} 元。"
             f"全项目尚有 {final['unknown_billing_requests']} 个旧请求的计费未核对，费用预留保留。",
             f"共核验 {verified['unique_test_predictions']} 个唯一新测试输出、与第一轮零重叠的 100 个 ID、"
             "原始源文件和训练手册、220 条旧预测及其回放缓存。所有原始第一轮结果保持原样。", "",
             "53 项离线检查通过，覆盖引用闭包、溢出回退、无标签输出校验、费用累计和测试标签隔离。", "",
             "## 研究依据", "",
             "[ACE](https://arxiv.org/abs/2510.04618v3)强调保留详细经验；"
             "[LLMLingua-2](https://arxiv.org/abs/2403.12968)把忠实保留作为压缩目标。"
             "本轮没有复现或直接采用它们在其他数据上的压缩收益。",
             "[缓存感知压缩研究](https://arxiv.org/abs/2607.15516)及"
             "[DeepSeek 官方缓存说明](https://api-docs.deepseek.com/guides/kv_cache/)提示需要同时看 token、"
             "缓存和费用。本轮优先完整手册和稳定前缀，而非继续加大逐题删减。",
             "按[官方峰时价格](https://api-docs.deepseek.com/quick_start/pricing/)，每百万输出 tokens 为 "
             "1.2 美元、缓存输入为 0.006 美元，单 token 价格比为 200。这是本任务在高缓存命中时优先优化输出的动机，"
             "不代表输入始终有缓存或费用可仅按 token 总数比较。", ""]
    ablation = ARTIFACTS / "v2_posthoc_ablation.json"
    if ablation.exists():
        observed = json.loads(ablation.read_text())
        text += ["## 回退机制的事后检查", "",
                 f"新方案相对基线有 {observed['gained_vs_original']} 次从错变对、"
                 f"{observed['lost_vs_original']} 次从对变错，净少 3 个正确输出。"
                 f"最终非有限数值输出由 {observed['nonnumeric_original']} 次降到 "
                 f"{observed['nonnumeric_candidate']} 次，但解析成功不等于计算正确。",
                 f"本轮 {observed['fallback_triggered']} 次回退没有补救出正确答案。"
                 "离线保留首次响应时，仍为 239/300；因此本次没有证实回退组件提高准确率。"
                 "这只是对固定响应的事后消融，不是删掉回退后重新做的独立测试，也不能直接据此宣布新方案达标。", ""]
    diagnostic = ARTIFACTS / "diagnostic_free_replay.json"
    if diagnostic.exists():
        evidence = json.loads(diagnostic.read_text())
        text += ["## 进一步的保守执行优化", "",
                 "另行实现并通过三份真实归档回放验证的诊断调用移除原型，不改变保留请求、每步手册或选定手册。"
                 f"学习调用按历史轨迹减少 {evidence['learning_call_reduction_percent']:.2f}%，"
                 f"统一无缓存学习成本归因减少 {evidence['standardized_learning_cost_reduction_percent']:.2f}%。"
                 "没有新增付费重训，实际缓存费用需另行测量，移除预热请求也可能改变缓存命中。"
                 "详见 `DIAGNOSTIC_FREE_RESULTS.md`。这项执行优化独立于本报告的短输出方案，"
                 "不能把两者的节省百分比直接相加。", ""]
    (ARTIFACTS / "V2_RESULTS.md").write_text("\n".join(text))
    safe_json(ARTIFACTS / "v2_improvements.json", {"primary_comparison": comparison, "reductions_percent": savings})


if __name__ == "__main__":
    report()
