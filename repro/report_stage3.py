"""Write a compact report from the complete frozen evaluation; no model calls or new tests."""

import json

from repro.prepare import ARTIFACTS

LABELS = {"base": "空手册 Generator", "ace": "原始 ACE", "selective": "选择性反思",
          "retrieval": "ACE＋检索", "combined": "选择性反思＋检索"}


def report():
    data = json.loads((ARTIFACTS / "stage3_analysis.json").read_text())
    audit = json.loads((ARTIFACTS / "stage3_audit.json").read_text())
    if data["status"] != "all_frozen_tests_scored" or audit["status"] != "passed":
        raise ValueError("Require full scoring and audit")
    groups = data["groups"]
    primary = data["comparisons"]["combined_minus_ace"]
    lo, hi = primary["ci95_percentage_points"]
    goal = "达到预设的观测点估计目标" if data["point_estimate_goal_met"] else "未达到预设的观测点估计目标"
    text = ["# 第三阶段结果：冻结手册后的独立盲测（2026-11-03）", "",
            "**状态：五组、三个训练顺序、100 个固定测试 ID，共 1,500 条预测全部完成并统一评分。**", "",
            "沿用 60 条训练、30 条选模验证、100 条独立测试的冻结规模。模型为 deepseek-flash，"
            "关闭思考、temperature=0、4096 输出上限。测试时只调用上游 Generator，不运行反思、整理或学习；"
            "真实标签在预处理前替换为空，不进入检索或生成。六份手册与评估代码在首个测试请求前记录 SHA-256。", "",
            "## 固定测试结果", "",
            "| 方案 | 种子 42 / 43 / 44（每次 100 题） | 平均准确率 ± 样本标准差 | 正确 / 总数 |",
            "|---|---|---|---|"]
    for variant, group in groups.items():
        seeds = " / ".join(f"{value*100:.0f}%" for value in group["seed_accuracy"])
        text.append(f"| {LABELS[variant]} | {seeds} | {group['accuracy_mean']*100:.2f}% ± "
                    f"{group['accuracy_sd']*100:.2f} pp | {group['correct']}/{group['denominator']} |")
    text += ["", f"预先指定的主要比较为组合方案减原始 ACE：**{primary['delta_percentage_points']:+.2f} 个百分点，"
             f"95% 配对区间 [{lo:+.2f}, {hi:+.2f}]**。", "",
             "区间按测试题 ID 聚类 bootstrap，保留每题三个种子的配对，重采样 10,000 次、固定随机种子 2026。"
             "有效题目数为 100，不能把 300 条答题看成 300 道独立题。区间条件于三份已训练手册；"
             "训练顺序种子不控制服务端随机性。", "",
             "| 配对比较 | 准确率差（pp） | 95% 区间（pp） |", "|---|---:|---|",]
    for name, comparison in data["comparisons"].items():
        a, b = comparison["ci95_percentage_points"]
        text.append(f"| {name} | {comparison['delta_percentage_points']:+.2f} | [{a:+.2f}, {b:+.2f}] |")
    text += ["", "其余比较为消融和探索性分析，没有进行多重检验校正，也没有根据测试成绩再调参。", "",
             "## 学习、答题与完整使用成本", "",
             "单位为元；每行代表一份平均训练手册及 100 次答题。学习包含上游训练后诊断调用，"
             "不包含选模验证开销；所有阶段费用另行完整累计。原始 ACE 与检索组共享原始训练成本，"
             "两个选择性组共享选择性训练成本，不重复训练。", "",
             "| 方案 | 答题：含缓存核算 | 答题：统一无缓存价 | 学习＋答题：含缓存核算 | 学习＋答题：统一无缓存价 |",
             "|---|---:|---:|---:|---:|"]
    for variant, group in groups.items():
        text.append(f"| {LABELS[variant]} | {group['inference_accounted_cny_per_100']:.3f} | "
                    f"{group['inference_standardized_cny_per_100']:.3f} | "
                    f"{group['lifecycle_accounted_cny_per_100']:.3f} | "
                    f"{group['lifecycle_standardized_cny_per_100']:.3f} |")
    text += ["", f"选择性反思的标准化学习成本减少 **{data['learning_standardized_cost_reduction_percent']:.2f}%**；"
             f"组合方案相对原始 ACE 的标准化答题成本减少 **{data['inference_standardized_cost_reduction_percent']:.2f}%**。"
             f"本次{goal}（准确率点估计下降不超过 2 pp，学习或答题成本下降至少 20%）。"
             "**这不是正式非劣效结论**，不能用点估计达标或区间包含零来证明准确率不劣。", "",
             f"第三阶段核算费用 **{data['stage3_accounted_cny']:.3f} 元**，全项目累计 "
             f"**{data['all_stages_accounted_cny']:.3f} 元**，低于 180 元项目停止线。"
             "采用冻结的官方峰时美元单价、8 元/美元保守核算系数与 API usage；"
             "不是已对账的服务商实付金额。缓存命中率、条件执行顺序和并行调度均可能影响含缓存费用与延迟。", "",
             "## 输出、失败与延迟", "",
             "| 方案 | 截断 / 300 | 无答案 / 300 | 普通异常 / 300 | 检索后手册平均 tokens | 端到端中位 / P95 秒 |",
             "|---|---:|---:|---:|---:|---|",]
    for variant, group in groups.items():
        text.append(f"| {LABELS[variant]} | {group['truncated']} | {group['no_answer']} | {group['errors']} | "
                    f"{group['mean_selected_book_tokens']:.1f} | {group['end_to_end_seconds_median']:.2f} / "
                    f"{group['end_to_end_seconds_p95']:.2f} |")
    text += ["", "非空截断沿用上游答案解析，保留在固定分母中；无答案和普通异常也没有从分母排除。"
             "延迟为本次网络和服务状态下的观测量，不作跨服务基准结论。"
             "运行期间官方报告过 Flash API 部分中断，证据与时间见 `STAGE3_SERVICE_EVENT.md`；"
             "不能把本次延迟差异全部归因于方法。", "",
             "## 完整性与适用范围", "",
             f"- 离线检查 33 项通过；最终核验 {audit['prediction_rows']} 个唯一方案/种子/题目组合、"
             f"{audit['frozen_hashes_verified']} 个冻结文件哈希、{audit['upstream_files_verified']} 个上游原始文件哈希。",
             f"- 第三阶段请求状态：`{audit['heldout_request_states']}`；"
             f"全阶段结果或计费状态不明请求数为 {audit['unknown_billing_requests']}。"
             "若存在未知计费请求，含缓存核算保留其预算预留；标准化成本只能覆盖已返回 usage 的请求，"
             "不能视为完整已核对账单。",
             f"- 服务端返回模型标识：`{audit['heldout_response_models']}`。",
             "- 原始数据标签与数值判分未修改。探索性案例见 `stage3_cases.json`，含组合方案相对 ACE 的最多五个改善和五个退化题目；"
             "案例在全部评分后选出，不用于修改本次方法或重新筛选测试集。",
             "- 这是更换骨干模型、缩小样本后的行为与效率复现，不能直接声称复现论文 85.5% 或论文的提升幅度。"
             "空手册基线使用上游 ACE Generator，不是另行重建的论文 Base LLM 提示。", "",
             "## 可检查文件与下一阶段", "",
             "- `stage3_lock.json`：首个测试调用前的代码、原始测试集、样本与手册哈希。",
             "- `stage3_scored.jsonl`：全部 1,500 条输出、原标签、判分、tokens、延迟和检索 ID。",
             "- `stage3_analysis.json`、`stage3_audit.json`：完整统计和完整性核验。",
             "- `stage3_accuracy_cost.png` / `.svg`：准确率与完整使用成本、配对差异区间。",
             "- 下一阶段仅整理交付：不超过 10 页报告、可复现的 GitHub 说明和不超过 15 分钟录屏。"
             "本次测试已经开封，后续若新增方法应标为探索性，不能把这批测试重新当作未见数据。", ""]
    (ARTIFACTS / "STAGE3.md").write_text("\n".join(text))


if __name__ == "__main__":
    report()
