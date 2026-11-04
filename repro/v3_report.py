"""Report V3 faithfully even when its development gate prevents opening the sealed evaluation."""

from collections import Counter
import json
import statistics

from repro import ROOT
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger
from repro.v3_protocol import CONTRACT, RULE, verify


def finish():
    contract = verify()
    router = json.loads(RULE.read_text())
    training = json.loads((ARTIFACTS / "v3_training_analysis.json").read_text())
    rows = CostLedger(ROOT / "results/repro/live_usage.jsonl", "v3-report", load_config()).records()
    scoped = [row for row in rows if row["run_id"].startswith("v3-")]
    original, nodiag = training["groups"]["original"], training["groups"]["nodiag"]
    reductions = {"learning_calls": 100*(1-nodiag["learning_calls"]/original["learning_calls"]),
                  "learning_accounted_cost": 100*(1-nodiag["learning_accounted_cny"]
                                                  /original["learning_accounted_cny"]),
                  "learning_standardized_cost": 100*(1-nodiag["learning_standardized_cny"]
                                                     /original["learning_standardized_cny"]),
                  "median_workflow_wall_time": 100*(1-statistics.median(nodiag["wall_seconds_per_seed"])
                                                    /statistics.median(original["wall_seconds_per_seed"]))}
    audit = {"status": "completed_with_evaluation" if router["passed"] else "completed_gate_failed_evaluation_sealed",
             "audited_at_utc": utcnow(), "routing_gate_passed": router["passed"],
             "evaluation_questions_loaded": router["passed"], "evaluation_questions": 160,
             "phase_accounted_cny": sum(row["accounted_cny"] for row in scoped),
             "project_accounted_cny": sum(row["accounted_cny"] for row in rows),
             "request_states": dict(Counter(row["state"] for row in scoped)),
             "training_reductions_percent": reductions, "contract_sha256": digest(CONTRACT.read_bytes()),
             "router_lock_sha256": digest(RULE.read_bytes()),
             "scope": "Live model responses and retained learning trajectories may differ; no live book identity claim"}
    if not router["passed"] and list((ROOT / "results/repro").glob("v3-eval-*/predictions.json")):
        raise ValueError("Evaluation ran despite failed development gate")
    text = ["# 第三轮结果：保守路由与真实训练去诊断", "",
            "## 路由开发门槛", "",
            "新开发集 40 题、三份原始冻结手册、两种输出方式，共 240 条预测。"
            "规则按题 ID 分四折交叉验证，三个手册输出始终留在同一折。", "",
            "| 最小题目支持数 | CV 准确率差 pp | CV 费用变化 | 全开发拟合费用变化 | 门槛 |",
            "|---|---:|---:|---:|---|"]
    for item in router["candidates"]:
        text.append(f"| {item['support']} | {item['cv']['delta_pp']:+.2f} | "
                    f"减少 {item['cv']['estimated_cost_reduction_percent']:.2f}% | "
                    f"减少 {item['full_fit']['estimated_cost_reduction_percent']:.2f}% | "
                    f"{'通过' if item['passed'] else '未通过'} |")
    if router["passed"]:
        evaluation = json.loads((ARTIFACTS / "v3_eval_analysis.json").read_text())
        audit["evaluation"] = evaluation
        text += ["", "开发门槛通过后才打开 160 题项目封存评估集，该集合来自原先未使用的验证池，"
                 "不称为新的官方测试集。三组各用三份原始冻结手册，不使用本轮重训手册。", "",
                 "| 条件 | 准确率 | 含缓存核算/100题 | 短模式比例 |", "|---|---:|---:|---:|"]
        for name, group in evaluation["groups"].items():
            text.append(f"| {name} | {group['accuracy']*100:.2f}% | {group['accounted_cny_per_100']:.4f} 元 | "
                        f"{group['brief_fraction']*100:.1f}% |")
        comparison = evaluation["comparisons"]["adaptive"]
        lo, hi = comparison["ci95_percentage_points"]
        text += ["", f"预指定主要比较 adaptive−original：{comparison['delta_percentage_points']:+.2f} pp，"
                 f"按题聚类 95% 区间 [{lo:+.2f}, {hi:+.2f}] pp；"
                 f"{'达到' if evaluation['point_goal_met'] else '未达到'}预设的点估计效率目标。"
                 "没有进行正式非劣效检验，不能保证其他数据上的准确率。"]
    else:
        text += ["", "**没有候选同时满足开发交叉验证和完整开发拟合的准确率/费用门槛，"
                 "因此未加载或评估封存的 160 题。** 没有降低门槛、继续调路由或只挑开发表现好的题。"
                 "本轮路由没有被证明有可采用的收益；开发数据不能作为独立测试结果。"]
    text += ["", "## 六次真实训练的执行开销", "",
             "训练 60 题，选模验证 30 题；每次在第 30、60 步验证。种子 42、43、44 成对，"
             "基线与去诊断条件交替先后执行。去诊断保留全部反思、修正和经验更新，诊断准确率标为未测量。", "",
             "| 指标（三次合计或注明中位数） | 原始训练 | 去诊断训练 |", "|---|---:|---:|"]
    for title, key, scale in (("学习调用", "learning_calls", "count"),
                              ("其中诊断调用", "diagnostic_calls", "count"),
                              ("保留的学习调用", "retained_learning_calls", "count"),
                              ("含缓存学习核算（元）", "learning_accounted_cny", "cost"),
                              ("统一无缓存学习成本（元）", "learning_standardized_cny", "cost"),
                              ("含验证的全过程核算（元）", "all_workflow_accounted_cny", "cost"),
                              ("学习输入缓存命中比例", "learning_cache_hit_ratio", "ratio")):
        a, b = original[key], nodiag[key]
        formatter = (lambda v: str(v)) if scale == "count" else (
            (lambda v: f"{v*100:.1f}%") if scale == "ratio" else (lambda v: f"{v:.4f}"))
        text.append(f"| {title} | {formatter(a)} | {formatter(b)} |")
    text += [f"| 全过程墙钟中位数（秒） | {statistics.median(original['wall_seconds_per_seed']):.1f} | "
             f"{statistics.median(nodiag['wall_seconds_per_seed']):.1f} |", "",
             "相对变化：" + "；".join(f"{key} 减少 {value:.2f}%" for key, value in reductions.items()) + "。", "",
             "真实调用的保留响应、修正轮数和手册可因服务端随机性而不同，"
             "不能把所有净差都归因于剔除诊断；已单列保留调用数、诊断数和缓存比例。"
             "此前归档回放证明的是固定响应下的等价性，本表测量的是新的真实运行开销。"
             "验证集参与选模，不能当作独立准确率证据；主路由实验不使用这些新手册。", "",
             "## 预算与边界", "",
             f"本阶段核算 {audit['phase_accounted_cny']:.3f} 元，全项目累计 {audit['project_accounted_cny']:.3f} 元。"
             "核算含失败预留、采用冻结峰时单价和保守换算系数，不代表已核对账单。"
             "不存在为了取得好分数而重选旧测试题、修改标签或改变预设门槛的操作。", ""]
    safe_json(ARTIFACTS / "v3_completion.json", audit)
    (ARTIFACTS / "V3_RESULTS.md").write_text("\n".join(text))
