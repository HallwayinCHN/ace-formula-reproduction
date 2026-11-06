"""Read-only V4 progress, without reading questions, answers or scores."""

from collections import Counter
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from repro import ROOT
from repro.pilot import load_config
from repro.transport import CostLedger


def display(interval=5):
    state = ROOT / "results/repro"
    status = json.loads((state/"v4-batch-status.json").read_text())
    ledger = CostLedger(state/"live_usage.jsonl", "v4-monitor", load_config()).records()
    phase = [row for row in ledger if row["run_id"].startswith("v4-")]
    counts = Counter(row["state"] for row in phase)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    names = {"starting": "启动", "checking": "只读检测服务", "running": "运行中", "waiting": "等待定时恢复",
             "blocked": "停止，需检查原因", "stopped": "已停止", "reporting": "统一评分与生成报告", "complete": "已完成"}
    lines = ["ACE V4 · 独立准确率验证", f"北京时间 {now:%Y-%m-%d %H:%M:%S}｜每 {interval:g} 秒刷新", "",
             f"状态：{names.get(status['state'], status['state'])}"]
    age = (now-datetime.fromisoformat(status["heartbeat_at_utc"])).total_seconds()
    if status["state"] not in ("complete", "blocked", "stopped") and age > 90:
        lines.append(f"心跳已 {age:.0f} 秒未更新，运行状态未确认。")
    total = 0
    for variant in ("original", "nodiag"):
        parts = []
        for seed in (42, 43, 44):
            folder = state/f"v4-{variant}-seed{seed}"
            count = len(list(folder.glob("items/*.json")))
            total += count
            parts.append(f"{seed}: {count}/160"+(" 完成" if (folder/"summary.json").exists() else ""))
        lines.append(f"{variant}: "+" | ".join(parts))
    lines += [f"预测合计：{total}/960", "",
              f"本阶段费用核算：{sum(row['accounted_cny'] for row in phase):.3f}/10 元",
              f"全项目费用核算：{sum(row['accounted_cny'] for row in ledger):.3f}/180 元",
              f"完成请求 {counts['completed']}｜在途 {counts['pending']}｜失败计费待核对 {counts['failed_billing_unknown']}"]
    if status.get("next_check_at_utc"):
        deadline = datetime.fromisoformat(status["next_check_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai"))
        lines.append(f"下次检测：{deadline:%H:%M:%S}")
    if status.get("reason"):
        lines.append("停止原因："+status["reason"])
    return "\n".join(lines+["", "含失败及在途预留，非已核对账单。", "只读监控，不调用 API、不显示成绩；Ctrl+C只退出窗口。"])
