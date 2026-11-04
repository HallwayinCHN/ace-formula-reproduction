"""Read-only V3 progress without loading predictions, evaluation questions or labels."""

from collections import Counter
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from repro import ROOT
from repro.pilot import load_config
from repro.transport import CostLedger

STATE = ROOT / "results/repro"


def display(interval=5):
    status = json.loads((STATE / "v3-batch-status.json").read_text())
    ledger = CostLedger(STATE / "live_usage.jsonl", "v3-monitor", load_config()).records()
    phase = [row for row in ledger if row["run_id"].startswith("v3-")]
    states = Counter(row["state"] for row in phase)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    labels = {"starting": "启动", "checking": "检查服务", "running": "运行中", "waiting": "等待定时恢复",
              "blocked": "停止，需检查原因", "stopped": "已停止", "complete": "实验与报告完成", "reporting": "生成报告"}
    lines = ["ACE V3 实时进度", f"北京时间 {now:%Y-%m-%d %H:%M:%S}｜每 {interval:g} 秒刷新", "",
             f"状态：{labels.get(status['state'], status['state'])}", f"当前任务：{status.get('label', '—')}"]
    for stage, variants, count in (("dev", ("original", "brief"), 40),
                                   ("train", ("original", "nodiag"), 60),
                                   ("eval", ("original", "brief", "adaptive"), 160)):
        if stage == "eval" and not status.get("router_gate_passed"):
            lines.append("封存评估：未启用；需路由门槛通过")
            continue
        title = {"dev": "开发集预测", "train": "真实训练步数", "eval": "封存集预测"}[stage]
        for variant in variants:
            counts = []
            for seed in (42, 43, 44):
                folder = STATE / f"v3-{stage}-{variant}-seed{seed}"
                amount = len(list(folder.glob("steps/*.json" if stage == "train" else "items/*.json")))
                done = "完成" if (folder / "summary.json").exists() else ""
                counts.append(f"{seed}: {amount}/{count}{done}")
            lines.append(f"{title} {variant}: " + " | ".join(counts))
    if status.get("next_check_at_utc"):
        next_check = datetime.fromisoformat(status["next_check_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai"))
        lines.append(f"下次检测：{next_check:%H:%M:%S}")
    if status.get("reason"):
        lines.append(f"停止原因：{status['reason']}")
    lines += ["", f"V3 费用核算：{sum(row['accounted_cny'] for row in phase):.3f} / 30 元",
              f"全项目费用核算：{sum(row['accounted_cny'] for row in ledger):.3f} / 180 元",
              f"V3 已完成请求 {states['completed']}｜在途 {states['pending']}｜失败待核对 {states['failed_billing_unknown']}",
              "费用含失败和在途预留，非已核对账单。",
              "只读窗口，不调用 API，不显示评估成绩。Ctrl+C 只关闭本窗口。"]
    return "\n".join(lines)
