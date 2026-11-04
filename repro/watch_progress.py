"""Read-only terminal window for blinded experiment progress; never call the model."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

from repro import ROOT
from repro.holdout_progress import main as read_progress
from repro.pilot import load_config
from repro.transport import CostLedger

NAMES = {"base": "空手册基线", "ace": "原始 ACE", "selective": "选择性反思",
         "retrieval": "ACE＋检索", "combined": "选择性反思＋检索"}
STATES = {"completed": "完成", "running": "运行中", "incomplete": "已停止", "not_started": "未开始"}


def pad(text, width):
    size = sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)
    return text + " " * max(0, width - size)


def snapshot():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        read_progress()
    data = json.loads(buffer.getvalue())
    config = load_config()
    records = CostLedger(ROOT / "results/repro/live_usage.jsonl", "window", config).records()
    heldout = [row for row in records if row["run_id"].startswith("holdout-")]
    data.update(request_states=dict(Counter(row["state"] for row in heldout)),
                errors=dict(Counter(row.get("status_code") for row in heldout if row.get("status_code"))),
                budget=config["project_limit_cny"])
    for job in data["jobs"]:
        folder = ROOT / "results/repro" / f"holdout-{job['variant']}-seed{job['seed']}"
        job["items"] = max(job["items"], len(list(folder.glob("items/*.json"))))
    data["predictions_completed"] = sum(job["items"] for job in data["jobs"])
    automatic = ROOT / "results/repro/stage3-auto-resume-status.json"
    data["automatic"] = json.loads(automatic.read_text()) if automatic.exists() else None
    return data


def render(data, interval):
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    done = data["predictions_completed"]
    filled = int(30 * done / 1500)
    jobs = data["jobs"]
    running = sum(job["state"] == "running" for job in jobs)
    stopped = sum(job["state"] == "incomplete" for job in jobs)
    if data["completed_conditions"] == 15:
        state = "全部预测完成；评分与报告状态见下方。"
    elif running:
        state = f"{running} 条队列运行中，{stopped} 条队列已停止。"
    elif stopped:
        state = "已停止，需处理失败请求后续跑（本窗口不会重启实验）。"
    else:
        state = "等待启动实验。"
    automatic = data.get("automatic")
    if automatic:
        alive = (datetime.now(ZoneInfo("UTC"))
                 - datetime.fromisoformat(automatic["heartbeat_at_utc"])).total_seconds() < 90
        labels = {"checking": "自动检测服务中（只读）", "probing": "正在用下一道未完成题目试跑恢复",
                  "running": "已自动恢复，正在断点续跑", "finalizing": "预测已齐，正在评分和生成报告"}
        if alive and automatic["state"] == "waiting":
            next_check = datetime.fromisoformat(automatic["next_check_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai"))
            state = f"自动恢复待命；下次检测 {next_check:%H:%M:%S}"
        elif alive and automatic["state"] in labels:
            state = labels[automatic["state"]]
        elif automatic["state"] in ("blocked", "stopped"):
            state = "自动恢复已停止，需检查恢复状态文件。"
    lines = ["ACE 第三阶段 · 实时进度", f"更新时间：{now}（北京时间）｜每 {interval:g} 秒刷新", "",
             f"[{'█' * filled}{'░' * (30-filled)}] {done}/1500  ({done/15:.1f}%)",
             f"完整条件：{data['completed_conditions']}/15", f"状态：{state}", ""]
    for variant, name in NAMES.items():
        parts = [pad(f"{job['seed']} {job['items']:3}/100 {STATES.get(job['state'], job['state'])}", 17)
                 for job in jobs if job["variant"] == variant]
        lines.append(pad("组合方案" if variant == "combined" else name, 18) + " " + " | ".join(parts))
    unknown = data["request_states"].get("failed_billing_unknown", 0)
    pending = data["request_states"].get("pending", 0)
    lines += ["", f"全项目费用核算：{data['all_stages_accounted_cny']:.3f} / {data['budget']:.0f} 元",
              "（含失败和在途费用预留，非已核对的实际扣费）",
              f"第三阶段在途请求：{pending}｜计费待核对失败请求：{unknown}"]
    if data["errors"]:
        lines.append("服务错误记录：" + "，".join(f"HTTP {code} × {count}" for code, count in data["errors"].items()))
    finalizer = ROOT / "results/repro/stage3-finalizer-status.json"
    if finalizer.exists():
        status = json.loads(finalizer.read_text())["status"]
        labels = {"watching_blinded_batch": "等待盲测完成", "waiting_for_complete_batch": "批次未完成，尚未评分",
                  "complete": "报告与图表已生成", "offline_finalization_failed": "离线收尾失败，需检查日志"}
        lines.append("报告：" + labels.get(status, status))
    return "\n".join(lines + ["", "只读监控，不新增 API 调用，不显示测试成绩。", "按 Ctrl+C 退出；关闭本窗口不会停止实验。"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print once, without refreshing")
    parser.add_argument("--interval", type=float, default=5, help="Refresh interval in seconds (minimum 1)")
    args = parser.parse_args()
    interval = max(1, args.interval)
    try:
        while True:
            try:
                if (ROOT / "results/repro/v3-batch-status.json").exists():
                    from repro.v3_progress import display
                    output = display(interval)
                elif (ROOT / "results/repro/v2-batch-status.json").exists():
                    from repro.v2_progress import display
                    output = display(interval)
                else:
                    output = render(snapshot(), interval)
            except (OSError, ValueError, KeyError) as exc:
                output = f"进度文件暂时无法读取（{type(exc).__name__}），稍后重新读取。"
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            if args.once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n监控窗口已退出；未向实验进程发送停止指令。")


if __name__ == "__main__":
    main()
