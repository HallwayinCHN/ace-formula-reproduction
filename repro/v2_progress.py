"""Read-only view of validation development; never read predictions, labels or test contents."""

from datetime import datetime
import json
from zoneinfo import ZoneInfo

from repro import ROOT
from repro.pilot import load_config
from repro.transport import CostLedger
from repro.v2_methods import VARIANTS


def display(interval):
    root = ROOT / "results/repro"
    batch = json.loads((root / "v2-batch-status.json").read_text())
    stage = batch["stage"]
    if stage == "screen":
        variants, size = VARIANTS, 20
    else:
        selected = json.loads((ROOT / "repro/artifacts/v2_confirmation_lock.json").read_text())["candidate"]
        variants, size = ("original", selected), 100 if stage == "final" else 40
    done, completed, lines = 0, 0, []
    for variant in variants:
        pieces = []
        for seed in (42, 43, 44):
            directory = root / f"v2-{stage}-{variant}-seed{seed}"
            count = len(list(directory.glob("items/*.json")))
            state = "完成" if (directory / "summary.json").exists() else "未开始"
            if (directory / "status.json").exists() and state != "完成":
                raw = json.loads((directory / "status.json").read_text())["state"]
                state = {"running": "运行中", "stopped": "已停止"}.get(raw, raw)
            done += count
            completed += state == "完成"
            pieces.append(f"{seed}: {count}/{size} {state}")
        lines += [variant, "  " + " ｜ ".join(pieces)]
    total = len(variants)*size*3
    rows = CostLedger(root / "live_usage.jsonl", "v2-window", load_config()).records()
    research = [row for row in rows if row["run_id"].startswith("v2-")]
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    title = {"screen": "方案筛选", "confirm": "开发确认", "final": "独立测试"}[stage]
    boundary = ("独立测试生成期间不读取标签；全部条件完成后统一评分。" if stage == "final"
                else "当前是验证集开发实验；剩余 100 道测试题继续封存。")
    return "\n".join([f"ACE 第二轮研究 · {title}", f"{now} 北京时间｜每 {interval:g} 秒刷新", "",
                      f"进度：{done}/{total}（{done/total*100:.1f}%）｜完成条件 {completed}/{len(variants)*3}",
                      f"批次状态：{batch['state']}", "", *lines, "",
                      f"第二轮核算：{sum(row['accounted_cny'] for row in research):.3f}/20 元（含预留）",
                      f"全项目核算：{sum(row['accounted_cny'] for row in rows):.3f}/180 元",
                      boundary,
                      "本窗口只读，不调用 API；Ctrl+C 或关闭窗口仅退出监控。"])
