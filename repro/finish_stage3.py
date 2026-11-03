"""Finish the authorized offline analysis after the entire blinded batch completes."""

import argparse
import fcntl
import json
import subprocess
import time

from repro import ROOT
from repro.prepare import ARTIFACTS
from repro.replay import atomic_json


def finish(watch=False):
    output = ROOT / "results/repro"
    with (output / "stage3-finalizer.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = ROOT / "repro/configs/protocol.json"
        status_path = output / "stage3-finalizer-status.json"
        while True:
            protocol = json.loads(protocol_path.read_text())
            state = protocol["held_out_test_status"]
            if state in ("inference_complete_blinded", "scored"):
                break
            if state == "inference_incomplete" or not watch:
                atomic_json(status_path, {"status": "waiting_for_complete_batch", "batch_status": state})
                return
            atomic_json(status_path, {"status": "watching_blinded_batch", "model_calls": 0})
            time.sleep(15)
        try:
            from repro.holdout_analysis import analyze
            from repro.verify_stage3 import verify
            from repro.report_stage3 import report
            analyze()
            verify()
            subprocess.run([str(ROOT / ".plot-venv/bin/python"), "-m", "repro.plot_stage3"], cwd=ROOT, check=True)
            report()
            protocol["held_out_test_status"] = "scored"
            atomic_json(protocol_path, protocol)
            readme = ROOT / "repro/README.md"
            text = readme.read_text().replace(
                "第三阶段已封存并启动：五组 × 三个种子 × 100 个固定测试 ID，全部预测完成前不评分。",
                "第三阶段 1,500 条真实预测全部完成并统一评分，结果与局限见 `artifacts/STAGE3.md`。",
            )
            readme.write_text(text)
            atomic_json(status_path, {"status": "complete", "model_calls": 0,
                                      "report": str(ARTIFACTS / "STAGE3.md")})
        except BaseException as exc:
            atomic_json(status_path, {"status": "offline_finalization_failed", "error_type": type(exc).__name__})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Wait for the existing batch; never start model calls")
    finish(parser.parse_args().watch)
