"""Run the six registered V4 jobs, then unblind once and generate the offline report."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import fcntl
import os
import subprocess
import sys
import time

from repro import ROOT
from repro.auto_resume import control_health
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger, ExperimentStopped
from repro.v4_protocol import verify
from repro.v4_transport import CIRCUIT, STATE, STOP

STATUS = STATE / "v4-batch-status.json"


def lane(seed, variants):
    for variant in variants:
        if (STATE / f"v4-{variant}-seed{seed}/summary.json").exists():
            continue
        if CIRCUIT.exists() or STOP.exists():
            return 75
        with (STATE / f"v4-{variant}-seed{seed}.console.log").open("a") as output:
            code = subprocess.run([sys.executable, "-u", "-m", "repro.v4_job", "--seed", str(seed),
                                   "--variant", variant], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
        if code:
            return code
    return 0


class Controller:
    def __init__(self):
        self.contract = verify()
        self.status = {"state": "starting", "pid": os.getpid(), "started_at_utc": utcnow(),
                       "phase_limit_cny": 10, "labels_scored": False}

    def update(self, **fields):
        self.status.update(fields, heartbeat_at_utc=utcnow())
        safe_json(STATUS, self.status)

    def guard(self):
        if STOP.exists():
            raise ExperimentStopped("User requested V4 stop")
        verify()
        rows = [row for row in CostLedger(STATE / "live_usage.jsonl", "v4-guard",
                                         self.contract["inference_config"]).records()
                if row["run_id"].startswith("v4-")]
        if any(row["state"] == "pending" for row in rows):
            raise ExperimentStopped("Prior V4 request is pending; no overlapping restart")
        if any(row["state"] == "failed_billing_unknown" and row.get("status_code") != 503 for row in rows):
            raise ExperimentStopped("Non-503 unknown billing requires reconciliation")

    def cooldown(self, attempt):
        deadline = datetime.now(timezone.utc)+timedelta(seconds=self.contract["503_cooldown_seconds"])
        self.update(state="waiting", recovery_round=attempt, next_check_at_utc=deadline.isoformat())
        while datetime.now(timezone.utc) < deadline and not STOP.exists():
            self.update()
            time.sleep(min(15, max(0, (deadline-datetime.now(timezone.utc)).total_seconds())))

    def run(self):
        for attempt in range(self.contract["maximum_recovery_rounds"]):
            self.guard()
            self.update(state="checking")
            if not control_health()["healthy"]:
                self.cooldown(attempt+1)
                continue
            CIRCUIT.unlink(missing_ok=True)
            self.update(state="running", next_check_at_utc=None)
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = {pool.submit(lane, int(seed), variants)
                           for seed, variants in self.contract["schedule"].items()}
                codes = []
                while pending:
                    done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    codes.extend(future.result() for future in done)
                    self.update()
            if not any(codes):
                break
            if any(code not in (0, 75) for code in codes):
                raise ExperimentStopped("Non-503 V4 job failure; no automatic restart")
            self.cooldown(attempt+1)
        else:
            raise ExperimentStopped("Bounded V4 recovery exhausted")
        self.update(state="reporting")
        from repro.v4_analysis import analyze
        analyze()
        self.update(labels_scored=True)
        from repro.v4_report import finish
        finish()
        self.update(state="complete", finished_at_utc=utcnow(), next_check_at_utc=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        STOP.write_text(utcnow()+"\n")
        print("V4 stop requested; in-flight responses may finish")
        return
    with (STATE / "v4-batch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("V4 controller already running") from None
        controller = Controller()
        try:
            controller.run()
        except BaseException as exc:
            reason = str(exc) if isinstance(exc, ExperimentStopped) else type(exc).__name__
            controller.update(state="stopped" if STOP.exists() else "blocked", reason=reason)
            print(f"V4_STOPPED: {reason}")
            raise SystemExit(2) from None


if __name__ == "__main__":
    main()
