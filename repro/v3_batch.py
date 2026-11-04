"""Execute the registered V3 phase with budget gates, bounded 503 recovery and no partial unblinding."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

from repro import ROOT
from repro.auto_resume import control_health
from repro.pilot import load_config
from repro.prepare import ARTIFACTS
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger, ExperimentStopped
from repro.v3_analysis import select_router, summarize_evaluation, summarize_training, training_checks
from repro.v3_protocol import verify
from repro.v3_transport import CIRCUIT, STATE, STOP

STATUS = STATE / "v3-batch-status.json"


def lane(tasks):
    for stage, variant, seed in tasks:
        directory = STATE / f"v3-{stage}-{variant}-seed{seed}"
        if (directory / "summary.json").exists():
            continue
        if CIRCUIT.exists() or STOP.exists():
            return 75
        with (STATE / f"v3-{stage}-{variant}-seed{seed}.console.log").open("a") as handle:
            code = subprocess.run([sys.executable, "-u", "-m", "repro.v3_jobs", "--stage", stage,
                                   "--variant", variant, "--seed", str(seed)], cwd=ROOT,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        if code:
            return code
    return 0


class Controller:
    def __init__(self):
        self.contract = verify()
        self.status = {"state": "starting", "pid": os.getpid(), "started_at_utc": utcnow(),
                       "evaluation_opened": False, "phase_limit_cny": 30}

    def update(self, **fields):
        self.status.update(fields, heartbeat_at_utc=utcnow())
        safe_json(STATUS, self.status)

    def cooldown(self, label, attempt):
        deadline = datetime.now(timezone.utc)+timedelta(seconds=self.contract["503_cooldown_seconds"])
        self.update(state="waiting", label=label, recovery_round=attempt, next_check_at_utc=deadline.isoformat())
        while not STOP.exists() and datetime.now(timezone.utc) < deadline:
            self.update()
            time.sleep(min(15, max(0, (deadline-datetime.now(timezone.utc)).total_seconds())))

    def wave(self, lanes, stage, label):
        for attempt in range(self.contract["maximum_recovery_rounds"]):
            if STOP.exists():
                raise ExperimentStopped("User requested V3 stop")
            verify()
            self.update(state="checking", stage=stage, label=label)
            health = control_health()
            if not health["healthy"]:
                self.cooldown(label, attempt+1)
                continue
            CIRCUIT.unlink(missing_ok=True)
            self.update(state="running", stage=stage, label=label, next_check_at_utc=None)
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = {pool.submit(lane, tasks) for tasks in lanes}
                codes = []
                while pending:
                    done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    codes.extend(future.result() for future in done)
                    self.update()
            if not any(codes):
                return
            if any(code not in (0, 75) for code in codes):
                raise ExperimentStopped(f"Non-503 failure in {label}; no automatic restart")
            self.cooldown(label, attempt+1)
        raise ExperimentStopped("Bounded V3 recovery rounds exhausted")

    def budget_after_pilot(self, router):
        pair = training_checks((42,))[0]
        rows = CostLedger(STATE / "live_usage.jsonl", "v3-budget", load_config()).records()
        scoped = [row for row in rows if row["run_id"].startswith("v3-")]
        spent = sum(row["accounted_cny"] for row in scoped)
        pair_cost = pair["original"]["accounted_cny"]+pair["nodiag"]["accounted_cny"]
        dev = {variant: sum(row["accounted_cny"] for row in scoped
                            if row["run_id"].startswith(f"v3-dev-{variant}-seed"))
               for variant in ("original", "brief")}
        evaluation = 4*(2*dev["original"]+dev["brief"]) if router["passed"] else 0
        remaining_forecast = (2*pair_cost+evaluation)*1.3
        result = {"spent_cny": spent, "remaining_forecast_plus_30pct_cny": remaining_forecast,
                  "total_forecast_cny": spent+remaining_forecast, "evaluation_gate_passed": router["passed"]}
        safe_json(ARTIFACTS / "v3_pilot_budget.json", result)
        if spent+remaining_forecast > 30 or sum(row["accounted_cny"] for row in rows)+remaining_forecast > 180:
            raise ExperimentStopped("Pilot-based remaining budget forecast exceeds the registered stop line")

    def run(self):
        dev_lanes = [[("dev", variant, seed) for variant in variants] for seed, variants in
                     ((42, ("original", "brief")), (43, ("brief", "original")), (44, ("original", "brief")))]
        self.wave(dev_lanes, "dev", "新开发集双输出对照")
        router = select_router()
        self.update(router_gate_passed=router["passed"], router_allowed_buckets=(router["rule"] or {}).get(
            "allowed_buckets", []))
        self.wave([[('train', variant, 42) for variant in self.contract["learning_schedule"]["42"]]],
                  "train", "seed42 真实训练配对检查")
        self.budget_after_pilot(router)
        training_lanes = [[("train", variant, seed) for variant in self.contract["learning_schedule"][str(seed)]]
                          for seed in (43, 44)]
        self.wave(training_lanes, "train", "seed43、44 真实训练配对")
        summarize_training()
        if router["passed"]:
            variants = self.contract["eval_variants"]
            eval_lanes = [[("eval", variant, seed) for variant in variants[offset:]+variants[:offset]]
                          for seed, offset in ((42, 0), (43, 1), (44, 2))]
            self.update(evaluation_opened=True)
            self.wave(eval_lanes, "eval", "160 题封存评估")
            summarize_evaluation()
        self.update(state="reporting")
        from repro.v3_report import finish
        finish()
        self.update(state="complete", finished_at_utc=utcnow(), next_check_at_utc=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        STOP.write_text(utcnow()+"\n")
        print("V3 stop requested; in-flight responses may finish")
        return
    with (STATE / "v3-batch.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("V3 controller already running") from None
        controller = Controller()
        try:
            controller.run()
        except BaseException as exc:
            message = str(exc) if isinstance(exc, ExperimentStopped) else type(exc).__name__
            controller.update(state="stopped" if STOP.exists() else "blocked", reason=message)
            print(f"V3_STOPPED: {message}")
            raise SystemExit(2) from None


if __name__ == "__main__":
    main()
