"""Periodically check service access and resume only explicit-503 checkpoints, still blinded."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

import httpx

from repro import ROOT
from repro.balance import get_balance
from repro.holdout import LOCK
from repro.pilot import load_config
from repro.recovery import (CIRCUIT, CONTRACT, STATE, STOP, failure_exposure, safe_json,
                            seal_recovery, utcnow, validate_recovery)
from repro.transport import CostLedger, ExperimentStopped

STATUS = STATE / "stage3-auto-resume-status.json"


def counts():
    folders = [STATE / f"holdout-{variant}-seed{seed}"
               for variant in ("base", "ace", "selective", "retrieval", "combined") for seed in (42, 43, 44)]
    return {"predictions_completed": sum(len(list(folder.glob("items/*.json"))) for folder in folders),
            "completed_conditions": sum((folder / "summary.json").exists() for folder in folders)}


def control_health():
    """These read-only endpoints do not prove inference health: the real next item is the probe."""
    config = load_config()
    try:
        balance = get_balance()
        if balance["is_available"] is not True:
            raise ExperimentStopped("No available provider balance")
        response = httpx.get(config["base_url"].rstrip("/") + "/models", timeout=30,
                             headers={"Authorization": "Bearer " + os.environ[config["key_env"]]})
        if response.status_code in (400, 401, 402, 422):
            raise ExperimentStopped(f"Control endpoint requires attention: HTTP {response.status_code}")
        if response.status_code != 200:
            return {"healthy": False, "http_status": response.status_code}
        available = {model["id"] for model in response.json()["data"]}
        return {"healthy": config["model"] in available, "model_available": config["model"] in available,
                "balance_available": True, "http_status": 200}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {"healthy": False, "error_type": type(exc).__name__}
    except RuntimeError as exc:
        if str(exc) in ("Local key is not configured", "Balance check status=401", "Balance check status=402"):
            raise ExperimentStopped("Local credentials or account need attention") from None
        return {"healthy": False, "error_type": type(exc).__name__}


def child_command(seed, variant, probe=False):
    command = [sys.executable, "-u", "-m", "repro.resume_holdout", "--seed", str(seed), "--variant", variant]
    return command + (["--probe"] if probe else [])


def run_lane(seed, variants):
    for variant in variants:
        folder = STATE / f"holdout-{variant}-seed{seed}"
        if (folder / "summary.json").exists():
            continue
        if CIRCUIT.exists() or STOP.exists():
            return 75
        with (STATE / f"holdout-{variant}-seed{seed}.recovery.console.log").open("a") as handle:
            code = subprocess.run(child_command(seed, variant), cwd=ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            return code
    return 0


class Controller:
    def __init__(self, contract):
        self.contract = contract
        self.policy = contract["policy"]
        self.schedule = json.loads(LOCK.read_text())["condition_schedule"]
        self.status = {"pid": os.getpid(), "started_at_utc": utcnow(), "cycles": 0,
                       "interval_seconds": self.policy["interval_seconds"], "state": "starting"}

    def update(self, state=None, **fields):
        if state:
            self.status["state"] = state
        self.status.update(fields, heartbeat_at_utc=utcnow(), **counts())
        safe_json(STATUS, self.status)

    def guard(self):
        validate_recovery()
        config = load_config()
        rows = CostLedger(STATE / "live_usage.jsonl", "auto-resume", config).records()
        heldout = [row for row in rows if row["run_id"].startswith("holdout-")]
        if any(row["state"] == "pending" for row in heldout):
            raise ExperimentStopped("Prior request still in flight; no overlapping restart")
        if any(row["state"] == "failed_billing_unknown" and row.get("status_code") != 503 for row in heldout):
            raise ExperimentStopped("Non-503 unknown response/billing requires reconciliation")
        exposure = failure_exposure(rows, self.contract["created_at_utc"])
        if exposure >= self.policy["max_extra_failure_reserve_cny"]:
            raise ExperimentStopped("Additional failure reservation budget exhausted")
        if sum(row["accounted_cny"] for row in rows) >= config["project_limit_cny"]:
            raise ExperimentStopped("Project budget exhausted")
        self.update(extra_failure_reserve_cny=exposure)

    def pending_job(self):
        for seed, variants in self.schedule.items():
            for variant in variants:
                if not (STATE / f"holdout-{variant}-seed{seed}/summary.json").exists():
                    return int(seed), variant
        return None

    def probe(self, seed, variant):
        self.update("probing", active_probe={"seed": seed, "variant": variant}, next_check_at_utc=None)
        before = counts()["predictions_completed"]
        with (STATE / f"holdout-{variant}-seed{seed}.recovery.console.log").open("a") as handle:
            process = subprocess.Popen(child_command(seed, variant, probe=True), cwd=ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
            while process.poll() is None:
                self.update()
                time.sleep(5)
        after = counts()["predictions_completed"]
        self.update(last_probe_exit_code=process.returncode, last_probe_new_predictions=after - before)
        if process.returncode in (0, 3) and after > before:
            return True
        if process.returncode == 75:
            return False
        raise ExperimentStopped("Checkpoint probe stopped for a non-503 reason; inspect its local log")

    def batch(self):
        self.update("running", active_probe=None)
        with ThreadPoolExecutor(max_workers=self.policy["max_active_conditions"]) as pool:
            futures = {pool.submit(run_lane, int(seed), variants) for seed, variants in self.schedule.items()}
            codes = []
            while futures:
                done, futures = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                codes.extend(future.result() for future in done)
                self.update()
        if any(code not in (0, 75) for code in codes):
            raise ExperimentStopped("A lane stopped for a non-503 reason; no automatic restart")

    def set_protocol(self, state):
        path = ROOT / "repro/configs/protocol.json"
        data = json.loads(path.read_text())
        data.update(held_out_test_status=state, automatic_recovery_contract=str(CONTRACT.relative_to(ROOT)))
        safe_json(path, data)

    def cooldown(self):
        deadline = datetime.now(timezone.utc) + timedelta(seconds=self.policy["interval_seconds"])
        self.update("waiting", next_check_at_utc=deadline.isoformat(), active_probe=None)
        while datetime.now(timezone.utc) < deadline and not STOP.exists():
            self.update()
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            time.sleep(min(15, max(0, remaining)))

    def run(self):
        deadline = datetime.fromisoformat(self.contract["created_at_utc"]) + timedelta(
            hours=self.policy["maximum_watch_hours"])
        while datetime.now(timezone.utc) < deadline and not STOP.exists():
            self.guard()
            if not self.pending_job():
                self.set_protocol("inference_complete_blinded")
                self.update("finalizing", next_check_at_utc=None)
                from repro.finish_stage3 import finish
                finish()
                self.update("complete")
                return
            self.status["cycles"] += 1
            self.update("checking", next_check_at_utc=None)
            health = control_health()
            self.update(last_health_check=health, last_health_check_at_utc=utcnow())
            if health["healthy"]:
                CIRCUIT.unlink(missing_ok=True)
                self.set_protocol("inference_blinded")
                if self.probe(*self.pending_job()):
                    self.batch()
                    if not self.pending_job():
                        continue
                self.set_protocol("inference_incomplete")
            self.cooldown()
        self.update("stopped", reason="User stop flag or maximum watch duration reached", next_check_at_utc=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seal", action="store_true", help="Seal recovery sources and preserved files; no API calls")
    group.add_argument("--stop", action="store_true", help="Stop future retries; allow current response to finish")
    group.add_argument("--resume", action="store_true", help="Explicitly clear a prior user stop flag and start")
    args = parser.parse_args()
    if args.stop:
        STOP.write_text(utcnow() + "\n")
        print("Stop requested; any in-flight response may finish before shutdown")
        return
    with (STATE / "stage3-auto-resume.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Automatic recovery controller already running") from None
        if args.seal:
            contract = seal_recovery()
            print(json.dumps({"sealed": True, "preserved_predictions":
                              contract["successful_predictions_before_recovery"]}))
            return
        controller = Controller(validate_recovery())
        if args.resume:
            STOP.unlink(missing_ok=True)
        try:
            controller.run()
        except BaseException as exc:
            message = str(exc) if isinstance(exc, ExperimentStopped) else type(exc).__name__
            if counts()["predictions_completed"] < 1500:
                controller.set_protocol("inference_incomplete")
            controller.update("stopped" if STOP.exists() else "blocked", reason=message, next_check_at_utc=None)
            print(f"AUTOMATIC_RECOVERY_STOPPED: {message}")
            raise SystemExit(2) from None


if __name__ == "__main__":
    main()
