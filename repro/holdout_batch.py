"""Run blinded held-out inference in three pre-rotated condition lanes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys

from repro import ROOT
from repro.balance import get_balance
from repro.holdout import LOCK
from repro.replay import atomic_json


def run_lane(seed, variants):
    results = []
    for variant in variants:
        log = ROOT / "results/repro" / f"holdout-{variant}-seed{seed}.console.log"
        with log.open("a") as handle:
            process = subprocess.run(
                [sys.executable, "-u", "-m", "repro.holdout", "--variant", variant, "--seed", str(seed)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            )
        results.append({"variant": variant, "seed": seed, "exit_code": process.returncode})
        print(json.dumps(results[-1]), flush=True)
        if process.returncode:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    freeze = json.loads(LOCK.read_text())
    if get_balance()["is_available"] is not True:
        raise SystemExit("Provider reports no available balance; no model call made")
    protocol_path = ROOT / "repro/configs/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["held_out_test_status"] = "inference_blinded"
    atomic_json(protocol_path, protocol)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_lane, int(seed), variants)
                for seed, variants in freeze["condition_schedule"].items()]
        for future in as_completed(jobs):
            results.extend(future.result())
            atomic_json(ROOT / "repro/artifacts/stage3_batch_status.json", {"jobs": results, "blinded": True})
    complete = len(results) == 15 and all(row["exit_code"] == 0 for row in results)
    protocol["held_out_test_status"] = "inference_complete_blinded" if complete else "inference_incomplete"
    atomic_json(protocol_path, protocol)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
