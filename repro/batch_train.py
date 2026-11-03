"""Run the three order-seed pairs concurrently; each individual learning sequence stays serial."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys

from repro import ROOT
from repro.replay import atomic_json


def train_seed(seed):
    outcomes = []
    for variant in ("ace", "selective"):
        log = ROOT / "results/repro" / f"formal-{variant}-seed{seed}.console.log"
        with log.open("a") as handle:
            process = subprocess.run(
                [sys.executable, "-u", "-m", "repro.formal", "--variant", variant, "--seed", str(seed)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            )
        outcomes.append({"variant": variant, "seed": seed, "exit_code": process.returncode,
                         "log": str(log.relative_to(ROOT))})
        print(json.dumps(outcomes[-1]), flush=True)
        if process.returncode:
            break
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    protocol_path = ROOT / "repro/configs/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected = {"train": 60, "validation": 30, "test": 100}
    if not protocol["formal_size_locked"] or protocol["selected_counts"] != expected:
        raise SystemExit("Unexpected formal protocol")
    protocol["formal_run_status"] = "training"
    atomic_json(protocol_path, protocol)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(train_seed, seed) for seed in protocol["training_order_seeds"]]
        for future in as_completed(jobs):
            results.extend(future.result())
            atomic_json(ROOT / "repro/artifacts/stage2_batch_status.json", {"jobs": results,
                                                                         "held_out_test_loaded": False})
    passed = len(results) == 6 and all(row["exit_code"] == 0 for row in results)
    protocol["formal_run_status"] = "training_complete" if passed else "training_incomplete"
    atomic_json(protocol_path, protocol)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
