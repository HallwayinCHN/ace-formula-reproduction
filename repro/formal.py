"""Stage two: six formal learning jobs, validation only, with no held-out test loading."""

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import json
from pathlib import Path
import random
import shutil

from dotenv import load_dotenv

from repro import ROOT
from repro.evaluation import evaluate_fixed
from repro.methods import SelectiveACE, select_playbook
from repro.pilot import environment_snapshot, load_config, run_status
from repro.prepare import ARTIFACTS, FILES, digest, load_rows, verify_upstream, write_json
from repro.replay import ReplayTransport, atomic_json
from repro.transport import ExperimentStopped, installed_transport

HARNESS = ("formal.py", "methods.py", "evaluation.py", "replay.py", "transport.py")


def formal_data(seed):
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    if not manifest["formal_size_locked"]:
        raise ExperimentStopped("Formal sample size is not frozen")
    from eval.finance.data_processor import DataProcessor
    processor = DataProcessor("formula")
    data = {}
    for split in ("train", "validation"):
        raw = (ROOT / "eval/finance/data" / FILES[split]).read_bytes()
        if digest(raw) != manifest["split_hashes"][split]:
            raise ExperimentStopped(f"Frozen data hash changed: {split}")
        indices = list(manifest["selected_indices"][split])
        if split == "train":
            random.Random(seed).shuffle(indices)
        rows = load_rows(split)
        data[split] = processor.process_task_data([rows[i] for i in indices])
        data[f"{split}_indices"] = indices
    return data, processor, manifest


@contextmanager
def fixed_evaluation():
    module = importlib.import_module("ace.ace")
    original = module.evaluate_test_set
    module.evaluate_test_set = evaluate_fixed
    try:
        yield
    finally:
        module.evaluate_test_set = original


def initialize_job(variant, seed):
    verify_upstream()
    load_dotenv(ROOT / ".env", override=False)
    config = {**load_config(), "max_calls": 650, "run_limit_cny": 20.0,
              "variant": variant, "order_seed": seed, "held_out_test_loaded": False}
    data, processor, manifest = formal_data(seed)
    config.update(train_indices=data["train_indices"], validation_indices=data["validation_indices"],
                  manifest_sha256=digest((ARTIFACTS / "sample_manifest.json").read_bytes()),
                  harness_hashes={name: digest((ROOT / "repro" / name).read_bytes()) for name in HARNESS})
    output = ROOT / "results/repro" / f"formal-{variant}-seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ExperimentStopped("Frozen job config differs; refusing to mix runs")
    atomic_json(path, config)
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", environment_snapshot())
    return config, data, processor, output


def summarize(output, system, transport, data, result, attempt):
    training_log = next(attempt.glob("*/train_results.json"))
    checkpoints = json.loads(training_log.read_text())["results"]
    if len(checkpoints) != 2 or any(row["val_result"]["total"] != len(data["validation"]) for row in checkpoints):
        raise ExperimentStopped("Incomplete validation checkpoints")
    book_path = next(attempt.glob("*/best_playbook.txt"))
    shutil.copyfile(book_path, output / "best_playbook.txt")
    records = [row for row in transport.ledger.records() if row["run_id"] == transport.run_id]
    learning = [row for row in records if row["call_id"].startswith("train_")]
    validation = [row for row in records if row["call_id"].startswith("test_")]
    c = transport.config
    full_miss = sum((row.get("prompt_tokens", 0) * c["input_usd_per_million"]
                     + row.get("completion_tokens", 0) * c["output_usd_per_million"])
                    / 1_000_000 * c["usd_to_cny_budget_factor"] for row in learning)
    trace = json.loads(next(attempt.glob("*/pre_train_post_train_results.json")).read_text())
    # Offline inspection of retrieval compression on validation questions, no additional model calls.
    selections = [select_playbook(row["question"], system.best_playbook) for row in data["validation"]]
    summary = {
        "status": "completed", "mock": transport.mock,
        "variant": transport.config["variant"], "seed": transport.config["order_seed"],
        "train_count": len(data["train"]), "validation_count": len(data["validation"]),
        "best_validation_accuracy": result["training_results"]["best_validation_accuracy"],
        "checkpoints": checkpoints, "logical_calls": transport.position, "replayed_calls": transport.replayed,
        "actual_request_attempts": len(records), "accounted_cny": sum(row["accounted_cny"] for row in records),
        "learning_calls": len(learning), "validation_calls": len(validation),
        "learning_accounted_cny": sum(row["accounted_cny"] for row in learning),
        "validation_accounted_cny": sum(row["accounted_cny"] for row in validation),
        "learning_full_cache_miss_cny": full_miss,
        "roles": {role: sum(row["role"] == role for row in records) for role in ("generator", "reflector", "curator")},
        "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in records),
        "output_tokens": sum(row.get("completion_tokens", 0) for row in records),
        "truncated": sum(row.get("response_finish_reason") == "length" for row in records),
        "api_seconds": sum(row.get("elapsed_seconds", 0) for row in records),
        "skipped_correct": sum(row["skipped"] for row in system.gate_events),
        "best_playbook_sha256": digest((output / "best_playbook.txt").read_bytes()),
        "retrieval_validation_tokens": [row["selected_tokens"] for row in selections],
        "full_book_tokens": selections[0]["full_tokens"], "held_out_test_loaded": False,
        "initial_train_correct": sum(row["pre_train_result"]["is_correct"] for row in trace),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def run_job(variant, seed):
    config, data, processor, output = initialize_job(variant, seed)
    with (output / "job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExperimentStopped("This formal job is already running") from None
        if (output / "summary.json").exists():
            print(f"[COMPLETE] {output.name}; no API calls")
            return
        attempt = output / "attempts" / f"attempt-{len(list((output / 'attempts').glob('attempt-*'))) + 1:03}"
        attempt.mkdir(parents=True)
        transport = ReplayTransport(config, output)
        with run_status(output), installed_transport(transport), fixed_evaluation():
            system = SelectiveACE(
                "deepseek_compatible", config["model"], config["model"], config["model"],
                max_tokens=config["max_tokens"], use_bulletpoint_analyzer=False,
                selective_enabled=variant == "selective", keep_every=5,
            )
            interval = len(data["train"]) // 2
            run_config = {**config, "task_name": "formula", "no_ground_truth": False,
                          "eval_steps": interval, "save_steps": interval, "save_dir": str(attempt)}
            result = system.run("offline", train_samples=data["train"], val_samples=data["validation"],
                                test_samples=[], data_processor=processor, config=run_config)
            summary = summarize(output, system, transport, data, result, attempt)
        print(json.dumps({"job": output.name, "accuracy": summary["best_validation_accuracy"],
                          "accounted_cny": summary["accounted_cny"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["ace", "selective"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    run_job(args.variant, args.seed)


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
