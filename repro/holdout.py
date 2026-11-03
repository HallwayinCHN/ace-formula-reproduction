"""Frozen stage-three inference only: no target is passed to any generation or retrieval function."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from repro import ROOT
from repro.methods import select_playbook
from repro.pilot import environment_snapshot, load_config
from repro.prepare import ARTIFACTS, FILES, digest, load_rows, verify_upstream
from repro.replay import atomic_json
from repro.resume_balance import BalanceAwareReplay
from repro.transport import CostLedger, ExperimentStopped, installed_transport

VARIANTS = ("base", "ace", "selective", "retrieval", "combined")
LOCK = ARTIFACTS / "stage3_lock.json"


def test_inputs(manifest):
    from eval.finance.data_processor import DataProcessor
    raw = (ROOT / "eval/finance/data" / FILES["test"]).read_bytes()
    if digest(raw) != manifest["split_hashes"]["test"]:
        raise ExperimentStopped("Held-out data hash differs from the frozen manifest")
    rows = load_rows("test")
    inputs = []
    processor = DataProcessor("formula")
    for sample_id in manifest["selected_indices"]["test"]:
        # The original question parser is unchanged; the actual label is discarded before preprocessing.
        processed = processor.process_task_data([{"context": rows[sample_id]["context"], "target": ""}])[0]
        inputs.append({"sample_id": sample_id, "question": processed["question"], "context": processed["context"]})
    return inputs


def book_for(variant, seed, books):
    if variant == "base":
        from ace import ACE
        return ACE._initialize_empty_playbook(None), None
    source = "ace" if variant in ("ace", "retrieval") else "selective"
    book = next(row for row in books["books"] if row["variant"] == source and row["seed"] == seed)
    raw = (ROOT / book["path"]).read_bytes()
    if digest(raw) != book["sha256"]:
        raise ExperimentStopped("Frozen playbook hash changed")
    return raw.decode(), book["sha256"]


def initialize_job(variant, seed):
    verify_upstream()
    freeze = json.loads(LOCK.read_text())
    for name, expected in freeze["file_hashes"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ExperimentStopped(f"Evaluation contract changed: {name}")
    load_dotenv(ROOT / ".env", override=False)
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    books = json.loads((ARTIFACTS / "frozen_books.json").read_text())
    book, book_hash = book_for(variant, seed, books)
    config = {**load_config(), "max_calls": 100, "run_limit_cny": 20.0,
              "variant": variant, "seed": seed, "book_sha256": book_hash,
              "contract_sha256": digest(LOCK.read_bytes()), "retrieval_max_entries": 12,
              "retrieval_max_tokens": 1500, "test_ids": manifest["selected_indices"]["test"]}
    output = ROOT / "results/repro" / f"holdout-{variant}-seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ExperimentStopped("Held-out job configuration differs; refusing to mix runs")
    atomic_json(path, config)
    old = CostLedger(ROOT / "results/repro/live_usage.jsonl", output.name, config).records()
    if any(row["run_id"] == output.name and row["state"] == "rejected_402" for row in old):
        from repro.balance import get_balance
        if get_balance()["is_available"] is not True:
            raise ExperimentStopped("Fresh balance check failed; rejected slot will not be reissued")
    if not (output / "environment.json").exists():
        atomic_json(output / "environment.json", environment_snapshot())
    return config, test_inputs(manifest), book, output


@contextmanager
def evaluation_status(output):
    state = {"blinded": True, "test_inputs_loaded": True, "labels_used_for_generation": False}
    atomic_json(output / "status.json", {**state, "status": "running"})
    try:
        yield
    except BaseException as exc:
        atomic_json(output / "status.json", {**state, "status": "incomplete", "error_type": type(exc).__name__})
        raise
    else:
        atomic_json(output / "status.json", {**state, "status": "completed"})


def infer_item(generator, transport, sample, book, config, output, log_dir):
    started = time.monotonic()
    retrieved = config["variant"] in ("retrieval", "combined")
    selection = select_playbook(sample["question"], book) if retrieved else None
    selected = selection["text"] if selection else book
    selection_seconds = time.monotonic() - started
    previous_replays = transport.replayed
    response, _, info = generator.generate(
        question=sample["question"], playbook=selected, context=sample["context"], reflection="(empty)",
        use_json_mode=config["json_mode"], call_id=f"holdout_{sample['sample_id']}", log_dir=str(log_dir),
    )
    from utils import count_tokens, extract_answer
    item = {"sample_id": sample["sample_id"], "final_answer": extract_answer(response), "error": None,
            "request_id": info["request_id"], "prompt_tokens": info["prompt_num_tokens"],
            "output_tokens": info["response_num_tokens"], "cached_tokens": info.get("cached_tokens", 0),
            "finish_reason": info.get("response_finish_reason"), "api_seconds": info["call_time"],
            "retrieval_seconds": selection_seconds if retrieved else 0.0,
            "end_to_end_seconds": time.monotonic() - started,
            "full_book_tokens": count_tokens(book), "selected_book_tokens": count_tokens(selected),
            "selected_ids": selection["ids"] if selection else None, "latency_reconstructed": False}
    saved = output / "items" / f"{sample['sample_id']:04}.json"
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous.get("request_id") != item["request_id"]:
            raise ExperimentStopped("Saved prediction disagrees with response replay")
        return previous
    if transport.replayed > previous_replays:
        item.update(end_to_end_seconds=info["call_time"] + selection_seconds, latency_reconstructed=True)
    return item


def run_job(variant, seed, client=None, mock=False):
    config, inputs, book, output = initialize_job(variant, seed)
    with (output / "job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExperimentStopped("Held-out job already running") from None
        if (output / "summary.json").exists():
            print(f"[COMPLETE] {output.name}; no model request")
            return
        attempt = output / "attempts" / f"attempt-{len(list((output / 'attempts').glob('attempt-*'))) + 1:03}"
        attempt.mkdir(parents=True)
        transport = BalanceAwareReplay(config, output, client=client, mock=mock)
        from ace.core.generator import Generator
        results = []
        with evaluation_status(output), installed_transport(transport):
            generator = Generator(transport.client, "deepseek_compatible", config["model"], config["max_tokens"])
            for sample in inputs:
                try:
                    item = infer_item(generator, transport, sample, book, config, output, attempt / "calls")
                except Exception as exc:
                    item = {"sample_id": sample["sample_id"], "final_answer": "No final answer found",
                            "error": type(exc).__name__}
                atomic_json(output / "items" / f"{sample['sample_id']:04}.json", item)
                results.append(item)
                atomic_json(output / "progress.json", {"items_completed": len(results), "total": len(inputs)})
            atomic_json(output / "predictions.json", results)
            records = [row for row in transport.ledger.records() if row["run_id"] == transport.run_id]
            atomic_json(output / "summary.json", {
                "status": "completed", "mock": mock, "variant": variant, "seed": seed, "blinded": True,
                "items": len(results), "logical_calls": transport.position, "replayed_calls": transport.replayed,
                "physical_requests": len(records), "accounted_cny": sum(row["accounted_cny"] for row in records),
                "predictions_sha256": digest((output / "predictions.json").read_bytes()),
                "book_sha256": config["book_sha256"], "contract_sha256": config["contract_sha256"],
            })
        print(json.dumps({"job": output.name, "items": len(results), "blinded": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    args = parser.parse_args()
    run_job(args.variant, args.seed)


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
