"""Fresh routing comparisons and live diagnostic-free training, using one global phase ledger."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import time

from dotenv import load_dotenv

from ace import ACE
from repro import ROOT
from repro.diagnostic_free_replay import learning_step
from repro.evaluation import evaluate_fixed
from repro.formal import formal_data
from repro.holdout import book_for
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest
from repro.recovery import safe_json, utcnow
from repro.transport import ExperimentStopped, installed_transport
from repro.v2_screen import prompt_template
from repro.v3_protocol import CONTRACT, RULE, selected_rows, verify
from repro.v3_router import choose
from repro.v3_transport import Overloaded, PhaseReplay

STATE = ROOT / "results/repro"


def rule():
    data = json.loads(RULE.read_text())
    if data["contract_sha256"] != digest(CONTRACT.read_bytes()) or not data["passed"]:
        raise ExperimentStopped("Routing gate not passed or contract changed")
    for path, expected in data["dev_prediction_hashes"].items():
        if digest((ROOT / path).read_bytes()) != expected:
            raise ExperimentStopped("Development predictions changed after routing selection")
    return data["rule"]


@contextmanager
def job(stage, variant, seed, max_calls):
    verify()
    load_dotenv(ROOT / ".env", override=False)
    output = STATE / f"v3-{stage}-{variant}-seed{seed}"
    output.mkdir(exist_ok=True)
    with (output / "job.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "summary.json").exists():
            yield None
            return
        config = {**load_config(), "max_calls": max_calls, "run_limit_cny": 10,
                  "stage": stage, "variant": variant, "seed": seed, "contract_sha256": digest(CONTRACT.read_bytes())}
        if stage == "eval":
            config["router_lock_sha256"] = digest(RULE.read_bytes())
        if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != config:
            raise ExperimentStopped("Job configuration changed")
        safe_json(output / "config.json", config)
        if not (output / "started.json").exists():
            safe_json(output / "started.json", {"started_at_utc": utcnow()})
        attempt = output / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        transport = PhaseReplay(config, output)
        started = time.monotonic()
        safe_json(output / "status.json", {"state": "running", "stage": stage,
                                           "training_uses_ground_truth": stage == "train",
                                           "new_evaluation_loaded": stage == "eval",
                                           "heldout_targets_supplied_to_agent": False})
        try:
            with installed_transport(transport):
                yield config, output, attempt, transport
        except BaseException as exc:
            safe_json(output / "status.json", {"state": "stopped", "error_type": type(exc).__name__})
            raise
        else:
            safe_json(output / "status.json", {"state": "completed", "stage": stage})
        finally:
            safe_json(attempt / "timing.json", {"wall_seconds": time.monotonic()-started,
                                                "new_calls": transport.calls, "replayed_calls": transport.replayed})


def infer(generator, sample, book, brief, attempt):
    from utils import extract_answer
    started = time.monotonic()
    with prompt_template(brief):
        response, _, info = generator.generate(sample["question"], book, sample["context"],
                                               call_id=f"v3_{sample['sample_id']}", log_dir=str(attempt))
    return {"sample_id": sample["sample_id"], "final_answer": extract_answer(response), "brief": brief,
            "request_id": info["request_id"], "prompt_tokens": info["prompt_num_tokens"],
            "output_tokens": info["response_num_tokens"], "cached_tokens": info.get("cached_tokens", 0),
            "accounted_cny": info["accounted_cny"], "length_response": info.get("response_finish_reason") == "length",
            "api_seconds": info["call_time"], "generation_seconds": time.monotonic()-started}


def inference_job(stage, variant, seed):
    contract = verify()
    if stage not in ("dev", "eval") or variant not in contract[f"{stage}_variants"]:
        raise ExperimentStopped("Unregistered routing condition")
    policy = rule() if stage == "eval" else None  # Check gate before loading any evaluation question.
    ids = contract["dev_ids" if stage == "dev" else "evaluation_ids"]
    inputs = selected_rows(ids)
    book, _ = book_for("ace", seed, json.loads((ARTIFACTS / "frozen_books.json").read_text()))
    with job(stage, variant, seed, len(ids)) as prepared:
        if prepared is None:
            return
        config, output, attempt, transport = prepared
        from ace.core.generator import Generator
        generator = Generator(transport.client, "deepseek_compatible", config["model"], config["max_tokens"])
        results = []
        for sample in inputs:
            started = time.monotonic()
            brief = variant == "brief"
            profile = None
            if variant == "adaptive":
                brief, profile = choose(sample["question"], policy)
            routing_seconds = time.monotonic()-started
            item = infer(generator, sample, book, brief, attempt)
            item.update(routing_seconds=routing_seconds, profile=profile,
                        end_to_end_seconds=item["generation_seconds"]+routing_seconds)
            path = output / "items" / f"{sample['sample_id']:04}.json"
            if path.exists():
                prior = json.loads(path.read_text())
                if prior["request_id"] != item["request_id"]:
                    raise ExperimentStopped("Durable result disagrees with replay")
                item = prior
            safe_json(path, item)
            results.append(item)
            safe_json(output / "progress.json", {"items_completed": len(results), "total": len(inputs)})
        safe_json(output / "predictions.json", results)
        safe_json(output / "summary.json", {"state": "completed", "stage": stage, "variant": variant,
                                            "seed": seed, "items": len(results),
                                            "contract_sha256": digest(CONTRACT.read_bytes()),
                                            "router_lock_sha256": config.get("router_lock_sha256"),
                                            "predictions_sha256": digest((output / "predictions.json").read_bytes())})


def training_job(variant, seed):
    if variant not in ("original", "nodiag"):
        raise ExperimentStopped("Unknown training condition")
    data, processor, _ = formal_data(seed)
    with job("train", variant, seed, 650) as prepared:
        if prepared is None:
            return
        config, output, attempt, transport = prepared
        system = ACE("deepseek_compatible", config["model"], config["model"], config["model"],
                     max_tokens=config["max_tokens"], use_bulletpoint_analyzer=False)
        params = system._extract_config_params({**config, "task_name": "formula", "no_ground_truth": False})
        traces, checkpoints = [], []
        best_accuracy, best_book = -1, None
        for step, sample in enumerate(data["train"], 1):
            kwargs = {"task_dict": sample, "data_processor": processor, "step_id": f"train_e_1_s_{step}",
                      "epoch": 1, "step": step, "usage_log_path": str(attempt / "usage.jsonl"),
                      "log_dir": str(attempt), "config_params": params, "total_samples": len(data["train"])}
            result = learning_step(system, **kwargs) if variant == "nodiag" else system._train_single_sample(**kwargs)
            trace = {"step": step, **result[2], "book_sha256": digest(system.playbook.encode())}
            traces.append(trace)
            safe_json(output / "steps" / f"{step:03}.json", trace)
            safe_json(output / "progress.json", {"items_completed": step, "total": 60})
            if step in (30, 60):
                summary, _ = evaluate_fixed(processor, system.generator, system.playbook, data["validation"],
                                             log_dir=str(attempt), use_json_mode=config["json_mode"])
                checkpoints.append({"step": step, **summary, "book_sha256": digest(system.playbook.encode())})
                if summary["accuracy"] > best_accuracy:
                    best_accuracy, best_book = summary["accuracy"], system.playbook
        (output / "best_playbook.txt").write_text(best_book)
        safe_json(output / "trace.json", traces)
        records = [row for row in transport.ledger.records() if row["run_id"] == output.name]
        accepted = [row for row in records if row["state"] == "completed"]
        learned = [row for row in accepted if row["call_id"].startswith("train_")]
        from repro.holdout_stats import standardized_cost
        summary = {"state": "completed", "variant": variant, "seed": seed, "steps": 60,
                   "checkpoints": checkpoints, "best_validation_accuracy": best_accuracy,
                   "best_book_sha256": digest(best_book.encode()), "learning_calls": len(learned),
                   "learning_accounted_cny": sum(row["accounted_cny"] for row in records
                                                  if row["call_id"].startswith("train_")),
                   "learning_standardized_cny": standardized_cost(
                       sum(row["prompt_tokens"] for row in learned),
                       sum(row["completion_tokens"] for row in learned), config),
                   "diagnostic_calls": sum(row["call_id"].endswith("_post_curate") for row in learned),
                   "accounted_cny": sum(row["accounted_cny"] for row in records),
                   "prompt_tokens": sum(row["prompt_tokens"] for row in accepted),
                   "output_tokens": sum(row["completion_tokens"] for row in accepted),
                   "cached_tokens": sum(row.get("cached_tokens", 0) for row in accepted),
                   "api_seconds": sum(row["elapsed_seconds"] for row in records if "elapsed_seconds" in row),
                   "post_diagnostic_measured": variant == "original", "new_eval_questions_loaded": False,
                   "contract_sha256": digest(CONTRACT.read_bytes()),
                   "wall_seconds": (datetime.now(timezone.utc)-datetime.fromisoformat(
                       json.loads((output / "started.json").read_text())["started_at_utc"])).total_seconds()}
        safe_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "eval", "train"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    try:
        training_job(args.variant, args.seed) if args.stage == "train" else inference_job(
            args.stage, args.variant, args.seed)
    except Overloaded as exc:
        print(f"OVERLOAD_COOLDOWN: {exc}")
        raise SystemExit(75)
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
