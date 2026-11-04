"""Small, separately sealed validation-only screen; no reserved test loader or automatic test call."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import subprocess
import sys
import time

from dotenv import load_dotenv

from repro import ROOT
from repro.holdout import book_for
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest, load_rows
from repro.recovery import safe_json, utcnow
from repro.replay import ReplayTransport
from repro.transport import CostLedger, ExperimentStopped, installed_transport
from repro.v2_methods import VARIANTS, brief_template, core_playbook, numeric_answer
from repro.v2_prepare import confirmation_candidate, verify

STATE = ROOT / "results/repro"


class DevelopmentLedger(CostLedger):
    def _check(self, amount):
        super()._check(amount)
        spent = sum(row["accounted_cny"] for row in self._records() if row["run_id"].startswith("v2-"))
        if spent + amount > 20:
            raise ExperimentStopped("V2 development budget reached; no request")


class DevelopmentReplay(ReplayTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = DevelopmentLedger(self.ledger.path, self.run_id, self.config)

    def _failure(self, record, exc, started):
        super()._failure(record, exc, started)
        raise ExperimentStopped(f"Development request stopped: {type(exc).__name__}; reserve retained") from None


@contextmanager
def prompt_template(brief):
    import ace.core.generator as module
    original = module.GENERATOR_PROMPT
    module.GENERATOR_PROMPT = brief_template() if brief else original
    try:
        yield
    finally:
        module.GENERATOR_PROMPT = original


def validation_inputs(contract, stage):
    raw = (ROOT / "eval/finance/data" / FILES["validation"]).read_bytes()
    if digest(raw) != contract["validation_sha256"]:
        raise ExperimentStopped("Validation hash changed")
    from eval.finance.data_processor import DataProcessor
    rows = load_rows("validation")
    ids = contract["screen_validation_ids" if stage == "screen" else "confirmation_validation_ids"]
    processed = DataProcessor("formula").process_task_data(
        [{"context": rows[index]["context"], "target": ""} for index in ids])
    return [{"sample_id": index, "question": row["question"], "context": row["context"]}
            for index, row in zip(ids, processed)]


def infer(generator, transport, sample, book, variant, attempt):
    from utils import count_tokens, extract_answer
    started = time.monotonic()
    selection = core_playbook(sample["question"], book) if variant.startswith("core_") else {
        "text": book, "mode": "full", "ids": None, "tokens": count_tokens(book)}
    brief = variant in ("brief_guard", "core_brief_guard")
    infos = []
    with prompt_template(brief):
        response, _, info = generator.generate(sample["question"], selection["text"], sample["context"],
                                               call_id=f"v2_{sample['sample_id']}_initial", log_dir=str(attempt))
    answer = extract_answer(response)
    first = answer
    infos.append(info)
    fallback = variant != "original" and not numeric_answer(answer)
    if fallback:
        with prompt_template(False):
            response, _, info = generator.generate(sample["question"], book, sample["context"],
                                                   call_id=f"v2_{sample['sample_id']}_fallback", log_dir=str(attempt))
        answer = extract_answer(response)
        infos.append(info)
    return {"sample_id": sample["sample_id"], "final_answer": answer, "first_answer": first,
            "fallback_used": fallback, "context_mode": selection["mode"], "selected_ids": selection["ids"],
            "selected_book_tokens": selection["tokens"], "request_ids": [info["request_id"] for info in infos],
            "prompt_tokens": sum(info["prompt_num_tokens"] for info in infos),
            "output_tokens": sum(info["response_num_tokens"] for info in infos),
            "cached_tokens": sum(info.get("cached_tokens", 0) for info in infos),
            "length_responses": sum(info.get("response_finish_reason") == "length" for info in infos),
            "api_seconds": sum(info["call_time"] for info in infos),
            "end_to_end_seconds": time.monotonic() - started}


def run_job(variant, seed, stage):
    contract = verify()
    load_dotenv(ROOT / ".env", override=False)
    config = {**load_config(), "max_calls": 80, "run_limit_cny": 10}
    inputs = validation_inputs(contract, stage)
    book, _ = book_for("ace", seed, json.loads((ARTIFACTS / "frozen_books.json").read_text()))
    output = STATE / f"v2-{stage}-{variant}-seed{seed}"
    output.mkdir(exist_ok=True)
    with (output / "job.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "summary.json").exists():
            return
        safe_json(output / "status.json", {"state": "running", "reserved_test_loaded": False})
        attempt = output / "attempts" / str(time.time_ns())
        transport = DevelopmentReplay(config, output)
        from ace.core.generator import Generator
        try:
            with installed_transport(transport):
                generator = Generator(transport.client, "deepseek_compatible", config["model"], config["max_tokens"])
                results = []
                for sample in inputs:
                    item = infer(generator, transport, sample, book, variant, attempt)
                    path = output / "items" / f"{sample['sample_id']:04}.json"
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if previous["request_ids"] != item["request_ids"]:
                            raise ExperimentStopped("Saved development output differs from replay")
                        item = previous
                    safe_json(path, item)
                    results.append(item)
                    safe_json(output / "progress.json", {"items_completed": len(results), "total": len(inputs)})
            safe_json(output / "predictions.json", results)
            safe_json(output / "summary.json", {"state": "completed", "variant": variant, "seed": seed,
                                                "items": len(results), "stage": stage, "test_loaded": False})
            safe_json(output / "status.json", {"state": "completed", "reserved_test_loaded": False})
        except BaseException as exc:
            safe_json(output / "status.json", {"state": "stopped", "error_type": type(exc).__name__})
            raise


def lane(seed, variants, stage):
    for variant in variants:
        with (STATE / f"v2-{stage}-{variant}-seed{seed}.console.log").open("a") as handle:
            code = subprocess.run([sys.executable, "-u", "-m", "repro.v2_screen", "--variant", variant,
                                   "--seed", str(seed), "--stage", stage], cwd=ROOT,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        if code:
            return code
    return 0


def batch(stage):
    verify()
    if stage == "screen":
        variants = list(VARIANTS)
    else:
        variants = ["original", confirmation_candidate()]
    from repro.balance import get_balance
    if get_balance()["is_available"] is not True:
        raise ExperimentStopped("Account balance unavailable")
    with (STATE / "v2-batch.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = STATE / "v2-batch-status.json"
        safe_json(status, {"state": "running", "stage": stage, "started_at_utc": utcnow()})
        schedules = [variants[index % len(variants):] + variants[:index % len(variants)] for index in (0, 2, 4)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, seed, schedule, stage) for seed, schedule in zip((42, 43, 44), schedules)]
            codes = [future.result() for future in futures]
        safe_json(status, {"state": "completed" if not any(codes) else "stopped", "stage": stage,
                           "lane_exit_codes": codes, "finished_at_utc": utcnow(), "test_loaded": False})
        if any(codes):
            raise ExperimentStopped("Development batch stopped; inspect logs before any restart")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    args = parser.parse_args()
    try:
        if args.variant:
            if args.seed is None:
                parser.error("--seed required with --variant")
            run_job(args.variant, args.seed, args.stage)
        else:
            batch(args.stage)
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
