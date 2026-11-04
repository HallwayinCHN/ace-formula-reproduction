"""One prospective test of the confirmed V2 candidate on the unused 100 test IDs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import statistics
import subprocess
import sys
import time

import numpy as np
from dotenv import load_dotenv

from repro import ROOT
from repro.holdout import book_for
from repro.holdout_stats import paired_interval, standardized_cost
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, FILES, digest, load_rows
from repro.recovery import safe_json, utcnow
from repro.transport import CostLedger, ExperimentStopped, installed_transport
from repro.v2_prepare import CONTRACT as SCREEN, confirmation_candidate, verify
from repro.v2_screen import DevelopmentReplay, infer

CONTRACT = ARTIFACTS / "v2_final_contract.json"
STATE = ROOT / "results/repro"


def freeze():
    screen = verify()
    candidate = confirmation_candidate()
    confirmation = json.loads((ARTIFACTS / "v2_confirm_analysis.json").read_text())
    if not confirmation["groups"][candidate]["screen_gate_met"]:
        raise ExperimentStopped("Candidate did not pass development confirmation; test stays sealed")
    config = load_config()
    original = confirmation["groups"]["original"]
    new = confirmation["groups"][candidate]
    forecast = (original["standardized_cny"] + new["standardized_cny"]) * 100/40 * 1.3
    spent = confirmation["v2_accounted_cny"]
    if spent + forecast > 20 or confirmation["project_accounted_cny"] + forecast > 180:
        raise ExperimentStopped("Conservative forecast exceeds the shared V2/project budget")
    data = {"created_at_utc": utcnow(), "candidate": candidate, "test_ids": screen["reserved_unopened_test_ids"],
            "test_sha256": digest((ROOT / "eval/finance/data" / FILES["test"]).read_bytes()),
            "prediction_count": 600, "question_count": 100, "seeds": [42, 43, 44],
            "source_sha256": digest((ROOT / "repro/v2_holdout.py").read_bytes()),
            "screen_contract_sha256": digest(SCREEN.read_bytes()),
            "confirmation_analysis_sha256": digest((ARTIFACTS / "v2_confirm_analysis.json").read_bytes()),
            "primary": "confirmed_candidate_minus_original",
            "decision": "Report paired question-cluster CI, cost, tokens and failures; no formal noninferiority claim",
            "schedule": {"42": ["original", candidate], "43": [candidate, "original"],
                         "44": ["original", candidate]},
            "budget": "Shared with screening and confirmation: all V2 <=20 CNY, project <=180 CNY",
            "additional_standardized_forecast_plus_30pct_cny": forecast,
            "blinding": "All six conditions finish before any test scoring; mask targets before preprocessing"}
    if CONTRACT.exists():
        check()
    else:
        safe_json(CONTRACT, data)
    print(json.dumps({"sealed": True, "test_predictions": 600, "candidate": candidate,
                      "additional_forecast_cny": forecast}))


def check():
    screen = verify()
    data = json.loads(CONTRACT.read_text())
    if data["test_ids"] != screen["reserved_unopened_test_ids"] or len(set(data["test_ids"])) != 100:
        raise ExperimentStopped("Prospective test cohort differs from the reserved IDs")
    if digest((ROOT / "repro/v2_holdout.py").read_bytes()) != data["source_sha256"]:
        raise ExperimentStopped("Prospective evaluation code changed")
    if digest(SCREEN.read_bytes()) != data["screen_contract_sha256"]:
        raise ExperimentStopped("Screen contract changed")
    if digest((ARTIFACTS / "v2_confirm_analysis.json").read_bytes()) != data["confirmation_analysis_sha256"]:
        raise ExperimentStopped("Confirmation report changed after final selection")
    if confirmation_candidate() != data["candidate"]:
        raise ExperimentStopped("Final method changed")
    return data


def masked_inputs(contract):
    if digest((ROOT / "eval/finance/data" / FILES["test"]).read_bytes()) != contract["test_sha256"]:
        raise ExperimentStopped("Test data changed")
    from eval.finance.data_processor import DataProcessor
    rows = load_rows("test")
    ids = contract["test_ids"]
    processed = DataProcessor("formula").process_task_data(
        [{"context": rows[index]["context"], "target": ""} for index in ids])
    return [{"sample_id": index, "question": row["question"], "context": row["context"]}
            for index, row in zip(ids, processed)]


def run_job(variant, seed):
    contract = check()
    if variant not in ("original", contract["candidate"]):
        raise ExperimentStopped("Unregistered final condition")
    load_dotenv(ROOT / ".env", override=False)
    config = {**load_config(), "max_calls": 200, "run_limit_cny": 10}
    inputs = masked_inputs(contract)
    book, book_hash = book_for("ace", seed, json.loads((ARTIFACTS / "frozen_books.json").read_text()))
    output = STATE / f"v2-final-{variant}-seed{seed}"
    output.mkdir(exist_ok=True)
    with (output / "job.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "summary.json").exists():
            return
        safe_json(output / "status.json", {"state": "running", "labels_used_for_generation": False})
        transport = DevelopmentReplay(config, output)
        from ace.core.generator import Generator
        attempt = output / "attempts" / str(time.time_ns())
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
                            raise ExperimentStopped("Saved held-out result differs from replay")
                        item = previous
                    safe_json(path, item)
                    results.append(item)
                    safe_json(output / "progress.json", {"items_completed": len(results), "total": 100})
            safe_json(output / "predictions.json", results)
            safe_json(output / "summary.json", {"state": "completed", "items": len(results), "variant": variant,
                                                "seed": seed, "book_sha256": book_hash,
                                                "contract_sha256": digest(CONTRACT.read_bytes()),
                                                "predictions_sha256": digest((output / "predictions.json").read_bytes())})
            safe_json(output / "status.json", {"state": "completed", "labels_used_for_generation": False})
        except BaseException as exc:
            safe_json(output / "status.json", {"state": "stopped", "error_type": type(exc).__name__})
            raise


def lane(seed, variants):
    for variant in variants:
        with (STATE / f"v2-final-{variant}-seed{seed}.console.log").open("a") as handle:
            code = subprocess.run([sys.executable, "-u", "-m", "repro.v2_holdout", "--variant", variant,
                                   "--seed", str(seed)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code:
            return code
    return 0


def batch():
    contract = check()
    from repro.balance import get_balance
    if get_balance()["is_available"] is not True:
        raise ExperimentStopped("Account balance unavailable")
    with (STATE / "v2-batch.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        safe_json(STATE / "v2-batch-status.json", {"state": "running", "stage": "final", "started_at_utc": utcnow()})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lane, int(seed), variants) for seed, variants in contract["schedule"].items()]
            codes = [future.result() for future in futures]
        safe_json(STATE / "v2-batch-status.json", {"state": "completed" if not any(codes) else "stopped",
                                                  "stage": "final", "lane_exit_codes": codes})
        if any(codes):
            raise ExperimentStopped("Final batch stopped; no partial test scoring")


def analyze():
    contract = check()
    variants = ("original", contract["candidate"])
    jobs = []
    for variant in variants:
        for seed in (42, 43, 44):
            output = STATE / f"v2-final-{variant}-seed{seed}"
            summary = json.loads((output / "summary.json").read_text())
            if summary["state"] != "completed" or summary["items"] != 100:
                raise ValueError("All six test conditions must finish before scoring")
            if summary["contract_sha256"] != digest(CONTRACT.read_bytes()):
                raise ValueError("Conditions used different contracts")
            if summary["predictions_sha256"] != digest((output / "predictions.json").read_bytes()):
                raise ValueError("Prediction file changed")
            jobs.append((variant, seed, json.loads((output / "predictions.json").read_text())))
    from eval.finance.data_processor import DataProcessor
    labels, processor = load_rows("test"), DataProcessor("formula")
    ids = contract["test_ids"]
    matrices = {variant: np.zeros((100, 3)) for variant in variants}
    rows = []
    for variant, seed, items in jobs:
        if len(items) != 100 or {item["sample_id"] for item in items} != set(ids):
            raise ValueError("Wrong test cohort")
        for item in items:
            target = labels[item["sample_id"]]["target"]
            correct = processor.answer_is_correct(item["final_answer"], target)
            matrices[variant][ids.index(item["sample_id"]), seed-42] = correct
            rows.append({**item, "variant": variant, "seed": seed, "target": target, "is_correct": bool(correct)})
    summarize(contract, rows, matrices)


def summarize(contract, rows, matrices):
    config = load_config()
    ledger = CostLedger(STATE / "live_usage.jsonl", "v2-final-analysis", config).records()
    groups = {}
    for variant in matrices:
        requests = [row for row in ledger if row["run_id"].startswith(f"v2-final-{variant}-seed")]
        items = [row for row in rows if row["variant"] == variant]
        prompt = sum(row.get("prompt_tokens", 0) for row in requests)
        output = sum(row.get("completion_tokens", 0) for row in requests)
        accuracy = matrices[variant].mean(axis=0).tolist()
        groups[variant] = {"accuracy": statistics.mean(accuracy), "accuracy_sd": statistics.stdev(accuracy),
                           "seed_accuracy": accuracy, "correct": int(matrices[variant].sum()), "total": 300,
                           "accounted_cny_per_100": sum(row["accounted_cny"] for row in requests)/3,
                           "standardized_cny_per_100": standardized_cost(prompt, output, config)/3,
                           "prompt_tokens": prompt, "output_tokens": output,
                           "cache_hit_ratio": sum(row.get("cached_tokens", 0) for row in requests)/prompt,
                           "fallback_items": sum(row["fallback_used"] for row in items),
                           "truncated_responses": sum(row["length_responses"] for row in items),
                           "missing_answers": sum(row["final_answer"] == "No final answer found" for row in items),
                           "median_api_seconds": statistics.median(row["api_seconds"] for row in items)}
    candidate = contract["candidate"]
    comparison = paired_interval(matrices[candidate], matrices["original"])
    report = {"status": "prospective_test_complete", "test_questions": 100, "prediction_rows": len(rows),
              "candidate": candidate, "groups": groups, "primary_comparison": comparison,
              "original_100_test_ids_excluded": True, "formal_noninferiority_proved": False,
              "v2_accounted_cny": sum(row["accounted_cny"] for row in ledger if row["run_id"].startswith("v2-")),
              "project_accounted_cny": sum(row["accounted_cny"] for row in ledger),
              "unknown_billing_requests": sum(row["state"] in ("pending", "failed_billing_unknown") for row in ledger)}
    safe_json(ARTIFACTS / "v2_final_analysis.json", report)
    with (ARTIFACTS / "v2_final_scored.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    try:
        if args.seal:
            freeze()
        elif args.analyze:
            analyze()
        elif args.variant:
            if args.seed is None:
                parser.error("--seed required with --variant")
            run_job(args.variant, args.seed)
        else:
            batch()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
