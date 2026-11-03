"""Run stage-one pilot only. This entrypoint never loads held-out test records."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
import json
import os
import uuid

from dotenv import load_dotenv

from repro import ROOT
from repro.prepare import PIN, pilot_data, verify_upstream, write_json
from repro.transport import ExperimentStopped, Transport, installed_transport


def load_config():
    return json.loads((ROOT / "repro/configs/pilot.json").read_text())


def preflight(config):
    verify_upstream()
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get(config["key_env"], "").strip()
    return {"key_configured": bool(key and not key.startswith("your_")),
            "key_env": config["key_env"], "model": config["model"],
            "pilot_train": len(pilot_data()["train"]), "pilot_validation": len(pilot_data()["validation"]),
            "held_out_test_loaded": False}


def environment_snapshot():
    return {"packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
            "upstream_commit": PIN, "upstream_source_hashes": verify_upstream()}


@contextmanager
def run_status(output):
    write_json(output / "status.json", {"status": "running", "held_out_test_loaded": False})
    try:
        yield
    except BaseException as exc:
        write_json(output / "status.json", {"status": "incomplete", "error_type": type(exc).__name__,
                                            "held_out_test_loaded": False})
        raise
    else:
        write_json(output / "status.json", {"status": "completed", "held_out_test_loaded": False})


def run_pilot(config, mock):
    from ace import ACE
    from eval.finance.data_processor import DataProcessor
    from utils import evaluate_test_set
    from repro.mock_client import mock_client, synthetic_data

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "results/repro" / (("mock-" if mock else "pilot-") + run_id)
    output.mkdir(parents=True)
    processor = DataProcessor("formula")
    if mock:
        data = synthetic_data()
        config = {**config, "model": "MOCK-NOT-A-MODEL"}
    else:
        data = {split: processor.process_task_data(values) for split, values in pilot_data().items()}
    write_json(output / "environment.json", environment_snapshot())
    write_json(output / "config.json", {**config, "mock": mock, "held_out_test_loaded": False})
    transport = Transport(config, output, client=mock_client() if mock else None, mock=mock)
    with run_status(output), installed_transport(transport):
        system = ACE("deepseek_compatible", config["model"], config["model"], config["model"],
                     max_tokens=config["max_tokens"], use_bulletpoint_analyzer=False)
        baseline, _ = evaluate_test_set(processor, system.generator, system.playbook, data["validation"],
                                        log_dir=str(output / "baseline_calls"), max_workers=1)
        if baseline["total"] != len(data["validation"]):
            raise ExperimentStopped("Incomplete pilot baseline denominator")
        run_config = {**config, "task_name": "formula", "no_ground_truth": False,
                      "eval_steps": max(1, len(data["train"]) // 2),
                      "save_steps": max(1, len(data["train"]) // 2), "save_dir": str(output / "upstream")}
        result = system.run("offline", train_samples=data["train"], val_samples=data["validation"],
                            test_samples=[], data_processor=processor, config=run_config)
        training_log = next((output / "upstream").glob("*/train_results.json"))
        checkpoints = json.loads(training_log.read_text())["results"]
        if not checkpoints:
            raise ExperimentStopped("No validation checkpoints were produced")
        for item in checkpoints:
            if item["val_result"]["total"] != len(data["validation"]):
                raise ExperimentStopped("Incomplete pilot validation denominator")
    records = [row for row in transport.ledger.records() if row["run_id"] == transport.run_id]
    summary = {
        "status": "mock_plumbing_passed" if mock else "pilot_completed",
        "mock": mock, "quality_result": "NOT VALID FOR MODEL QUALITY" if mock else "pilot validation only",
        "held_out_test_loaded": False, "train_count": len(data["train"]),
        "validation_count": len(data["validation"]), "baseline": baseline,
        "training_results": result["training_results"], "logical_calls": transport.calls,
        "real_request_attempts": 0 if mock else len(records),
        "actual_provider_charge_cny": None, "cost_is_estimate": True,
        "accounted_run_cny": transport.ledger.totals()[1],
        "formal_size_locked": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(output),
                      "calls": transport.calls, "held_out_test_loaded": False}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check", action="store_true", help="Check readiness without making API calls")
    choice.add_argument("--mock", action="store_true", help="Synthetic local HTTP fixtures; no real API calls")
    choice.add_argument("--live", action="store_true", help="Run the 10-train/10-validation paid pilot")
    args = parser.parse_args()
    config = load_config()
    readiness = preflight(config)
    if args.check:
        print(json.dumps(readiness, indent=2))
        return
    if args.live and not readiness["key_configured"]:
        raise ExperimentStopped("DEEPSEEK_API_KEY is missing; no model request was sent")
    run_pilot(config, args.mock)


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
