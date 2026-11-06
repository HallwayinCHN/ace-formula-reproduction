"""One of six blinded fixed-book V4 inference jobs, with durable response replay."""

import argparse
import fcntl
import json
import time

from dotenv import load_dotenv

from ace.core.generator import Generator
from repro import ROOT
from repro.prepare import digest
from repro.recovery import safe_json, utcnow
from repro.transport import ExperimentStopped, installed_transport
from repro.v4_protocol import CONTRACT, book_for, read_rows, verify
from repro.v4_transport import Overloaded, PhaseReplay, STATE
from utils import extract_answer


def run(variant, seed):
    contract = verify()
    if variant not in contract["variants"] or seed not in contract["seeds"]:
        raise ExperimentStopped("Unregistered V4 condition")
    load_dotenv(ROOT / ".env", override=False)
    output = STATE / f"v4-{variant}-seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "job.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "summary.json").exists():
            return
        book = book_for(contract, variant, seed)
        book_sha = contract["books"][f"{variant}-seed{seed}"]["sha256"]
        inputs = read_rows(contract["evaluation_ids"])
        config = {**contract["inference_config"], "variant": variant, "seed": seed,
                  "contract_sha256": digest(CONTRACT.read_bytes()), "book_sha256": book_sha}
        if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != config:
            raise ExperimentStopped("Job config changed")
        safe_json(output / "config.json", config)
        attempt = output / "attempts" / str(time.time_ns())
        attempt.mkdir(parents=True)
        transport = PhaseReplay(config, output)
        started = time.monotonic()
        safe_json(output / "status.json", {"state": "running", "targets_supplied_to_agent": False,
                                           "started_at_utc": utcnow()})
        try:
            with installed_transport(transport):
                generator = Generator(transport.client, "deepseek_compatible", config["model"], config["max_tokens"])
                items = []
                for sample in inputs:
                    before = time.monotonic()
                    previous_replays = transport.replayed
                    response, _, info = generator.generate(
                        question=sample["question"], context=sample["context"], playbook=book,
                        reflection="(empty)", use_json_mode=config["json_mode"],
                        call_id=f"v4_{sample['sample_id']}", log_dir=str(attempt))
                    item = {"sample_id": sample["sample_id"], "final_answer": extract_answer(response),
                            "request_id": info["request_id"], "prompt_tokens": info["prompt_num_tokens"],
                            "output_tokens": info["response_num_tokens"], "cached_tokens": info.get("cached_tokens", 0),
                            "accounted_cny": info["accounted_cny"], "api_seconds": info["call_time"],
                            "end_to_end_seconds": (time.monotonic()-before
                                                   if transport.replayed == previous_replays else None),
                            "length_response": info.get("response_finish_reason") == "length"}
                    path = output / "items" / f"{sample['sample_id']:04}.json"
                    if path.exists():
                        prior = json.loads(path.read_text())
                        if prior["request_id"] != item["request_id"]:
                            raise ExperimentStopped("Durable prediction differs from replay")
                        item = prior
                    safe_json(path, item)
                    items.append(item)
                    safe_json(output / "progress.json", {"items_completed": len(items), "total": len(inputs)})
            safe_json(output / "predictions.json", items)
            safe_json(output / "summary.json", {"state": "completed", "variant": variant, "seed": seed,
                                                "items": len(items), "book_sha256": book_sha,
                                                "contract_sha256": digest(CONTRACT.read_bytes()),
                                                "predictions_sha256": digest(
                                                    (output / "predictions.json").read_bytes()),
                                                "targets_supplied_to_agent": False})
            safe_json(output / "status.json", {"state": "completed"})
        except BaseException as exc:
            safe_json(output / "status.json", {"state": "stopped", "error_type": type(exc).__name__})
            raise
        finally:
            safe_json(attempt / "timing.json", {"wall_seconds": time.monotonic()-started,
                                                "new_calls": transport.calls, "replayed_calls": transport.replayed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "nodiag"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    try:
        run(args.variant, args.seed)
    except Overloaded as exc:
        print(f"OVERLOAD_COOLDOWN: {exc}")
        raise SystemExit(75)
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
