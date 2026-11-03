"""Single-call transport diagnosis for the first pilot's truncated reflection."""

from datetime import datetime, timezone
import json
import uuid

from repro import ROOT
from repro.pilot import load_config, preflight, run_status
from repro.prepare import pilot_data, write_json
from repro.transport import ExperimentStopped, Transport, installed_transport


def main():
    config = {**load_config(), "json_mode": True, "max_calls": 1, "allow_truncated_nonempty": False}
    if not preflight(config)["key_configured"]:
        raise ExperimentStopped("Local key is not configured")
    from ace import ACE
    from eval.finance.data_processor import DataProcessor
    from playbook_utils import extract_playbook_bullets
    from utils import extract_answer

    source_run = ROOT / "results/repro/pilot-20261103T074246Z-64ca69d4"
    matches = []
    for path in source_run.glob("upstream/*/detailed_llm_logs/*.json"):
        row = json.loads(path.read_text())
        if row.get("call_id") == "train_e_1_s_1_gen_initial":
            matches.append(row)
    if len(matches) != 1:
        raise ExperimentStopped("Expected exactly one recorded source generation")
    source = matches[0]
    sample = DataProcessor("formula").process_task_data(pilot_data()["train"][:1])[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results/repro" / f"diagnostic-json-{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    write_json(output / "config.json", {**config, "purpose": "single reflection diagnosis",
                                       "source_run": source_run.name, "held_out_test_loaded": False})
    transport = Transport(config, output)
    with run_status(output), installed_transport(transport):
        system = ACE("deepseek_compatible", config["model"], config["model"], config["model"])
        reflection, tags, info = system.reflector.reflect(
            question=sample["question"], reasoning_trace=source["response"],
            predicted_answer=extract_answer(source["response"]), ground_truth=sample["target"],
            environment_feedback="Predicted answer does not match ground truth",
            bullets_used=extract_playbook_bullets(system.playbook, []),
            use_ground_truth=True, use_json_mode=True, call_id="diagnose_first_reflection",
            log_dir=str(output / "calls"),
        )
        parsed = json.loads(reflection)
        write_json(output / "summary.json", {"status": "completed", "json_valid": isinstance(parsed, dict),
                                            "output_tokens": info["response_num_tokens"],
                                            "response_finish_reason": info["response_finish_reason"],
                                            "held_out_test_loaded": False})
    print(json.dumps({"output": str(output), "json_valid": True, "output_tokens": info["response_num_tokens"]}))


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as exc:
        print(f"STOPPED: {exc}")
        raise SystemExit(2)
