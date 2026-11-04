"""Verify removal of measurement-only post-curation calls against real archived trajectories. No API access."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile

from ace import ACE
from repro import ROOT
from repro.evaluation import evaluate_fixed
from repro.formal import formal_data
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest, verify_upstream
from repro.recovery import safe_json
from repro.transport import ExperimentStopped, installed_transport
from utils import count_tokens, extract_answer


class DiagnosticSkipped(BaseException):
    pass


def learning_step(system, **kwargs):
    """Only for this explicit training driver; missing post-update accuracy is recorded as unavailable."""
    original = system.generator.generate
    initial = {}

    def generate(*args, **values):
        if values["call_id"] == kwargs["step_id"] + "_post_curate":
            raise DiagnosticSkipped()
        result = original(*args, **values)
        if values["call_id"] == kwargs["step_id"] + "_gen_initial":
            answer = extract_answer(result[0])
            initial.update(final_answer=answer,
                           is_correct=kwargs["data_processor"].answer_is_correct(answer, kwargs["task_dict"]["target"]),
                           playbook_num_tokens=count_tokens(system.playbook), playbook_length=len(system.playbook))
        return result

    system.generator.generate = generate
    try:
        system._train_single_sample(**kwargs)
        raise AssertionError("Expected to stop only at the diagnostic call")
    except DiagnosticSkipped:
        return initial["final_answer"], None, {
            "pre_train_result": initial, "post_train_result": {"status": "not_measured", "final_answer": None,
                                                               "is_correct": None}}
    finally:
        system.generator.generate = original


class Archive:
    """No network client is constructed. Every consumed request must match an archived fingerprint."""
    def __init__(self, seed, skip_diagnostics):
        folder = ROOT / "results/repro" / f"formal-ace-seed{seed}/replay"
        self.records = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]
        if skip_diagnostics:
            self.records = [row for row in self.records if not row["info"]["call_id"].endswith("_post_curate")]
        self.position = 0
        self.client = object()
        self.config = load_config()

    def call(self, client, api_provider, model, prompt, role, call_id, max_tokens=4096,
             log_dir=None, use_json_mode=False, **unused):
        if self.position >= len(self.records):
            raise ExperimentStopped("Replay unexpectedly requires a new request")
        row = self.records[self.position]
        material = [model, prompt, role, call_id, max_tokens, use_json_mode,
                    self.config["temperature"], self.config["thinking"]]
        fingerprint = hashlib.sha256(json.dumps(material, ensure_ascii=False).encode()).hexdigest()
        if fingerprint != row["fingerprint"]:
            raise ExperimentStopped(f"Learning trajectory changed at {call_id}/{role}; no API request made")
        self.position += 1
        return row["text"], dict(row["info"])


def replay(seed, skip_diagnostics):
    data, processor, _ = formal_data(seed)
    archive = Archive(seed, skip_diagnostics)
    config = {**load_config(), "task_name": "formula", "no_ground_truth": False}
    states, checkpoints = [], []
    best_accuracy, best_book = -1, None
    with tempfile.TemporaryDirectory(prefix="ace-diagnostic-replay-") as directory:
        folder = Path(directory)
        with installed_transport(archive), redirect_stdout(io.StringIO()):
            system = ACE("deepseek_compatible", config["model"], config["model"], config["model"],
                         max_tokens=config["max_tokens"], use_bulletpoint_analyzer=False)
            params = system._extract_config_params(config)
            for step, sample in enumerate(data["train"], 1):
                kwargs = {"task_dict": sample, "data_processor": processor, "step_id": f"train_e_1_s_{step}",
                          "epoch": 1, "step": step, "usage_log_path": str(folder / "usage.jsonl"),
                          "log_dir": str(folder), "config_params": params, "total_samples": len(data["train"])}
                result = learning_step(system, **kwargs) if skip_diagnostics else system._train_single_sample(**kwargs)
                states.append({"step": step, "book_sha256": digest(system.playbook.encode()),
                               "next_global_id": system.next_global_id, "pre_train_answer": result[0]})
                if skip_diagnostics:
                    assert result[1] is None and result[2]["post_train_result"]["is_correct"] is None
                if step in (30, 60):
                    validation, _ = evaluate_fixed(processor, system.generator, system.playbook, data["validation"],
                                                   log_dir=str(folder), use_json_mode=config["json_mode"])
                    checkpoints.append({"step": step, "correct": validation["correct"], "total": validation["total"],
                                        "book_sha256": states[-1]["book_sha256"]})
                    if validation["accuracy"] > best_accuracy:
                        best_accuracy, best_book = validation["accuracy"], system.playbook
            final_hash = digest(system.playbook.encode())
    if archive.position != len(archive.records):
        raise ExperimentStopped("Archived request consumption differs from the full training schedule")
    return {"seed": seed, "skip_diagnostics": skip_diagnostics, "matched_requests": archive.position,
            "states": states, "checkpoints": checkpoints, "best_book_sha256": digest(best_book.encode()),
            "final_book_sha256": final_hash, "best_validation_accuracy": best_accuracy}


def main():
    verify_upstream()
    runs = []
    for seed in (42, 43, 44):
        original = replay(seed, False)
        optimized = replay(seed, True)
        for key in ("states", "checkpoints", "best_book_sha256", "final_book_sha256", "best_validation_accuracy"):
            if original[key] != optimized[key]:
                raise ExperimentStopped(f"Diagnostic removal changed {key} at seed {seed}")
        frozen = ARTIFACTS / f"books/ace-seed{seed}.txt"
        if digest(frozen.read_bytes()) != optimized["best_book_sha256"]:
            raise ExperimentStopped("Replayed selected book differs from the actual frozen book")
        runs.append({"seed": seed, "original_matched_requests": original["matched_requests"],
                     "optimized_matched_requests": optimized["matched_requests"],
                     "all_60_step_states_identical": True, "validation_checkpoints_identical": True,
                     "selected_book_matches_original_frozen_book": True,
                     "selected_book_sha256": optimized["best_book_sha256"]})
        print(json.dumps(runs[-1]), flush=True)
    training = json.loads((ARTIFACTS / "stage2_analysis.json").read_text())["groups"]["ace"]
    result = {"status": "exact_archive_replay_verified", "new_api_calls": 0, "runs": runs,
              "training_calls_before": training["learning_calls"],
              "training_calls_after": training["learning_without_diagnostic_calls"],
              "standardized_learning_cny_before": training["learning_full_cache_miss_cny"],
              "standardized_learning_cny_after": training["learning_without_diagnostic_full_miss_cny"],
              "claim_scope": "Exact state/prompt equivalence conditional on archived retained responses; "
                             "historical cost attribution, not a new paid retraining or live randomness guarantee. "
                             "Post-curation training accuracy is unavailable, not zero or copied from initial accuracy."}
    result["learning_call_reduction_percent"] = 100*(1-result["training_calls_after"]/result["training_calls_before"])
    result["standardized_learning_cost_reduction_percent"] = 100*(
        1-result["standardized_learning_cny_after"]/result["standardized_learning_cny_before"])
    result["prototype_source_sha256"] = digest((ROOT / "repro/diagnostic_free_replay.py").read_bytes())
    safe_json(ARTIFACTS / "diagnostic_free_replay.json", result)


if __name__ == "__main__":
    main()
