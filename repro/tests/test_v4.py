from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import openai

from repro.mock_client import mock_client
from repro.pilot import load_config
from repro.prepare import digest
from repro.recovery import safe_json
from repro.transport import ExperimentStopped
from repro.v4_analysis import collect, compute
from repro.v4_job import run
from repro.v4_protocol import read_rows
from repro.v4_transport import Overloaded, PhaseLedger, PhaseReplay


class DataBoundaryTests(unittest.TestCase):
    def test_unselected_invalid_json_is_not_parsed_and_agent_has_no_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.jsonl"
            row = {"context": 'Question: "What is 8 - 5?". Answer:', "target": "CANARY_HELDOUT_LABEL"}
            path.write_text("UNSELECTED INVALID JSON\n"+json.dumps(row)+"\n")
            with patch("repro.v4_protocol.DATA", path):
                result = read_rows([1])
            self.assertEqual(result[0]["sample_id"], 1)
            self.assertNotIn("target", result[0])
            self.assertNotIn("CANARY_HELDOUT_LABEL", json.dumps(result))

    def test_target_is_masked_before_upstream_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.jsonl"
            path.write_text(json.dumps({"context": "fixture", "target": "CANARY"})+"\n")
            with patch("repro.v4_protocol.DATA", path):
                with patch("eval.finance.data_processor.DataProcessor.process_task_data",
                           return_value=[{"question": "fixture", "context": "", "target": ""}]) as processor:
                    read_rows([0])
            self.assertEqual(processor.call_args.args[0], [{"context": "fixture", "target": ""}])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ExperimentStopped):
            read_rows([1, 1])

    def test_incomplete_jobs_cannot_load_scoring_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("repro.v4_analysis.STATE", Path(directory)):
                with patch("repro.v4_analysis.verify", return_value={"variants": ["original", "nodiag"],
                           "seeds": [42, 43, 44], "evaluation_ids": [1]}):
                    with patch("repro.v4_analysis.read_rows") as labels:
                        with self.assertRaises(ExperimentStopped):
                            collect()
            labels.assert_not_called()

    def test_changed_predictions_block_labels_before_unblinding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root/"contract.json"
            safe_json(contract_path, {"fixture": True})
            folder = root/"v4-original-seed42"
            safe_json(folder/"predictions.json", [{"sample_id": 1, "request_id": "fixture"}])
            safe_json(folder/"summary.json", {"state": "completed", "items": 1,
                      "contract_sha256": digest(contract_path.read_bytes()), "book_sha256": "book",
                      "targets_supplied_to_agent": False, "predictions_sha256": "corrupt"})
            c = {"variants": ["original", "nodiag"], "seeds": [42, 43, 44], "evaluation_ids": [1],
                 "books": {"original-seed42": {"sha256": "book"}}}
            with patch("repro.v4_analysis.STATE", root), patch("repro.v4_analysis.CONTRACT", contract_path):
                with patch("repro.v4_analysis.verify", return_value=c), patch("repro.v4_analysis.read_rows") as labels:
                    with self.assertRaises(ExperimentStopped):
                        collect()
            labels.assert_not_called()


class TransportTests(unittest.TestCase):
    def test_two_yuan_failure_cap_retains_prior_stage_reserves_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhaseLedger(Path(directory)/"ledger.jsonl", "v4-fixture", load_config())
            ledger.append({"request_id": "old-stage", "run_id": "v3-prior", "state": "failed_billing_unknown",
                           "accounted_cny": 3})
            ledger.append({"request_id": "failed", "run_id": "v4-prior", "state": "failed_billing_unknown",
                           "accounted_cny": 1.9})
            ledger.reserve_and_append({"request_id": "one", "run_id": "v4-fixture", "state": "pending",
                                       "accounted_cny": .05})
            with self.assertRaises(ExperimentStopped):
                ledger.reserve_and_append({"request_id": "two", "run_id": "v4-fixture", "state": "pending",
                                           "accounted_cny": .1})
            self.assertAlmostEqual(ledger.totals()[0], 4.95)

    def test_ten_yuan_phase_limit_is_atomic_across_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = PhaseLedger(Path(directory)/"ledger.jsonl", "v4-fixture", load_config())
            ledger.append({"request_id": "existing", "run_id": "v4-prior", "state": "completed",
                           "accounted_cny": 9.7})

            def reserve(index):
                try:
                    ledger.reserve_and_append({"request_id": str(index), "run_id": f"v4-fixture-{index}",
                                               "state": "pending", "accounted_cny": .2})
                    return True
                except ExperimentStopped:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(pool.map(reserve, (1, 2))), 1)
            self.assertAlmostEqual(ledger.totals()[0], 9.9)

    def exercise(self, error, recover):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("repro.v4_transport.CIRCUIT", root/"circuit"), patch("repro.v4_transport.STOP", root/"stop"):
                def transport():
                    return PhaseReplay(load_config(), root/"v4-fixture", client=mock_client(),
                                       ledger_path=root/"ledger.jsonl")

                def call(value):
                    return value.call(None, "fixture", "fixture", "**Question:**\nWhat is 8 - 5?\n**Context:**\n",
                                      "generator", "v4_1", log_dir=value.output/"attempts/calls")

                first = transport()
                with patch.object(first.client.chat.completions, "create", side_effect=error) as create:
                    with self.assertRaises(Overloaded if recover else ExperimentStopped):
                        call(first)
                self.assertEqual(create.call_count, 1)
                prior = first.ledger.records()[0]
                self.assertGreater(prior["accounted_cny"], 0)
                if recover:
                    (root/"circuit").unlink()
                    second = transport()
                    call(second)
                    self.assertEqual(second.ledger.records()[0], prior)
                    replay = transport()
                    call(replay)
                    self.assertEqual(replay.calls, 0)

    def test_503_recovery_retains_reserve_and_replays_paid_response(self):
        response = httpx.Response(503, request=httpx.Request("POST", "https://fixture.invalid"))
        self.exercise(openai.InternalServerError("overload", response=response, body={}), recover=True)

    def test_timeout_does_not_trigger_blind_retry(self):
        self.exercise(openai.APITimeoutError(request=httpx.Request("POST", "https://fixture.invalid")), recover=False)


class InferenceAndStatisticsTests(unittest.TestCase):
    def test_mock_job_is_one_call_per_question_and_completed_job_is_not_recharged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root/"contract.json"
            safe_json(contract_path, {"fixture": True})
            config = {**load_config(), "max_calls": 2}
            c = {"variants": ["original", "nodiag"], "seeds": [42, 43, 44], "evaluation_ids": [0, 1],
                 "books": {"original-seed42": {"sha256": digest(b"fixture book")}}, "inference_config": config}
            inputs = [{"sample_id": i, "question": "What is 8 - 5?", "context": ""} for i in (0, 1)]
            client = mock_client()

            def transport(cfg, output):
                return PhaseReplay(cfg, output, client=client, mock=True, ledger_path=root/"ledger.jsonl")

            with patch("repro.v4_job.STATE", root), patch("repro.v4_job.CONTRACT", contract_path):
                with patch("repro.v4_job.verify", return_value=c), patch("repro.v4_job.read_rows", return_value=inputs):
                    with patch("repro.v4_job.book_for", return_value="fixture book"), patch("repro.v4_job.load_dotenv"):
                        with patch("repro.v4_job.PhaseReplay", side_effect=transport):
                            with patch.object(client.chat.completions, "create", wraps=client.chat.completions.create) as api:
                                run("original", 42)
                                run("original", 42)
                                self.assertEqual(api.call_count, 2)
            items = json.loads((root/"v4-original-seed42/predictions.json").read_text())
            self.assertEqual([row["final_answer"] for row in items], ["3", "3"])
            self.assertTrue(all("target" not in row for row in items))

    def fixture(self):
        c = {"evaluation_ids": [10, 20], "seeds": [42, 43, 44], "variants": ["original", "nodiag"],
             "predictions": 12, "origin": "fixture", "inference_config": load_config(),
             "operational_accuracy_target": {"minimum_delta_pp": -1, "minimum_ci95_lower_pp": -2}}
        rows = [{"variant": v, "seed": seed, "sample_id": i, "is_correct": True, "prompt_tokens": 100,
                 "output_tokens": 10, "cached_tokens": 0, "length_response": False, "final_answer": "1",
                 "api_seconds": 1, "end_to_end_seconds": 1.1}
                for v in c["variants"] for seed in c["seeds"] for i in c["evaluation_ids"]]
        usage = [{"run_id": f"v4-{v}-seed42", "accounted_cny": .1} for v in c["variants"]]
        training = {"groups": {v: {"all_workflow_accounted_cny": 1} for v in c["variants"]}}
        return c, rows, usage, training

    def test_question_cluster_and_failure_gate_are_used(self):
        c, rows, usage, training = self.fixture()
        rows[6]["is_correct"] = False
        result = compute(rows, usage, c, training)
        self.assertEqual(result["primary_comparison"]["clusters"], 2)
        self.assertEqual(result["groups"]["original"]["predictions"], 6)
        self.assertFalse(result["operational_accuracy_target_passed"])
        self.assertFalse(result["formal_noninferiority_proved"])

    def test_fixed_denominator_rejects_missing_answer_record(self):
        c, rows, usage, training = self.fixture()
        with self.assertRaises(ValueError):
            compute(rows[:-1], usage, c, training)
