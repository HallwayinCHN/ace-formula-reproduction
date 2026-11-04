from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import openai

from repro.mock_client import mock_client, synthetic_data
from repro.pilot import load_config
from repro.recovery import safe_json, utcnow
from repro.transport import ExperimentStopped, Transport, installed_transport
from repro.v3_analysis import collect, simulated
from repro.v3_jobs import inference_job, training_job
from repro.v3_router import choose, fit, profile
from repro.v3_transport import Overloaded, PhaseLedger, PhaseReplay


class RouterTests(unittest.TestCase):
    def test_upstream_unit_example_does_not_make_every_question_scaled(self):
        text = "Calculate profit from revenue 100 and expense 40."
        suffix = " Your answer should be a plain floating point number, e.g. 5 million should be 5000000.0."
        self.assertEqual(profile(text), profile(text+suffix))
        self.assertEqual(profile(text)["bucket"], "plain_numeric")

    def test_protected_complex_question_cannot_be_enabled_by_allowlist(self):
        policy = {"allowed_buckets": ["protected_complex"]}
        self.assertFalse(choose("Compute the Black-Scholes option value with 100 and 90.", policy)[0])

    def pairs(self, count, harmful=False):
        return [{"sample_id": index, "profile": {"bucket": "plain_numeric"},
                 "original_correct": [True, True, True], "brief_correct": [not harmful, True, True],
                 "original_cost": [1, 1, 1], "brief_cost": [.5, .5, .5]} for index in range(count)]

    def test_support_counts_questions_and_harm_vetoes_bucket(self):
        self.assertEqual(fit(self.pairs(2), 3)["allowed_buckets"], [])
        self.assertEqual(fit(self.pairs(3), 3)["allowed_buckets"], ["plain_numeric"])
        self.assertEqual(fit(self.pairs(5, harmful=True), 3)["allowed_buckets"], [])

    def test_simulated_gate_does_not_accept_savings_with_accuracy_loss(self):
        pairs = self.pairs(5, harmful=True)
        result = simulated(pairs, {item["sample_id"]: True for item in pairs})
        self.assertEqual(result["estimated_cost_reduction_percent"], 50)
        self.assertFalse(result["passed"])

    def test_failed_router_gate_blocks_evaluation_loader(self):
        with patch("repro.v3_jobs.verify", return_value={"eval_variants": ["adaptive"]}):
            with patch("repro.v3_jobs.rule", side_effect=ExperimentStopped("gate failed")):
                with patch("repro.v3_jobs.selected_rows") as rows:
                    with self.assertRaises(ExperimentStopped):
                        inference_job("eval", "adaptive", 42)
        rows.assert_not_called()

    def test_incomplete_evaluation_cannot_load_labels(self):
        with patch("repro.v3_analysis.verify", return_value={"eval_variants": ["original"], "evaluation_ids": [1]}):
            with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
                with patch("repro.v3_analysis.selected_rows") as labels:
                    with self.assertRaises(FileNotFoundError):
                        collect("eval")
        labels.assert_not_called()


class PhaseBudgetTests(unittest.TestCase):
    def test_failure_reservations_are_atomic_and_not_erased_by_503_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = PhaseLedger(root / "ledger.jsonl", "v3-test", {**load_config(), "run_limit_cny": 30})

            def reserve(index):
                try:
                    ledger.reserve_and_append({"request_id": str(index), "run_id": f"v3-test-{index}",
                                               "accounted_cny": 3, "state": "pending"})
                    return True
                except ExperimentStopped:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(sum(pool.map(reserve, (1, 2))), 1)
            self.assertEqual(ledger.totals()[0], 3)

    def test_explicit_503_retry_retains_old_cost_and_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("repro.v3_transport.CIRCUIT", root / "circuit.json"), patch("repro.v3_transport.STOP", root/"stop"):
                def transport():
                    return PhaseReplay(load_config(), root/"v3-fixture", client=mock_client(),
                                       ledger_path=root/"ledger.jsonl")

                def call(value):
                    return value.call(None, "fixture", "fixture",
                                      "You are an analysis expert\n**Question:**\n8 - 5\n**Context:**\n",
                                      "generator", "sample", log_dir=value.output/"attempts/calls")

                first = transport()
                response = httpx.Response(503, request=httpx.Request("POST", "https://fixture.invalid"))
                error = openai.InternalServerError("overload", response=response, body={})
                with patch.object(first.client.chat.completions, "create", side_effect=error):
                    with self.assertRaises(Overloaded):
                        call(first)
                prior = first.ledger.records()[0]
                self.assertGreater(prior["accounted_cny"], 0)
                (root/"circuit.json").unlink()
                restarted = transport()
                call(restarted)
                self.assertEqual(restarted.ledger.records()[0], prior)
                replayed = transport()
                call(replayed)
                self.assertEqual(replayed.calls, 0)


class LiveDriverFixtureTests(unittest.TestCase):
    def test_training_driver_retains_learning_and_does_not_fabricate_post_accuracy(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            contract = root/"contract.json"
            safe_json(contract, {"fixture": True})
            data = synthetic_data()
            prepared = {"train": data["train"]*30, "validation": data["validation"]*30}
            from eval.finance.data_processor import DataProcessor
            processor = DataProcessor("formula")
            summaries = {}
            for variant in ("original", "nodiag"):
                output, attempt = root/variant, root/variant/"attempt"
                attempt.mkdir(parents=True)
                safe_json(output/"started.json", {"started_at_utc": utcnow()})
                config = {**load_config(), "max_calls": 650}
                transport = Transport(config, output, client=mock_client(), mock=True,
                                      ledger_path=root/f"{variant}.jsonl")

                @contextmanager
                def fake_job(*args):
                    with installed_transport(transport):
                        yield config, output, attempt, transport

                with patch("repro.v3_jobs.formal_data", return_value=(prepared, processor, {})):
                    with patch("repro.v3_jobs.job", side_effect=fake_job), patch("repro.v3_jobs.CONTRACT", contract):
                        training_job(variant, 42)
                summaries[variant] = json.loads((output/"summary.json").read_text())
                if variant == "nodiag":
                    trace = json.loads((output/"trace.json").read_text())
                    self.assertTrue(all(row["post_train_result"]["is_correct"] is None for row in trace))
            self.assertEqual(summaries["original"]["diagnostic_calls"], 60)
            self.assertEqual(summaries["nodiag"]["diagnostic_calls"], 0)
            self.assertEqual(summaries["original"]["best_book_sha256"], summaries["nodiag"]["best_book_sha256"])
