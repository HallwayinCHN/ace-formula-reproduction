from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repro.formal import formal_data, run_job
from repro.mock_client import mock_client, synthetic_data
from repro.pilot import load_config
from repro.prepare import load_rows
from repro.replay import ReplayTransport
from repro.transport import CostLedger, ExperimentStopped


class AtomicBudgetTests(unittest.TestCase):
    def test_concurrent_reservations_cannot_both_exceed_run_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CostLedger(Path(directory) / "ledger.jsonl", "same-run", load_config())

            def reserve(index):
                try:
                    ledger.reserve_and_append({"request_id": str(index), "run_id": "same-run", "accounted_cny": 12})
                    return True
                except ExperimentStopped:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(reserve, (1, 2)))
            self.assertEqual(sum(outcomes), 1)
            self.assertEqual(ledger.totals()[0], 12)


class ReplayTests(unittest.TestCase):
    def transport(self, directory):
        root = Path(directory)
        return ReplayTransport(load_config(), root / "job", client=mock_client(), mock=True,
                               ledger_path=root / "ledger.jsonl")

    def call(self, transport, text="What is 8 - 5?"):
        prompt = f"You are an analysis expert\n**Question:**\n{text}\n**Context:**\n"
        return transport.call(None, "fixture", "fixture", prompt, "generator", "train_e_1_s_1",
                              log_dir=str(transport.output / "attempts/one/logs"))

    def test_completed_call_is_replayed_without_new_request(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.transport(directory)
            original = self.call(first)
            restarted = self.transport(directory)
            self.assertEqual(self.call(restarted), original)
            self.assertEqual((restarted.calls, restarted.replayed), (0, 1))
            self.assertEqual(len(restarted.ledger.records()), 1)

    def test_changed_prompt_is_rejected_without_a_new_request(self):
        with tempfile.TemporaryDirectory() as directory:
            self.call(self.transport(directory))
            restarted = self.transport(directory)
            with self.assertRaises(ExperimentStopped):
                self.call(restarted, "Different question")
            self.assertEqual(restarted.calls, 0)

    def test_orphaned_completed_request_is_not_blindly_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.transport(directory)
            self.call(first)
            for path in first.output.rglob("*.json"):
                path.unlink()
            restarted = self.transport(directory)
            with self.assertRaises(ExperimentStopped):
                self.call(restarted)
            self.assertEqual(restarted.calls, 0)


class FormalDataTests(unittest.TestCase):
    def test_training_loader_never_reads_held_out_test_records(self):
        with patch("repro.formal.load_rows", wraps=load_rows) as spy:
            data, _, manifest = formal_data(42)
        self.assertEqual([call.args[0] for call in spy.call_args_list], ["train", "validation"])
        self.assertEqual((len(data["train"]), len(data["validation"])), (60, 30))
        self.assertNotIn("test", data)
        self.assertTrue(manifest["formal_size_locked"])

    def test_complete_training_pipeline_with_mock_http_for_both_variants(self):
        from eval.finance.data_processor import DataProcessor

        for variant in ("ace", "selective"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "mock-job"
                output.mkdir()
                config = {**load_config(), "model": "MOCK-NOT-A-MODEL", "variant": variant, "order_seed": 42}

                def transport(config, output):
                    return ReplayTransport(config, output, client=mock_client(), mock=True,
                                           ledger_path=Path(directory) / "ledger.jsonl")

                prepared = (config, synthetic_data(), DataProcessor("formula"), output)
                with patch("repro.formal.initialize_job", return_value=prepared):
                    with patch("repro.formal.ReplayTransport", side_effect=transport):
                        run_job(variant, 42)
                summary = json.loads((output / "summary.json").read_text())
                self.assertTrue(summary["mock"])
                self.assertFalse(summary["held_out_test_loaded"])
                self.assertEqual(len(summary["checkpoints"]), 2)
                self.assertEqual(summary["skipped_correct"], 0 if variant == "ace" else 1)
