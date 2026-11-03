import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repro.holdout import book_for, run_job, test_inputs
from repro.holdout_analysis import collect_predictions
from repro.holdout_stats import paired_interval
from repro.mock_client import mock_client
from repro.pilot import load_config
from repro.prepare import digest
from repro.resume_balance import BalanceAwareReplay


class BlindingTests(unittest.TestCase):
    def test_different_labels_produce_identical_label_free_inputs(self):
        manifest = {"selected_indices": {"test": [0]}, "split_hashes": {"test": digest(b"fixture")}}
        context = 'Use formula. Question: "What is 8 - 5?". Answer:'
        inputs = []
        for target in ("FORBIDDEN_TARGET_A", "FORBIDDEN_TARGET_B"):
            with patch("repro.holdout.load_rows", return_value=[{"context": context, "target": target}]):
                with patch("pathlib.Path.read_bytes", return_value=b"fixture"):
                    inputs.append(test_inputs(manifest))
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(set(inputs[0][0]), {"sample_id", "question", "context"})
        self.assertNotIn("FORBIDDEN_TARGET", json.dumps(inputs))

    def test_missing_job_blocks_label_loading(self):
        with patch("repro.holdout_analysis.load_rows") as labels:
            with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    collect_predictions()
            labels.assert_not_called()

    def test_all_five_conditions_run_generation_only_and_remain_unscored(self):
        for variant in ("base", "ace", "selective", "retrieval", "combined"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "mock-holdout"
                output.mkdir()
                config = {**load_config(), "model": "MOCK-NOT-A-MODEL", "variant": variant,
                          "book_sha256": "fixture", "contract_sha256": "fixture"}
                inputs = [{"sample_id": 3, "question": "What is 8 - 5?", "context": ""}]
                book = "## MATH\n[calc-00001] helpful=1 harmful=0 :: Subtract carefully."

                def transport(config, output, client, mock):
                    return BalanceAwareReplay(config, output, client=client, mock=mock,
                                               ledger_path=Path(directory) / "ledger.jsonl")

                with patch("repro.holdout.initialize_job", return_value=(config, inputs, book, output)):
                    with patch("repro.holdout.BalanceAwareReplay", side_effect=transport):
                        with patch("ace.core.reflector.Reflector.__init__", side_effect=AssertionError):
                            with patch("ace.core.curator.Curator.__init__", side_effect=AssertionError):
                                run_job(variant, 42, client=mock_client(), mock=True)
                predictions = json.loads((output / "predictions.json").read_text())
                self.assertEqual(predictions[0]["final_answer"], "3")
                self.assertNotIn("target", predictions[0])
                self.assertNotIn("is_correct", predictions[0])
                saved = (output / "predictions.json").read_bytes()
                with patch("repro.holdout.initialize_job", return_value=(config, inputs, book, output)):
                    run_job(variant, 42, client=None, mock=True)
                self.assertEqual(saved, (output / "predictions.json").read_bytes())


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_effect_has_exact_interval(self):
        result = paired_interval([[1, 1, 1]] * 5, [[0, 0, 0]] * 5)
        self.assertEqual(result["delta_percentage_points"], 100)
        self.assertEqual(result["ci95_percentage_points"], [100, 100])

    def test_seeds_are_kept_inside_question_clusters(self):
        result = paired_interval([[1, 0, 1]] * 5, [[0, 0, 1]] * 5)
        self.assertAlmostEqual(result["delta_percentage_points"], 100 / 3)
        self.assertAlmostEqual(result["ci95_percentage_points"][0], 100 / 3)
        self.assertAlmostEqual(result["ci95_percentage_points"][1], 100 / 3)

    def test_wrong_seed_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_interval([[1, 0]], [[0, 0]])
