import json
from pathlib import Path
import tempfile
import unittest

from repro import ROOT
from ace import ACE
from eval.finance.data_processor import DataProcessor
from repro.evaluation import evaluate_fixed
from repro.methods import SelectiveACE, parse_entries, select_playbook
from repro.mock_client import mock_client
from repro.pilot import load_config
from repro.transport import ExperimentStopped, Transport, installed_transport


def run_sample(cls, question, target, step, **options):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        transport = Transport(load_config(), root / "job", client=mock_client(), mock=True,
                              ledger_path=root / "ledger.jsonl")
        log_dir = root / "logs"
        log_dir.mkdir()
        with installed_transport(transport):
            system = cls("fixture", "fixture", "fixture", "fixture", **options)
            before = system.playbook
            result = system._train_single_sample(
                task_dict={"question": question, "target": target, "context": ""},
                data_processor=DataProcessor("formula"), step_id=f"train_e_1_s_{step}", epoch=1, step=step,
                usage_log_path=str(root / "usage.jsonl"), log_dir=str(log_dir), total_samples=60,
                config_params=system._extract_config_params(load_config()),
            )
        calls = [(row["role"], row["call_id"]) for row in transport.ledger.records()]
        return result, before, system.playbook, calls


class SelectiveTests(unittest.TestCase):
    def test_disabled_switch_matches_original_call_sequence_and_book(self):
        args = ("SYNTHETIC_FAIL_ONCE: What is 3 + 4?", "7", 1)
        original = run_sample(ACE, *args)
        disabled = run_sample(SelectiveACE, *args, selective_enabled=False)
        self.assertEqual(original, disabled)

    def test_correct_sample_skips_without_duplicate_generation(self):
        result, before, after, calls = run_sample(SelectiveACE, "What is 8 - 5?", "3", 1)
        self.assertEqual(before, after)
        self.assertEqual(calls, [("generator", "train_e_1_s_1_gen_initial")])
        self.assertEqual(result[0], result[1])

    def test_every_fifth_correct_sample_retains_original_flow(self):
        args = ("What is 8 - 5?", "3", 5)
        self.assertEqual(run_sample(ACE, *args), run_sample(SelectiveACE, *args))

    def test_incorrect_sample_keeps_full_original_reflection_flow(self):
        args = ("SYNTHETIC_FAIL_ONCE: What is 3 + 4?", "7", 1)
        self.assertEqual(run_sample(ACE, *args), run_sample(SelectiveACE, *args))


class RetrievalTests(unittest.TestCase):
    def test_small_book_is_preserved_exactly(self):
        book = "## MATH\n[calc-00001] helpful=1 harmful=0 :: Check units.\n"
        result = select_playbook("units", book)
        self.assertEqual(result["text"], book)
        self.assertEqual(result["ids"], ["calc-00001"])

    def test_zero_match_uses_quality_then_stable_id(self):
        book = "## MATH\n[calc-00002] helpful=1 harmful=0 :: bond\n"
        book += "[calc-00001] helpful=1 harmful=0 :: apple\n"
        self.assertEqual(select_playbook("zzzx", book, max_entries=1)["ids"], ["calc-00001"])

    def test_whole_entries_obey_budget_including_headings(self):
        book = "## MATH\n[calc-00001] helpful=0 harmful=0 :: " + "mortgage " * 500
        book += "\n[calc-00002] helpful=1 harmful=0 :: Check units."
        result = select_playbook("mortgage", book, max_tokens=40)
        self.assertNotIn("calc-00001", result["ids"])
        self.assertLessEqual(result["selected_tokens"], 40)
        self.assertIn("## MATH", result["text"])

    def test_multiline_entry_content_survives(self):
        book = "## MATH\n[calc-00001] helpful=1 harmful=0 :: Check units.\nContinuation text."
        self.assertIn("Continuation text.", parse_entries(book)[0].content)


class EvaluationTests(unittest.TestCase):
    def test_failure_stays_in_denominator(self):
        class Generator:
            def generate(self, **kwargs):
                if kwargs["question"] == "fail":
                    raise ValueError("fixture")
                return json.dumps({"final_answer": "3"}), [], {"prompt_num_tokens": 1, "response_num_tokens": 1}

        data = [{"question": q, "context": "", "target": "3"} for q in ("fail", "pass")]
        result, details = evaluate_fixed(DataProcessor("formula"), Generator(), "", data)
        self.assertEqual((result["correct"], result["total"], result["accuracy"]), (1, 2, 0.5))
        self.assertEqual(len(details["individual_results"]), 2)

    def test_budget_stop_is_not_swallowed_as_a_wrong_answer(self):
        class Generator:
            def generate(self, **kwargs):
                raise ExperimentStopped("fixture")

        with self.assertRaises(ExperimentStopped):
            evaluate_fixed(DataProcessor("formula"), Generator(), "", [{"question": "q", "context": "", "target": "3"}])
