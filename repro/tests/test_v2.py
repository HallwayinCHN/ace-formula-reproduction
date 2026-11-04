from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ace.prompts.generator import GENERATOR_PROMPT
from repro.methods import parse_entries
from repro.v2_methods import brief_template, core_playbook, dependency_closure, numeric_answer
from repro.v2_screen import infer


BOOK = ("## FORMULAS & CALCULATIONS\n"
        "[calc-00001] helpful=1 harmful=0 :: Return on equity = income / equity * 100. See calc-00002.\n"
        "[calc-00002] helpful=1 harmful=0 :: Use consistent money units. See calc-00001.\n"
        "[calc-00003] helpful=0 harmful=0 :: Dividend price is unrelated.\n"
        "## COMMON MISTAKES TO AVOID\n"
        "[err-00001] helpful=1 harmful=0 :: Round the final result only.\n")


class ContextTests(unittest.TestCase):
    def test_retrieval_keeps_core_and_transitive_dependencies(self):
        result = core_playbook("Return on equity income equity", BOOK, top_k=1)
        self.assertEqual(result["mode"], "core_retrieval")
        self.assertEqual(set(result["ids"]), {"err-00001", "calc-00001", "calc-00002"})
        self.assertIn(parse_entries(BOOK)[0].text, result["text"])
        self.assertTrue(result["text"].startswith("## COMMON MISTAKES"))

    def test_overflow_uses_identical_full_book(self):
        self.assertEqual(core_playbook("return equity income", BOOK, token_budget=1)["text"], BOOK)

    def test_no_lexical_evidence_uses_full_book(self):
        result = core_playbook("quasars nebulae", BOOK)
        self.assertEqual(result["text"], BOOK)
        self.assertEqual(result["reason"], "weak_lexical_match")

    def test_missing_source_reference_is_reported_not_invented(self):
        entries = parse_entries(BOOK.replace("See calc-00002.", "See calc-99999."))
        ids, missing = dependency_closure({"calc-00001"}, entries)
        self.assertEqual(ids, {"calc-00001"})
        self.assertEqual(missing, {"calc-99999"})

    def test_brief_prompt_keeps_every_original_book_character(self):
        question = "What is return on equity?"
        prompt = brief_template().format(BOOK, "(empty)", question, "")
        self.assertIn(BOOK, prompt)
        self.assertIn(question, prompt)
        self.assertIn("60 words", prompt)
        self.assertIn("detailed analysis", GENERATOR_PROMPT)
        self.assertNotIn("detailed analysis", brief_template())


class OutputGuardTests(unittest.TestCase):
    def test_numeric_guard_rejects_infinity_but_not_finite_wrong_numbers(self):
        self.assertFalse(numeric_answer("NaN"))
        self.assertFalse(numeric_answer("inf"))
        self.assertFalse(numeric_answer("insufficient information"))
        self.assertTrue(numeric_answer("-999"))

    def info(self, request):
        return {"request_id": request, "prompt_num_tokens": 100, "response_num_tokens": 20,
                "response_finish_reason": "stop", "call_time": .1}

    def test_invalid_answer_retries_original_full_book_once_and_counts_both_calls(self):
        generator = Mock()
        generator.generate.side_effect = [("no output", [], self.info("a")),
                                          ('{"final_answer":"3"}', [], self.info("b"))]
        item = infer(generator, None, {"sample_id": 0, "question": "8 - 5", "context": ""},
                     BOOK, "brief_guard", Path("/tmp/unused-v2-fixture"))
        self.assertTrue(item["fallback_used"])
        self.assertEqual(item["final_answer"], "3")
        self.assertEqual(item["output_tokens"], 40)
        self.assertEqual(item["request_ids"], ["a", "b"])
        self.assertEqual(generator.generate.call_args_list[1].args[1], BOOK)

    def test_finite_wrong_answer_does_not_trigger_label_oracle(self):
        generator = Mock()
        generator.generate.return_value = ('{"final_answer":"-999"}', [], self.info("a"))
        item = infer(generator, None, {"sample_id": 0, "question": "8 - 5", "context": ""},
                     BOOK, "brief_guard", Path("/tmp/unused-v2-fixture"))
        self.assertFalse(item["fallback_used"])
        self.assertEqual(generator.generate.call_count, 1)

    def test_original_control_does_not_gain_new_fallback_behavior(self):
        generator = Mock()
        generator.generate.return_value = ("no output", [], self.info("a"))
        item = infer(generator, None, {"sample_id": 0, "question": "8 - 5", "context": ""},
                     BOOK, "original", Path("/tmp/unused-v2-fixture"))
        self.assertFalse(item["fallback_used"])
        self.assertEqual(generator.generate.call_count, 1)
