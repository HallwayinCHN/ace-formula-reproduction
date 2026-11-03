import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import openai

from repro import ROOT
from repro.pilot import load_config
from repro.prepare import ARTIFACTS, digest, pilot_data
from repro.transport import CostLedger, ExperimentStopped, Transport


class ManifestTests(unittest.TestCase):
    def test_pilot_is_disjoint_and_never_includes_held_out_test(self):
        manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
        self.assertEqual(set(pilot_data()), {"train", "validation"})
        for split, item in manifest["splits"].items():
            self.assertFalse(set(item["pilot_indices"]) & set(item["formal_order"]))
            self.assertEqual(len(item["pilot_indices"]), 0 if split == "test" else 10)

    def test_upstream_files_match_frozen_hashes(self):
        audit = json.loads((ARTIFACTS / "data_audit.json").read_text())
        for item in audit["splits"].values():
            self.assertEqual(digest((ROOT / item["path"]).read_bytes()), item["sha256"])


class LedgerTests(unittest.TestCase):
    def test_settlement_replaces_reservation_and_pending_cost_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CostLedger(Path(directory) / "usage.jsonl", "run", load_config())
            row = {"request_id": "r1", "run_id": "run", "accounted_cny": 2}
            ledger.append(row)
            ledger.append({**row, "accounted_cny": 0.5})
            ledger.append({"request_id": "r2", "run_id": "other", "accounted_cny": 3})
            self.assertEqual(ledger.totals(), (3.5, 0.5))

    def test_hard_limit_blocks_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = CostLedger(Path(directory) / "usage.jsonl", "run", load_config())
            with self.assertRaises(ExperimentStopped):
                ledger.reserve(21)
            ledger.append({"request_id": "r1", "run_id": "other", "accounted_cny": 179.99})
            with self.assertRaises(ExperimentStopped):
                ledger.reserve(0.02)


class TransportTests(unittest.TestCase):
    def make_transport(self, directory, create):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return Transport(load_config(), Path(directory) / "run", client=client,
                         ledger_path=Path(directory) / "ledger.jsonl")

    def test_timeout_attempts_are_bounded_and_unknown_billing_is_reserved(self):
        attempts = []

        def create(**kwargs):
            attempts.append(kwargs)
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://fixture.invalid"))

        with tempfile.TemporaryDirectory() as directory, patch("repro.transport.time.sleep"):
            transport = self.make_transport(directory, create)
            with self.assertRaises(ExperimentStopped):
                transport.call(None, "fixture", "fixture", "test", "generator", "test_1")
            self.assertEqual(len(attempts), 3)
            self.assertGreater(transport.ledger.totals()[0], 0)
            self.assertEqual(attempts[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_auth_error_is_not_retried(self):
        attempts = []

        def create(**kwargs):
            attempts.append(kwargs)
            response = httpx.Response(401, request=httpx.Request("POST", "https://fixture.invalid"))
            raise openai.AuthenticationError("fixture", response=response, body=None)

        with tempfile.TemporaryDirectory() as directory:
            transport = self.make_transport(directory, create)
            with self.assertRaises(ExperimentStopped):
                transport.call(None, "fixture", "fixture", "test", "generator", "test_1")
            self.assertEqual(len(attempts), 1)

    def test_missing_usage_keeps_pending_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = self.make_transport(directory, lambda **kwargs: SimpleNamespace(usage=None))
            with self.assertRaises(ExperimentStopped):
                transport.call(None, "fixture", "fixture", "test", "generator", "test_1")
            self.assertGreater(transport.ledger.totals()[0], 0)

    def test_truncated_response_is_saved_but_not_accepted(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4096), model="fixture",
            choices=[SimpleNamespace(message=SimpleNamespace(content="incomplete fixture"), finish_reason="length")],
        )
        with tempfile.TemporaryDirectory() as directory:
            transport = self.make_transport(directory, lambda **kwargs: response)
            transport.config["allow_truncated_nonempty"] = False
            with self.assertRaises(ExperimentStopped):
                transport.call(None, "fixture", "fixture", "test", "reflector", "train_1", log_dir=directory)
            record = transport.ledger.records()[0]
            self.assertFalse(record["response_valid"])
            saved = json.loads((Path(directory) / f"{record['request_id']}.json").read_text())
            self.assertEqual(saved["response"], "incomplete fixture")

    def test_upstream_compatibility_preserves_nonempty_truncated_text_and_marks_it(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4096), model="fixture",
            choices=[SimpleNamespace(message=SimpleNamespace(content="incomplete fixture"), finish_reason="length")],
        )
        with tempfile.TemporaryDirectory() as directory:
            transport = self.make_transport(directory, lambda **kwargs: response)
            text, info = transport.call(None, "fixture", "fixture", "test", "reflector", "train_1")
            self.assertEqual(text, "incomplete fixture")
            self.assertFalse(info["response_valid"])
            self.assertEqual(info["response_finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
