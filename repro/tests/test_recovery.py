from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import openai

from repro import auto_resume
from repro.holdout import run_job
from repro.mock_client import mock_client
from repro.pilot import load_config
from repro.recovery import OverloadAwareReplay, ProbeFinished, RecoveryLedger, ServiceUnavailable
from repro.transport import ExperimentStopped


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.contract = {"created_at_utc": "2020-01-01T00:00:00+00:00",
                         "policy": {"max_extra_failure_reserve_cny": 5}}
        self.contract_path = self.root / "contract.json"
        self.contract_path.write_text("{}")
        for name, value in (("CIRCUIT", self.root / "circuit.json"), ("STOP", self.root / "stop"),
                            ("CONTRACT", self.contract_path)):
            manager = patch("repro.recovery." + name, value)
            manager.start()
            self.addCleanup(manager.stop)

    def transport(self, probe=False):
        return OverloadAwareReplay(load_config(), self.root / "holdout-fixture", client=mock_client(),
                                   ledger_path=self.root / "ledger.jsonl", recovery=self.contract, probe=probe)

    def call(self, transport, text="What is 8 - 5?"):
        prompt = f"You are an analysis expert\n**Question:**\n{text}\n**Context:**\n"
        return transport.call(None, "fixture", "fixture", prompt,
                              "generator", "fixture", log_dir=transport.output / "attempts/one/calls")

    def overload(self, transport):
        response = httpx.Response(503, request=httpx.Request("POST", "https://fixture.invalid"))
        error = openai.InternalServerError("overloaded", response=response, body={})
        with patch.object(transport.client.chat.completions, "create", side_effect=error) as requests:
            with self.assertRaises(ServiceUnavailable):
                self.call(transport)
        self.assertEqual(requests.call_count, 1)

    def test_503_resume_keeps_previous_reserve_and_success_is_replayed(self):
        first = self.transport()
        self.overload(first)
        failed = first.ledger.records()[0]
        self.assertGreater(failed["accounted_cny"], 0)
        self.assertEqual(failed["state"], "failed_billing_unknown")
        (self.root / "circuit.json").unlink()
        restarted = self.transport()
        result = self.call(restarted)
        self.assertEqual(restarted.ledger.records()[0], failed)
        again = self.transport()
        self.assertEqual(self.call(again), result)
        self.assertEqual((again.calls, again.replayed), (0, 1))
        self.assertEqual(len(again.ledger.records()), 2)

    def test_503_prompt_mismatch_cannot_restart(self):
        self.overload(self.transport())
        (self.root / "circuit.json").unlink()
        restarted = self.transport()
        with self.assertRaises(ExperimentStopped):
            self.call(restarted, "Changed prompt")
        self.assertEqual(restarted.calls, 0)

    def test_timeout_is_not_reissued_or_reclassified_as_free(self):
        first = self.transport()
        error = openai.APITimeoutError(request=httpx.Request("POST", "https://fixture.invalid"))
        with patch.object(first.client.chat.completions, "create", side_effect=error) as requests:
            with self.assertRaises(ExperimentStopped):
                self.call(first)
        self.assertEqual(requests.call_count, 1)
        self.assertGreater(first.ledger.records()[0]["accounted_cny"], 0)
        restarted = self.transport()
        with patch.object(restarted.client.chat.completions, "create") as requests:
            with self.assertRaises(ExperimentStopped):
                self.call(restarted)
        requests.assert_not_called()

    def test_pending_response_is_not_reissued(self):
        transport = self.transport()
        self.overload(transport)
        record = transport.ledger.records()[0]
        record.update(state="pending", status_code=None)
        transport.ledger.append(record)
        (self.root / "circuit.json").unlink()
        restarted = self.transport()
        with self.assertRaises(ExperimentStopped):
            self.call(restarted)
        self.assertEqual(restarted.calls, 0)

    def test_shared_circuit_prevents_new_requests(self):
        (self.root / "circuit.json").write_text("{}")
        transport = self.transport()
        with self.assertRaises(ServiceUnavailable):
            self.call(transport)
        self.assertEqual(transport.ledger.records(), [])

    def test_one_item_probe_and_resume_preserve_existing_prediction_bytes(self):
        output = self.root / "holdout-fixture"
        output.mkdir()
        config = {**load_config(), "variant": "base", "book_sha256": None, "contract_sha256": "fixture"}
        inputs = [{"sample_id": i, "question": "What is 8 - 5?", "context": ""} for i in (3, 4)]

        def factory(probe):
            return lambda config, output, **options: OverloadAwareReplay(
                config, output, client=mock_client(), mock=True, probe=probe, recovery=self.contract,
                ledger_path=self.root / "ledger.jsonl")

        with patch("repro.holdout.initialize_job", return_value=(config, inputs, "", output)):
            with patch("repro.holdout.BalanceAwareReplay", side_effect=factory(True)):
                with self.assertRaises(ProbeFinished):
                    run_job("base", 42)
            saved = (output / "items/0003.json").read_bytes()
            self.assertFalse((output / "items/0004.json").exists())
            with patch("repro.holdout.BalanceAwareReplay", side_effect=factory(False)):
                run_job("base", 42)
        self.assertEqual((output / "items/0003.json").read_bytes(), saved)
        predictions = json.loads((output / "predictions.json").read_text())
        self.assertEqual(len(predictions), 2)
        self.assertFalse(any("target" in item or "is_correct" in item for item in predictions))

    def test_failure_budget_reservation_is_atomic_across_concurrent_requests(self):
        contract = {**self.contract, "policy": {"max_extra_failure_reserve_cny": .07}}
        ledger = RecoveryLedger(self.root / "ledger.jsonl", "holdout-fixture", load_config(), contract)

        def reserve(index):
            try:
                ledger.reserve_and_append({"request_id": str(index), "run_id": "holdout-fixture",
                                           "accounted_cny": .05, "timestamp": "2030-01-01T00:00:00+00:00",
                                           "state": "pending"})
                return True
            except ExperimentStopped:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(reserve, (1, 2))), 1)
        self.assertAlmostEqual(ledger.totals()[0], .05)


class HealthTests(unittest.TestCase):
    def test_unavailable_balance_blocks_automatic_resume(self):
        with patch("repro.auto_resume.get_balance", return_value={"is_available": False}):
            with self.assertRaises(ExperimentStopped):
                auto_resume.control_health()

    def test_health_failure_never_starts_a_paid_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = root / "stop"
            contract = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "policy": {"maximum_watch_hours": 24, "interval_seconds": 0}}
            with patch("repro.auto_resume.STOP", stop), patch("repro.auto_resume.STATUS", root / "status.json"):
                with patch("repro.auto_resume.counts", return_value={"predictions_completed": 220}):
                    controller = auto_resume.Controller(contract)
                    controller.guard = lambda: None
                    controller.pending_job = lambda: (42, "ace")

                    def unhealthy():
                        stop.touch()
                        return {"healthy": False}

                    with patch("repro.auto_resume.control_health", side_effect=unhealthy):
                        with patch.object(controller, "probe") as probe, patch.object(controller, "batch") as batch:
                            controller.run()
                    probe.assert_not_called()
                    batch.assert_not_called()
