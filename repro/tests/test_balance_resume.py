import json
from pathlib import Path
import tempfile
import unittest

from repro.mock_client import mock_client
from repro.pilot import load_config
from repro.resume_balance import BalanceAwareReplay, reconcile_rejections
from repro.transport import ExperimentStopped


class BalanceResumeTests(unittest.TestCase):
    def test_only_explicit_402_rejections_are_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            from repro.transport import CostLedger
            ledger = CostLedger(Path(directory) / "usage.jsonl", "fixture", load_config())
            for code in (402, 500):
                ledger.append({"request_id": str(code), "run_id": "fixture", "state": "failed_billing_unknown",
                               "status_code": code, "accounted_cny": 1.0})
            self.assertEqual(reconcile_rejections(ledger), 1)
            rows = {row["status_code"]: row for row in ledger.records()}
            self.assertEqual(rows[402]["accounted_cny"], 0)
            self.assertEqual(rows[500]["accounted_cny"], 1)

    def test_only_known_rejected_slots_can_be_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = BalanceAwareReplay(load_config(), Path(directory) / "job", client=mock_client(), mock=True,
                                           ledger_path=Path(directory) / "usage.jsonl")
            transport.position = 1
            record = {"request_id": "rejected", "run_id": "job", "state": "rejected_402",
                      "accounted_cny": 0, "logical_index": 1, "prompt_fingerprint": "same"}
            transport.ledger.append(record)
            self.assertIsNone(transport._recover("same"))
            with self.assertRaises(ExperimentStopped):
                transport._recover("different")
