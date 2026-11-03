"""Replay completed logical calls on restart; never blindly retry an unresolved prior request."""

import hashlib
import json
from pathlib import Path

from repro.transport import ExperimentStopped, Transport


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class ReplayTransport(Transport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = 0
        self.replayed = 0

    def call(self, client, api_provider, model, prompt, role, call_id, max_tokens=4096,
             log_dir=None, use_json_mode=False, **unused):
        self.position += 1
        if self.position > self.config["max_calls"]:
            raise ExperimentStopped("Logical-call sequence limit reached")
        material = [model, prompt, role, call_id, max_tokens, use_json_mode,
                    self.config["temperature"], self.config["thinking"]]
        fingerprint = hashlib.sha256(json.dumps(material, ensure_ascii=False).encode()).hexdigest()
        path = self.output / "replay" / f"{self.position:05}.json"
        saved = json.loads(path.read_text()) if path.exists() else self._recover(fingerprint)
        if saved is not None:
            if saved["fingerprint"] != fingerprint:
                raise ExperimentStopped(f"Replay diverged at logical call {self.position}; no request sent")
            self.replayed += 1
            atomic_json(path, saved)
            if log_dir:
                atomic_json(Path(log_dir) / f"{saved['info']['request_id']}.json", saved["info"])
            return saved["text"], saved["info"]
        self.request_context = {"logical_index": self.position, "prompt_fingerprint": fingerprint}
        text, info = super().call(client, api_provider, model, prompt, role, call_id, max_tokens,
                                  log_dir=log_dir, use_json_mode=use_json_mode, **unused)
        atomic_json(path, {"fingerprint": fingerprint, "text": text, "info": info})
        return text, info

    def _recover(self, fingerprint):
        records = [row for row in self.ledger.records()
                   if row["run_id"] == self.run_id and row.get("logical_index") == self.position]
        if not records:
            return None
        if any(row.get("prompt_fingerprint") != fingerprint for row in records):
            raise ExperimentStopped("A prior logical request has a different fingerprint")
        for row in reversed(records):
            if row["state"] == "completed":
                matches = list(self.output.glob(f"attempts/**/{row['request_id']}.json"))
                if matches:
                    info = json.loads(matches[0].read_text())
                    if info.get("response"):
                        return {"fingerprint": fingerprint, "text": info["response"], "info": info}
        raise ExperimentStopped("Prior request has unknown completion/billing; reconcile before restarting")
