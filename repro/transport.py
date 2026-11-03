"""Bounded transport for the unmodified upstream agents, with a persistent cost ledger."""

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import time
import uuid

import openai
import tiktoken

from repro import ROOT


class ExperimentStopped(BaseException):
    """Fatal stop that must propagate through upstream's broad Exception handlers."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class CostLedger:
    def __init__(self, path, run_id, config):
        self.path, self.run_id, self.config = Path(path), run_id, config

    def records(self):
        latest = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                latest[row["request_id"]] = row
        return list(latest.values())

    def totals(self):
        records = self.records()
        total = sum(row["accounted_cny"] for row in records)
        run = sum(row["accounted_cny"] for row in records if row["run_id"] == self.run_id)
        return total, run

    def reserve(self, amount):
        total, run = self.totals()
        if total + amount > self.config["project_limit_cny"]:
            raise ExperimentStopped("Project cost limit reached; no new request was sent")
        if run + amount > self.config["pilot_limit_cny"]:
            raise ExperimentStopped("Pilot cost limit reached; no new request was sent")

    def append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Transport:
    def __init__(self, config, output, client=None, mock=False, ledger_path=None):
        self.config, self.output, self.mock = config, Path(output), mock
        self.output.mkdir(parents=True, exist_ok=True)
        self.run_id = self.output.name
        self.calls = 0
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.client = client or self._live_client()
        ledger_name = "mock_usage.jsonl" if mock else "live_usage.jsonl"
        self.ledger = CostLedger(ledger_path or ROOT / "results/repro" / ledger_name, self.run_id, config)

    def _live_client(self):
        key = os.environ.get(self.config["key_env"], "").strip()
        if not key or key.startswith("your_"):
            raise ExperimentStopped(f"Set {self.config['key_env']} in the ignored repository .env first")
        return openai.OpenAI(
            api_key=key, base_url=self.config["base_url"],
            timeout=self.config["timeout_seconds"], max_retries=0,
        )

    def cost(self, prompt_tokens, completion_tokens, cached_tokens=0):
        c = self.config
        cached = min(prompt_tokens, cached_tokens)
        usd = ((prompt_tokens - cached) * c["input_usd_per_million"]
               + cached * c["cached_input_usd_per_million"]
               + completion_tokens * c["output_usd_per_million"]) / 1_000_000
        return usd * c["usd_to_cny_budget_factor"]

    def call(self, client, api_provider, model, prompt, role, call_id, max_tokens=4096,
             log_dir=None, use_json_mode=False, **unused):
        if self.calls >= self.config["max_calls"]:
            raise ExperimentStopped("Pilot logical-call limit reached")
        self.calls += 1
        # cl100k is not DeepSeek's tokenizer: reserve 2x local input estimate, then reconcile API usage.
        reserve = self.cost(2 * len(self.encoder.encode(prompt)), max_tokens)
        for attempt in range(1, self.config["max_attempts"] + 1):
            record = self._reserve_record(reserve, role, call_id, attempt)
            started = time.monotonic()
            try:
                params = {
                    "model": model, "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config["temperature"], "max_tokens": max_tokens,
                    "extra_body": {"thinking": {"type": self.config["thinking"]}},
                }
                if use_json_mode:
                    params["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**params)
            except Exception as exc:
                self._failure(record, exc, started)
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))
                transient = transient or status == 429 or (status is not None and status >= 500)
                if not transient or attempt == self.config["max_attempts"]:
                    raise ExperimentStopped(f"Request stopped: {type(exc).__name__}, status={status}") from None
                time.sleep(2 ** attempt)
                continue
            return self._success(record, response, prompt, model, started, log_dir)
        raise AssertionError("Unreachable retry state")

    def _reserve_record(self, reserve, role, call_id, attempt):
        accounted = 0.0 if self.mock else reserve
        self.ledger.reserve(accounted)
        record = {
            "request_id": uuid.uuid4().hex, "run_id": self.run_id, "role": role, "call_id": call_id,
            "attempt": attempt, "state": "pending", "accounted_cny": accounted,
            "mock": self.mock, "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.ledger.append(record)
        return record

    def _failure(self, record, exc, started):
        # Unknown billing after timeout remains reserved, instead of treating a failed request as free.
        record.update(state="failed_billing_unknown", error_type=type(exc).__name__,
                      status_code=getattr(exc, "status_code", None), elapsed_seconds=time.monotonic() - started)
        self.ledger.append(record)

    def _success(self, record, response, prompt, model, started, log_dir):
        usage = response.usage
        if usage is None:
            raise ExperimentStopped("Missing API usage; pending budget reservation retained")
        cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = cached or (getattr(details, "cached_tokens", 0) if details else 0) or 0
        cost = self.cost(usage.prompt_tokens, usage.completion_tokens, cached)
        record.update(state="completed", accounted_cny=0.0 if self.mock else cost,
                      prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                      cached_tokens=cached, estimated_cny_at_peak=cost,
                      response_model=response.model, elapsed_seconds=time.monotonic() - started)
        content = response.choices[0].message.content if response.choices else None
        finish_reason = response.choices[0].finish_reason if response.choices else None
        record.update(response_finish_reason=finish_reason, response_valid=bool(content) and finish_reason != "length")
        self.ledger.append(record)
        info = {
            **record, "model": model, "prompt": prompt, "response": content,
            "prompt_num_tokens": usage.prompt_tokens, "response_num_tokens": usage.completion_tokens,
            "total_time": record["elapsed_seconds"], "call_time": record["elapsed_seconds"],
        }
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            write_json(Path(log_dir) / f"{record['request_id']}.json", info)
        allow_partial = finish_reason == "length" and self.config.get("allow_truncated_nonempty", False)
        if not content or (not record["response_valid"] and not allow_partial):
            raise ExperimentStopped("Empty or truncated response; raw output and usage recorded; pilot stopped")
        if allow_partial:
            print(f"[TRUNCATED] {record['role']}/{record['call_id']}: upstream handling; finish_reason=length")
        return content, info


@contextmanager
def installed_transport(transport):
    """Patch only runtime bindings; the upstream source, prompts, parser and scoring stay intact."""
    bindings = []
    for name in ("generator", "reflector", "curator", "bulletpoint_analyzer"):
        module = importlib.import_module(f"ace.core.{name}")
        if hasattr(module, "timed_llm_call"):
            bindings.append((module, "timed_llm_call", module.timed_llm_call))
            module.timed_llm_call = transport.call
    module = importlib.import_module("ace.ace")
    bindings.append((module, "initialize_clients", module.initialize_clients))
    module.initialize_clients = lambda provider: (transport.client,) * 3
    try:
        yield
    finally:
        for module, name, original in reversed(bindings):
            setattr(module, name, original)
