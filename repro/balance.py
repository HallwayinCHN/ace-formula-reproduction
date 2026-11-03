"""Read-only DeepSeek balance check; never expose credentials or make model calls."""

from datetime import datetime, timezone
import json
import os

from dotenv import load_dotenv
import httpx

from repro import ROOT
from repro.pilot import load_config
from repro.replay import atomic_json


def get_balance():
    config = load_config()
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get(config["key_env"], "").strip()
    if not key:
        raise RuntimeError("Local key is not configured")
    response = httpx.get(config["base_url"].rstrip("/") + "/user/balance",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Balance check status={response.status_code}")
    data = response.json()
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "is_available": data.get("is_available"),
              "balances": [{"currency": row.get("currency"), "total_balance": row.get("total_balance")}
                           for row in data.get("balance_infos", [])]}
    atomic_json(ROOT / "results/repro/account-status.json", result)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(get_balance(), indent=2))
    except Exception as exc:
        print(f"Balance check failed: {type(exc).__name__}")
        raise SystemExit(2)
