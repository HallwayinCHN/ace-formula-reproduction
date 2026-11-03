"""FTEC5660 reproduction harness; upstream method and prompts remain unchanged."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / ".runtime-cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE / "tiktoken"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
