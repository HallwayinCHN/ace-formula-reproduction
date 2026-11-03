"""Audit upstream data without displaying held-out text or labels."""

import hashlib
import json
import platform
import random
import re
import subprocess
import unicodedata
from pathlib import Path

from repro import ROOT

PIN = "82709de050e1db6e6ef2f07bcb0393560b94992a"
FILES = {
    "train": "formula_train_subset_500.jsonl",
    "validation": "formula_val_subset_300.jsonl",
    "test": "formula_test.jsonl",
}
ARTIFACTS = ROOT / "repro/artifacts"
SOURCES = (
    "ace/ace.py", "ace/core/generator.py", "ace/core/reflector.py", "ace/core/curator.py",
    "ace/prompts/generator.py", "ace/prompts/reflector.py", "ace/prompts/curator.py",
    "eval/finance/data_processor.py", "utils.py", "playbook_utils.py", "llm.py", "uv.lock",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_rows(split):
    path = ROOT / "eval/finance/data" / FILES[split]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_key(row):
    text = row["context"].split("Question:", 1)[-1]
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_upstream():
    subprocess.run(["git", "merge-base", "--is-ancestor", PIN, "HEAD"], cwd=ROOT, check=True)
    hashes = {}
    for name in SOURCES:
        expected = subprocess.check_output(["git", "show", f"{PIN}:{name}"], cwd=ROOT)
        if (ROOT / name).read_bytes() != expected:
            raise RuntimeError(f"Upstream baseline source changed: {name}")
        hashes[name] = digest(expected)
    return hashes


def prepare():
    source_hashes = verify_upstream()
    rows = {split: load_rows(split) for split in FILES}
    keys = {split: [question_key(row) for row in values] for split, values in rows.items()}
    audit = {"upstream_commit": PIN, "python": platform.python_version(),
             "upstream_source_hashes": source_hashes, "splits": {}}
    for split, values in rows.items():
        raw = (ROOT / "eval/finance/data" / FILES[split]).read_bytes()
        invalid = sum(not row.get("context") or not str(row.get("target", "")).strip() for row in values)
        if invalid or len(keys[split]) != len(set(keys[split])):
            raise ValueError(f"Invalid or duplicated records in {split}")
        audit["splits"][split] = {
            "path": f"eval/finance/data/{FILES[split]}", "records": len(values),
            "sha256": digest(raw), "missing_fields": invalid, "duplicate_questions": 0,
        }
    overlaps = {
        f"{left}__{right}": len(set(keys[left]) & set(keys[right]))
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    if any(overlaps.values()):
        raise ValueError(f"Cross-split duplicates require review: {overlaps}")
    audit["cross_split_question_overlap"] = overlaps
    audit["test_policy"] = "Structural audit only; no held-out text/labels displayed or sent to any model."
    manifest = {"selection_seed": 42, "upstream_commit": PIN, "split_hashes": {}, "splits": {}}
    for split, values in rows.items():
        order = list(range(len(values)))
        random.Random(42).shuffle(order)
        pilot = order[:10] if split != "test" else []
        # Pilot instances are separate from all formal experiment subsets.
        formal = order[10:] if split != "test" else order
        manifest["splits"][split] = {"pilot_indices": pilot, "formal_order": formal}
        manifest["split_hashes"][split] = audit["splits"][split]["sha256"]
    manifest["formal_tiers"] = [[100, 40, 200], [60, 30, 100], [30, 20, 60]]
    manifest["formal_size_locked"] = False
    manifest["note"] = "Choose size using pilot costs before formal training; never select by test performance."
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACTS / "data_audit.json", audit)
    manifest_path = ARTIFACTS / "sample_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing[key] != manifest[key] for key in ("upstream_commit", "split_hashes", "splits")):
            raise RuntimeError("Existing manifest differs; refusing to overwrite the frozen sample selection")
    else:
        write_json(manifest_path, manifest)
    print(json.dumps(audit, indent=2))


def pilot_data():
    manifest = json.loads((ARTIFACTS / "sample_manifest.json").read_text())
    selected = {}
    for split in ("train", "validation"):
        path = ROOT / "eval/finance/data" / FILES[split]
        if digest(path.read_bytes()) != manifest["split_hashes"][split]:
            raise ValueError(f"Data hash changed: {split}")
        rows = load_rows(split)
        selected[split] = [rows[i] for i in manifest["splits"][split]["pilot_indices"]]
    return selected


if __name__ == "__main__":
    prepare()
