"""Accuracy-first inference candidates. Original ACE sources and trained books stay frozen."""

import math
import re

from ace.prompts.generator import GENERATOR_PROMPT
from repro.methods import parse_entries, render_entries
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import count_tokens

VARIANTS = ("original", "original_guard", "brief_guard", "core_guard", "core_brief_guard")
REFERENCE = re.compile(r"\b[a-z]{3,}-\d{5}\b")


def brief_template():
    text = GENERATOR_PROMPT.replace(
        "- Show your reasoning step-by-step",
        "- Give a brief calculation check, keeping essential arithmetic and unit conversions")
    text = text.replace("- Be concise but thorough in your analysis",
                        "- Keep the reasoning field within 60 words; do not repeat the question or playbook")
    for prefix in ("your", "Your"):
        phrase = prefix + " chain of thought / reasoning / thinking process, detailed analysis and calculations"
        text = text.replace(phrase, "A concise calculation check of at most 60 words")
    return text


def numeric_answer(answer):
    try:
        return math.isfinite(float(str(answer).replace(",", "")))
    except (TypeError, ValueError, OverflowError):
        return False


def referenced_ids(entry):
    return set(REFERENCE.findall(entry.content)) - {entry.id}


def dependency_closure(ids, entries):
    by_id = {entry.id: entry for entry in entries}
    selected, missing = set(ids), set()
    queue = list(ids)
    while queue:
        current = queue.pop()
        for reference in referenced_ids(by_id[current]):
            if reference not in by_id:
                missing.add(reference)
            elif reference not in selected:
                selected.add(reference)
                queue.append(reference)
    return selected, missing


def core_playbook(question, book, top_k=6, token_budget=4300, minimum_score=.08):
    entries = parse_entries(book)
    core = [entry for entry in entries if "FORMULAS" not in entry.heading.upper()]
    formulas = [entry for entry in entries if entry not in core]
    all_ids = [entry.id for entry in entries]

    def fallback(reason):
        return {"text": book, "ids": all_ids, "mode": "full_fallback", "reason": reason,
                "core_ids": [entry.id for entry in core], "tokens": count_tokens(book)}

    if not formulas:
        return fallback("no_formula_entries")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        matrix = vectorizer.fit_transform([entry.content for entry in formulas])
        scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
    except ValueError:
        return fallback("no_lexical_evidence")
    if max(scores) < minimum_score:
        return fallback("weak_lexical_match")
    ranked = sorted(range(len(formulas)), key=lambda index: (-scores[index], formulas[index].id))[:top_k]
    ids = {entry.id for entry in core} | {formulas[index].id for index in ranked}
    closed, missing = dependency_closure(ids, entries)
    # Core is a stable prefix; all retained rule contents and existing referenced entries are intact.
    selected = core + [entry for entry in entries if entry.id in closed and entry not in core]
    text = render_entries(selected)
    if count_tokens(text) > token_budget:
        return fallback("core_and_dependencies_exceed_budget")
    return {"text": text, "ids": [entry.id for entry in selected], "mode": "core_retrieval", "reason": None,
            "core_ids": [entry.id for entry in core], "tokens": count_tokens(text),
            "source_already_missing_references": sorted(missing)}
