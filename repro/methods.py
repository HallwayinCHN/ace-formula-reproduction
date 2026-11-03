"""Selective reflection and deterministic, label-free playbook retrieval."""

from dataclasses import dataclass
import json
from pathlib import Path

from ace import ACE
from logger import log_bullet_usage
from playbook_utils import parse_playbook_line
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import count_tokens, extract_answer


class SkipCorrect(Exception):
    def __init__(self, answer, bullet_ids):
        self.answer, self.bullet_ids = answer, bullet_ids


class SelectiveACE(ACE):
    def __init__(self, *args, selective_enabled=True, keep_every=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.selective_enabled, self.keep_every = selective_enabled, keep_every
        self.gate_events = []

    def _train_single_sample(self, **kwargs):
        if not self.selective_enabled:
            return super()._train_single_sample(**kwargs)
        original = self.generator.generate

        def gated_generate(*args, **call_kwargs):
            result = original(*args, **call_kwargs)
            if call_kwargs["call_id"] == f"{kwargs['step_id']}_gen_initial":
                answer = extract_answer(result[0])
                correct = kwargs["data_processor"].answer_is_correct(answer, kwargs["task_dict"]["target"])
                skip = correct and kwargs["step"] % self.keep_every != 0
                event = {"step": kwargs["step"], "initial_correct": correct, "skipped": skip}
                self.gate_events.append(event)
                with (Path(kwargs["log_dir"]) / "gate_events.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
                if skip:
                    raise SkipCorrect(answer, result[1])
            return result

        self.generator.generate = gated_generate
        try:
            return super()._train_single_sample(**kwargs)
        except SkipCorrect as event:
            return self._skip_result(kwargs, event)
        finally:
            self.generator.generate = original

    def _skip_result(self, args, event):
        log_bullet_usage(args["usage_log_path"], args["epoch"], args["step"], args["task_dict"],
                        event.bullet_ids, playbook=self.playbook, is_correct=True)
        state = {"final_answer": event.answer, "is_correct": True,
                 "playbook_num_tokens": count_tokens(self.playbook), "playbook_length": len(self.playbook)}
        print(f"[SELECTIVE] step {args['step']}: correct; skipped reflection, curation and duplicate generation")
        return event.answer, event.answer, {"pre_train_result": dict(state), "post_train_result": dict(state)}


@dataclass
class Entry:
    id: str
    heading: str
    text: str
    content: str
    quality: int


def parse_entries(playbook):
    entries, heading = [], "## GENERAL"
    for line in playbook.splitlines():
        if line.strip().startswith("##"):
            heading = line.strip()
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            entries.append(Entry(parsed["id"], heading, line, parsed["content"],
                                 parsed["helpful"] - parsed["harmful"]))
        elif line.strip():
            if not entries or entries[-1].heading != heading:
                raise ValueError("Unstructured playbook content; refusing to silently drop it")
            entries[-1].text += "\n" + line
            entries[-1].content += "\n" + line
    return entries


def render_entries(entries):
    parts, heading = [], None
    for entry in entries:
        if entry.heading != heading:
            parts.append(entry.heading)
            heading = entry.heading
        parts.append(entry.text)
    return "\n".join(parts)


def select_playbook(question, playbook, max_entries=12, max_tokens=1500):
    entries = parse_entries(playbook)
    full_tokens = count_tokens(playbook)
    if full_tokens <= max_tokens and len(entries) <= max_entries:
        return {"text": playbook, "ids": [entry.id for entry in entries],
                "full_tokens": full_tokens, "selected_tokens": full_tokens}
    scores = [0.0] * len(entries)
    if entries:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            matrix = vectorizer.fit_transform([entry.content for entry in entries])
            scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel().tolist()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
    order = sorted(range(len(entries)), key=lambda i: (-scores[i], -entries[i].quality, entries[i].id))
    selected = []
    for index in order:
        if len(selected) == max_entries:
            break
        candidate = selected + [entries[index]]
        if count_tokens(render_entries(candidate)) <= max_tokens:
            selected = candidate
    text = render_entries(selected)
    return {"text": text, "ids": [entry.id for entry in selected],
            "full_tokens": full_tokens, "selected_tokens": count_tokens(text)}
