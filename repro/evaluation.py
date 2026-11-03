"""Official generation/extraction/scoring, with a fixed denominator and complete per-item results."""

import time

from utils import extract_answer


def evaluate_fixed(data_processor, generator, playbook, test_samples, max_tokens=4096,
                   log_dir=None, max_workers=1, use_json_mode=False):
    results = []
    for index, sample in enumerate(test_samples):
        started = time.monotonic()
        item = {"index": index, "is_correct": False, "target": sample["target"], "error": None}
        try:
            response, _, info = generator.generate(
                question=sample["question"], playbook=playbook, context=sample["context"], reflection="(empty)",
                use_json_mode=use_json_mode, call_id=f"test_eval_{index}", log_dir=log_dir,
            )
            answer = extract_answer(response)
            item.update(final_answer=answer, is_correct=data_processor.answer_is_correct(answer, sample["target"]),
                        prompt_tokens=info["prompt_num_tokens"], output_tokens=info["response_num_tokens"],
                        finish_reason=info.get("response_finish_reason"))
        except Exception as exc:
            item.update(final_answer="No final answer found", error=type(exc).__name__)
        item["elapsed_seconds"] = time.monotonic() - started
        results.append(item)
    correct = sum(item["is_correct"] for item in results)
    summary = {"accuracy": correct / len(results) if results else 0.0, "correct": correct, "total": len(results),
               "no_answer": sum(item["final_answer"] == "No final answer found" for item in results),
               "errors": sum(item["error"] is not None for item in results)}
    return summary, {"individual_results": results, "errors": [r for r in results if not r["is_correct"]]}
