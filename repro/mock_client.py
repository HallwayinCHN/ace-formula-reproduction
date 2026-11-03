"""Local HTTP fixtures. These exercise plumbing, never measure model quality."""

import json

import httpx
from openai import OpenAI


def mock_client():
    seen_failure = False

    def respond(request):
        nonlocal seen_failure
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        prompt = body["messages"][0]["content"]
        if prompt.startswith("You are a master curator"):
            content = {"reasoning": "Synthetic fixture", "operations": [
                {"type": "ADD", "section": "formulas_and_calculations",
                 "content": "MOCK fixture: check synthetic arithmetic."}
            ]}
        elif prompt.startswith("You are an expert analyst"):
            content = {"reasoning": "Synthetic fixture", "key_insight": "Check arithmetic", "bullet_tags": []}
        else:
            question = prompt.split("**Question:**", 1)[1].split("**Context:**", 1)[0]
            answer = "7" if "3 + 4" in question else "3"
            if "SYNTHETIC_FAIL_ONCE" in question and not seen_failure:
                seen_failure, answer = True, "-999"
            content = {"reasoning": "Synthetic fixture", "bullet_ids": [], "final_answer": answer}
        return httpx.Response(200, json={
            "id": "mock-completion", "object": "chat.completion", "created": 0, "model": "MOCK-NOT-A-MODEL",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
        })

    return OpenAI(api_key="mock-not-a-secret", base_url="https://mock.invalid/v1", max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(respond)))


def synthetic_data():
    return {
        "train": [
            {"question": "SYNTHETIC_FAIL_ONCE: What is 3 + 4?", "context": "", "target": "7"},
            {"question": "SYNTHETIC_B: What is 8 - 5?", "context": "", "target": "3"},
        ],
        "validation": [{"question": "SYNTHETIC_VALIDATION: What is 8 - 5?", "context": "", "target": "3"}],
    }
