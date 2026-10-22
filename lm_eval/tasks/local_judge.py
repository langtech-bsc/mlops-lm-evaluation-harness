"""Shared client for binary grading with a locally served OpenAI-compatible model."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


SYSTEM_PROMPT = """You are a strict but fair evaluator.
Decide whether the candidate answer is substantively correct given the question
and reference answer. Accept mathematically equivalent expressions, valid
paraphrases, and harmless formatting differences. Reject incomplete answers,
contradictions, and answers that merely repeat the question.
Return only 1 for correct or 0 for incorrect."""


def judge_correctness(question: str, reference: str, prediction: str) -> float:
    base_url = os.getenv("LM_EVAL_JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.getenv("LM_EVAL_JUDGE_MODEL", "deepseek-v4-flash")
    api_key = os.getenv("LM_EVAL_JUDGE_API_KEY", "dummy")
    timeout = float(os.getenv("LM_EVAL_JUDGE_TIMEOUT", "120"))
    if urllib.parse.urlparse(base_url).scheme not in {"http", "https"}:
        raise ValueError("LM_EVAL_JUDGE_BASE_URL must use http or https")

    user_prompt = f"""Question:
{question.strip()}

Reference answer:
{reference.strip()}

Candidate answer:
{prediction.strip()}

Is the candidate answer correct? Return only 1 or 0."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(  # noqa: S310 - scheme is validated above
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme is validated above
                request, timeout=timeout
            ) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"].strip()
            labels = re.findall(r"\b[01]\b", content)
            if not labels:
                raise ValueError(f"Judge returned no binary label: {content!r}")
            return float(labels[-1])
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(
        f"Local judge failed at {base_url!r} with model {model!r}"
    ) from last_error
