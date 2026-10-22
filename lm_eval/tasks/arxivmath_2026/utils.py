from __future__ import annotations

import re

from lm_eval.tasks.aime.utils import is_equiv, last_boxed_only_string, remove_boxed
from lm_eval.tasks.local_judge import judge_correctness


def _extract_answer(response: str) -> str:
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        try:
            return remove_boxed(boxed).strip()
        except (AssertionError, IndexError):
            pass

    final = re.findall(
        r"(?:final answer|answer)\s*(?:is|:)?\s*\$?(.+?)(?:\$|\n|$)",
        response,
        flags=re.IGNORECASE,
    )
    return final[-1].strip() if final else response.strip()


def process_math_results(doc: dict, results: list[str]) -> dict[str, int]:
    prediction = results[0]
    extracted = _extract_answer(prediction)
    exact = int(is_equiv(extracted, str(doc["answer"])))

    verified = 0
    try:
        from math_verify import parse, verify

        verified = int(verify(gold=parse(str(doc["answer"])), target=parse(prediction)))
    except (ImportError, ValueError, TypeError):
        verified = exact

    return {"exact_match": exact, "math_verify": verified}


def process_results(doc: dict, results: list[str]) -> dict[str, int | float]:
    scores = process_math_results(doc, results)
    prediction = results[0]

    judged = judge_correctness(
        question=doc["problem"],
        reference=str(doc["answer"]),
        prediction=prediction,
    )
    return {"acc": judged, **scores}
