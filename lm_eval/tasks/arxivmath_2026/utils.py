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


def _normalize_reference(reference: str) -> str:
    """Reduce a display equation reference to the expression being predicted."""
    reference = reference.strip()
    if reference.startswith("\\[") and reference.endswith("\\]"):
        reference = reference[2:-2].strip()
    elif reference.startswith("$$") and reference.endswith("$$"):
        reference = reference[2:-2].strip()
    elif reference.startswith("$") and reference.endswith("$"):
        reference = reference[1:-1].strip()

    # PHYBench references often name the requested quantity (for example,
    # ``F_max = ...``), while models correctly return only the right-hand side.
    if "=" in reference:
        reference = reference.split("=", 1)[1].strip()
    return reference


def process_math_results(doc: dict, results: list[str]) -> dict[str, int]:
    prediction = results[0]
    extracted = _extract_answer(prediction)
    reference = _normalize_reference(str(doc["answer"]))
    exact = int(is_equiv(extracted, reference))

    verified = 0
    try:
        from math_verify import parse, verify

        verified = int(verify(gold=parse(reference), target=parse(extracted)))
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
