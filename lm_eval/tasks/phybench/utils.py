from __future__ import annotations

from lm_eval.tasks.arxivmath_2026.utils import process_math_results


def process_docs(dataset):
    answered = dataset.filter(
        lambda doc: isinstance(doc.get("answer"), str) and bool(doc["answer"].strip())
    )
    seen = set()
    unique_indices = []
    for index, identifier in enumerate(answered["id"]):
        if identifier not in seen:
            seen.add(identifier)
            unique_indices.append(index)
    return answered.select(unique_indices)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    return process_math_results(doc, results)
