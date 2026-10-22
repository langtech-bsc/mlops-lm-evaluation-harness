from __future__ import annotations

from functools import partial

from lm_eval.tasks.local_judge import judge_correctness


CATEGORIES = {
    "cultural": "Cultural Reasoning & Traditions",
    "math": "Mathematical & Logical Reasoning with Cultural Relevance",
    "wordplay": "Wordplay & Riddles with Cultural Context",
    "linguistic": "Linguistic Reasoning with Cultural Significance",
}


def process_docs(dataset, category: str):
    return dataset.filter(
        lambda doc: doc["language"] == "Spanish"
        and doc["category"] == CATEGORIES[category]
    )


process_cultural = partial(process_docs, category="cultural")
process_math = partial(process_docs, category="math")
process_wordplay = partial(process_docs, category="wordplay")
process_linguistic = partial(process_docs, category="linguistic")


def doc_to_text(doc: dict) -> str:
    return (
        f"{doc['i18n_prompt']}\n\n"
        "Explica tu razonamiento y termina con una respuesta final breve en español."
    )


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    return {
        "acc": judge_correctness(
            question=doc["i18n_prompt"],
            reference=doc["i18n_gtfa"],
            prediction=results[0],
        )
    }
