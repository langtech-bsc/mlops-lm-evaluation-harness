from __future__ import annotations

from functools import partial

import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.tasks.local_judge import judge_correctness


INSTRUCTIONS = {
    "ca": ("Context", "Pregunta", "Resposta"),
    "es": ("Contexto", "Pregunta", "Respuesta"),
    "en": ("Context", "Question", "Answer"),
}


def doc_to_text(doc: dict) -> str:
    language = doc.get("language") or _language_from_id(doc.get("id", ""))
    context, question, answer = INSTRUCTIONS.get(language, INSTRUCTIONS["en"])
    return f"{context}: {doc['context']}\n\n{question}: {doc['question']}\n\n{answer}:"


def _language_from_id(identifier: str) -> str:
    if "://" in identifier:
        host = identifier.split("://", 1)[1].split(".", 1)[0]
        return host
    return "en"


def process_docs(dataset, language: str):
    return dataset.map(lambda doc: {"language": language})


process_ca = partial(process_docs, language="ca")
process_es = partial(process_docs, language="es")
process_en = partial(process_docs, language="en")


def doc_to_target(doc: dict) -> str:
    answers = doc["answers"]["text"]
    return answers[0] if answers else ""


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = results[0].strip().splitlines()[0] if results[0].strip() else ""
    references = doc["answers"]["text"] or [""]
    return {
        "acc": judge_correctness(
            question=doc["question"],
            reference=" | ".join(references),
            prediction=prediction,
        ),
        "exact_match": max(
            squad_metrics.compute_exact(reference, prediction)
            for reference in references
        ),
        "f1": max(
            squad_metrics.compute_f1(reference, prediction) for reference in references
        ),
    }
