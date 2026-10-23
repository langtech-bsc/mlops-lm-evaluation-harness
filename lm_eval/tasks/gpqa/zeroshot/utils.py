from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")

    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def extract_generated_answer(response: str) -> str | None:
    """Extract the final multiple-choice letter from a generated response."""
    tagged = re.findall(
        r"<answer>\s*\(?\s*([A-D])\s*\)?\s*</answer>",
        response,
        flags=re.IGNORECASE,
    )
    if tagged:
        return tagged[-1].upper()

    explicit = re.findall(
        r"(?:resposta|answer)(?:\s+final)?\s*(?:és|is)?\s*[:=]?\s*\(?([A-D])\)?",
        response,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()

    # Last-resort support for a response consisting only of a choice letter.
    standalone = re.fullmatch(r"\s*\(?([A-D])\)?[.\s]*", response, re.IGNORECASE)
    return standalone.group(1).upper() if standalone else None


def process_generated_results(doc: dict, results: list[str]) -> dict[str, int]:
    prediction = extract_generated_answer(results[0])
    target = re.sub(r"[^A-D]", "", str(doc["answer"]).upper())
    return {"exact_match": int(prediction == target)}
