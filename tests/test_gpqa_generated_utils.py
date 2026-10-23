from lm_eval.tasks.gpqa.zeroshot.utils import (
    extract_generated_answer,
    process_generated_results,
)


def test_extract_generated_answer_prefers_last_tag():
    response = "Potser B, però finalment <answer>C</answer>."

    assert extract_generated_answer(response) == "C"


def test_extract_generated_answer_supports_explicit_fallback():
    assert extract_generated_answer("La resposta final és (D).") == "D"


def test_process_generated_results_scores_tagged_answer():
    result = process_generated_results(
        {"answer": "(B)"}, ["Raonament breu.\n<answer>B</answer>"]
    )

    assert result == {"exact_match": 1}


def test_process_generated_results_rejects_missing_answer():
    assert process_generated_results({"answer": "(A)"}, ["No ho sé."]) == {
        "exact_match": 0
    }
