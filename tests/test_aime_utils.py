from lm_eval.tasks.aime.utils import process_results


def test_process_results_extracts_explicit_answer_after_math_work():
    response = (
        "Let $m=252$ and $n=25$. Therefore $$m+n=252+25=277$$\n\n"
        "Answer: 277"
    )

    assert process_results({"answer": "277"}, [response]) == {"exact_match": 1}


def test_process_results_prefers_last_boxed_answer():
    response = r"An intermediate estimate is 100, but the result is \boxed{277}."

    assert process_results({"answer": "277"}, [response]) == {"exact_match": 1}


def test_process_results_uses_final_integer_fallback():
    response = "After simplifying the expression, the requested integer is 277"

    assert process_results({"answer": "277"}, [response]) == {"exact_match": 1}
