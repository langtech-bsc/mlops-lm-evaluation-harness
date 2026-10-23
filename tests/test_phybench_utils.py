from lm_eval.tasks.arxivmath_2026.utils import (
    _normalize_reference,
    process_math_results,
)


def test_normalize_reference_display_equation():
    assert _normalize_reference(r"$$\gamma = \frac{2}{5}$$") == r"\frac{2}{5}"
    assert _normalize_reference(r"\[v = x + y\]") == "x + y"


def test_process_math_results_accepts_equivalent_boxed_rhs():
    doc = {"answer": r"$$\gamma = \frac{2}{5}$$"}
    response = r"The contrast is 0.4, hence \boxed{\frac{2}{5}}"

    scores = process_math_results(doc, [response])

    assert scores["exact_match"] == 1
    assert scores["math_verify"] == 1
