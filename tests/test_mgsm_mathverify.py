from lm_eval.tasks.mgsm.utils import process_results


def test_mgsm_mathverify_accepts_equivalent_numeric_forms():
    result = process_results({"answer_number": 6}, ["6.0"])

    assert result["exact_match"] == 0
    assert result["math_verify"] == 1


def test_mgsm_mathverify_preserves_legacy_numeric_extraction():
    result = process_results({"answer_number": 1234}, ["$1,234"])

    assert result["exact_match"] == 1
    assert result["math_verify"] == 1


def test_mgsm_mathverify_rejects_wrong_answer():
    result = process_results({"answer_number": 6}, ["7"])

    assert result == {"exact_match": 0, "math_verify": 0}
