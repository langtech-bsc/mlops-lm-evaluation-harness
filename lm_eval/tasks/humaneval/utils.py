import importlib
import os
import sys
from pathlib import Path

import evaluate as hf_evaluate


def _load_code_eval():
    if os.getenv("HF_HUB_OFFLINE") == "1":
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        modules_cache = Path(os.getenv("HF_MODULES_CACHE", hf_home / "modules"))
        cached = sorted(
            (modules_cache / "evaluate_modules" / "metrics").glob(
                "code_eval/*/code_eval.py"
            )
        )
        if cached:
            sys.path.insert(0, str(modules_cache))
            module_name = ".".join(
                cached[-1].relative_to(modules_cache).with_suffix("").parts
            )
            return importlib.import_module(module_name).CodeEval()
    return hf_evaluate.load("code_eval")


try:
    compute_ = _load_code_eval()
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs, strict=True)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Join instruct completions to the prompt used by code evaluation.

    Salamandra sometimes emits three spaces before the first line after the
    indented function docstring, while the prompt's docstring uses four.  The
    resulting concatenated program raises ``IndentationError`` before the
    candidate can be evaluated.  Normalize only this known prompt boundary;
    indentation inside the completion is left unchanged.
    """

    def clean_response(response: str, prompt: str) -> str:
        fence = response.find("```")
        completion = response if fence == -1 else response[:fence]
        if (
            prompt.endswith('    """\n')
            and completion.startswith("   ")
            and not completion.startswith("    ")
        ):
            completion = " " + completion
        return completion

    return [
        [doc["prompt"] + clean_response(r, doc["prompt"]) for r in resp]
        for resp, doc in zip(resps, docs, strict=True)
    ]
