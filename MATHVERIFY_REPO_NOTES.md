# Repo notes: updating evaluations to Math-Verify

## What this repository is

This is EleutherAI's `lm-evaluation-harness` (`lm_eval` package), a task-driven
evaluation framework. Built-in evaluations live under `lm_eval/tasks/`. Most
tasks are YAML configurations; task-specific Python helpers live beside the
YAML in `utils.py`. The CLI entry point is `lm-eval` / `lm_eval`, configured in
`pyproject.toml`.

Tasks are discovered automatically by `TaskManager` from `lm_eval/tasks/`.
Additional task directories can be supplied with `--include_path`; included
tasks take precedence over built-ins. The task index recognizes YAML tasks,
groups, tags, and Python tasks.

## Math-Verify is already a dependency

The `math` optional extra in `pyproject.toml` is:

```text
sympy>=1.12
antlr4-python3-runtime==4.11
math_verify[antlr4_11_0]
```

The checked-in `requirements.txt` also contains the corresponding installed
packages (`math-verify==0.9.0`, `antlr4-python3-runtime==4.11.0`, SymPy and
`latex2sympy2_extended`). Install the task dependencies in a fresh environment
with:

```bash
pip install -e ".[math]"
```

or install the aggregate task extras with `pip install -e ".[tasks]"`.

## Existing Math-Verify integration

`lm_eval/tasks/minerva_math/utils.py` is the current reference implementation.
It imports:

```python
from math_verify import parse, verify
```

Its `process_results(doc, results)` keeps the legacy normalized answer check as
`exact_match`, and additionally computes:

```python
verify(gold=parse(doc["solution"]), target=parse(candidates))
```

and returns both result keys:

```python
{"exact_match": retval, "math_verify": mathval}
```

This means Math-Verify is currently computed for Minerva-MATH tasks that inherit
that utility, but it is not a globally registered metric. It is a per-example
result key returned by the task's `process_results` function.

`minerva_math/minerva_math500.yaml` includes the subject-specific YAML (for
example `minerva_math_algebra.yaml`), so the subject configs are the place to
check the inherited `process_docs`, `process_results`, prompt, and metric
configuration before changing the aggregate task.

## How task scoring is wired

There are two relevant scoring styles:

1. YAML-only scoring. A task declares `metric_list`, commonly
   `exact_match`, and may declare a `filter_list` to extract the answer before
   scoring. For example, `gsm8k/gsm8k.yaml` uses regex filters for the
   `#### answer` convention and then exact string matching.

2. Python scoring. A YAML declares something like
   `process_results: !function utils.process_results`. The function receives
   one document and the generated result list, and returns a dictionary whose
   keys must correspond to configured metrics. This is how Minerva-MATH and
   AIME customize answer handling.

The YAML loader resolves `!function` relative to the YAML directory, preferring
the neighboring Python file. YAML `include:` is recursive and merged with
local keys winning, so a subject task can inherit a common scoring function and
override only selected fields. See `lm_eval/tasks/_yaml_loader.py` and
`lm_eval/config/task.py`.

The standard `exact_match` implementation is in `lm_eval/api/metrics.py` and
is string-oriented. A custom result key such as `math_verify` does not need a
new global metric registration when it is emitted by `process_results` and
listed in that task's `metric_list`; however, the task config must explicitly
list it if the result is expected to be aggregated/reported.

## Math-related task families found here

- `lm_eval/tasks/minerva_math/`: already uses Math-Verify in `utils.py`, while
  retaining the older normalized/SymPy comparison as `exact_match`.
- `lm_eval/tasks/hendrycks_math/`: uses `utils.py` with a custom boxed-answer
  extractor and SymPy-based equivalence (`is_equiv`). Its group YAML aggregates
  the subject tasks with `exact_match`.
- `lm_eval/tasks/gsm8k/`: YAML regex extraction and exact matching; CoT variants
  extract phrases such as `The answer is ...`.
- `lm_eval/tasks/gsm_plus/` and `gsm8k_platinum/`: related generated-answer
  tasks that should be audited separately because their answer formats differ.
- `lm_eval/tasks/aime/`: `utils.py` extracts `$...$` or `\\boxed{...}` and uses
  a custom equivalence check; AIME answers are normally short integers, so
  Math-Verify may be unnecessary unless the intended change is broader than
  numeric final-answer scoring.

## Likely change plan for updating evals

For each target family:

1. Decide whether Math-Verify should replace legacy `exact_match` or be added
   as a second diagnostic metric. The Minerva implementation currently does
   the latter, which is useful for comparing score changes.
2. Add or reuse a local `process_results` that passes the correct gold and
   candidate text to `math_verify.parse` and `math_verify.verify`.
3. Add `math_verify` to that task's `metric_list` (and keep `exact_match` if
   comparison is desired). For a group, update `aggregate_metric_list` if the
   group score should aggregate the new metric.
4. Check answer extraction carefully. Passing the entire chain-of-thought to
   Math-Verify can parse differently from passing the final answer; the gold
   side may also need the full solution or only its boxed final expression,
   depending on the dataset.
5. Preserve task-specific generation stopping strings and prompt behavior.
6. Add focused tests with equivalent LaTeX forms, boxed answers, malformed
   output, and cases where legacy string matching and Math-Verify disagree.

## Useful commands

```bash
# List task names
lm-eval ls tasks

# Validate task discovery/configuration
lm-eval validate --tasks minerva_math500,hendrycks_math,gsm8k

# Run a small smoke evaluation
lm-eval run --model dummy --tasks minerva_math500 --limit 2

# Run tests
python -m pytest
```

## Important caveats

- `math_verify` is an optional task dependency, not a base dependency. Any
  environment running affected tasks must install `[math]`.
- `minerva_math/utils.py` raises an import-time error when the optional parser
  dependencies are absent and specifically requires the antlr 4.11 runtime.
- `process_results` returns per-example metric values; the YAML controls which
  values are aggregated and shown. Adding a result key alone is not enough to
  make it part of the reported task metrics.
- The repository snapshot is version `0.4.13.dev0` in `pyproject.toml`, while
  the pinned `requirements.txt` is an environment-style lock/list. Prefer the
  project extra as the source of truth for adding a new dependency.
