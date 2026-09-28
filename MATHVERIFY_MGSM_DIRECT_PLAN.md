# MGSM direct → Math-Verify implementation plan

## Phase 1 — Add the shared scorer

- Add a `process_results` helper to `lm_eval/tasks/mgsm/utils.py`.
- Read the numeric gold answer from `doc["answer_number"]`.
- Preserve the current flexible numeric extraction for the legacy score.
- Compute a second score with `math_verify.parse` and `math_verify.verify`.
- Add focused unit tests for numeric, formatted, equivalent, and malformed answers.

## Phase 2 — Wire the scorer into the shared direct config

- Add `process_results: !function ../utils.process_results` to `direct_yaml`.
- Add `math_verify` to `metric_list` while retaining `exact_match` for comparison.
- Confirm YAML function resolution works from the `direct/` directory.

## Phase 3 — Regenerate all language configs

- Run the existing generator in direct mode with `--overwrite`.
- Confirm all ten language configs still include `direct_yaml` and retain their
  language-specific prompt/target fields.

## Phase 4 — Validate and compare

- Validate the `mgsm_direct` tag/group.
- Run a small smoke evaluation.
- Compare legacy `exact_match` and `math_verify` outputs on logged samples.
- Check that the aggregate group reports both metrics.
