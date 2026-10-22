# Multi-Wiki-QA

Extractive reading-comprehension tasks for the cached Catalan, Spanish, and
English configurations of `alexandrainst/multi-wiki-qa`. The dataset exposes a
single split named `train`; it is evaluation-only and is therefore used as the
test split.

The primary metric is correctness from the configured local judge. Standard
SQuAD normalized exact match and token F1 are retained as deterministic
diagnostics, taking the best score over all supplied reference answers.

See `lm_eval/tasks/local_judge.py` for the required endpoint environment
variables.
