# ArxivMath 2026

Tasks for the June, August, and September 2026 MathArena ArxivMath releases.
The Hugging Face datasets expose their evaluation data under a split named
`train`, so that split is intentionally used as the test split here.

The primary metric uses the shared locally served judge configured in
`lm_eval/tasks/local_judge.py`. Conservative normalized exact match and
`math_verify` symbolic equivalence are also reported as deterministic
diagnostics. A local judge other than MathArena's reference judge does not
produce directly comparable leaderboard scores.
