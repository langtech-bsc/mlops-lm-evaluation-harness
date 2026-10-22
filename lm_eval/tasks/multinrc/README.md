# MultiNRC Spanish

Spanish-only tasks for the four MultiNRC reasoning categories. Run
`multinrc_es` to evaluate all categories and receive both per-category and
size-weighted aggregate judge accuracy.

MultiNRC requires semantic grading. These tasks call the shared local
OpenAI-compatible judge in `lm_eval/tasks/local_judge.py`. Before evaluation,
serve the judge with the post-training inference engine and set, as needed:

```bash
export LM_EVAL_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export LM_EVAL_JUDGE_MODEL=deepseek-v4-flash
export LM_EVAL_JUDGE_API_KEY=dummy
```
