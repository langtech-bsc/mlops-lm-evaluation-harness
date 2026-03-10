### IberoBench, instructed

IberoBench task variants that are adapted for instruction-tuned and/or aligned models. The changes mostly consist in different stop criteria (`generation_kwargs.until`) and generation limit (`generation_kwargs.max_gen_toks`) to avoid issues where generation would be cut in the middle due to the model's usage of `\n\n` between paragraphs, and incomplete generations due to limits that were too short for aligned models that provide CoT.

All the tasks retain the same name but suffixed by `_inst` to reflect the variation in `generation_kwargs`. These variants should only be used to evaluate instruction-tuned/aligned models and may not even be appropriate for all instruction-tuned models; look at generated samples to ensure that these settings allow the model to generate complete answers (if it is able to do so).

* **`mgsm_native_cot_eu_inst`**, **`mgsm_direct_{ca,gl,en,es_spanish_bench}_inst`**: Longer `max_gen_toks` (8191); no generation stop criteria. Remove all criteria to stop on *`Question:`* strings, `</s>` or `<|im_end|>`.
* **`cabreu_inst`** tag, with tasks **`cabreu_{abstractive, extractive, extreme}_inst`**: Longer `max_gen_toks` (8191); no generation stop criteria (modified in a new `_cabreu_inst_common_yaml`).
* **`summarization_gl_inst`**, **`xlsum_es_inst`**: Longer `max_gen_toks` (1024); no generation stop criteria.
* **`xsum_inst`**: New implementation of `xsum` as a regular summarization task, without using `unitxt`. Different prompt that asks for a *"One-sentence summary"*; use the same processing function as `summarization_gl` (`process_summarization` from `galician_bench/utils.py`); longer `max_gen_toks` (1024); no stop criteria.
* **`xquad_{ca,es,en}_inst`**: Stop only on `\n\n` and not on `\n`. (`xquad_{en,es}_inst` import from `_xquad_inst_common_yaml`, while `xquad_ca_inst` is defined on its own.)
