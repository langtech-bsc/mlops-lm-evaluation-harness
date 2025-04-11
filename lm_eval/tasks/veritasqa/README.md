# VeritasQA

### Paper

**Title:** `VeritasQA: A Truthfulness Benchmark Aimed at Multilingual Transferability`

**URL:** [`https://aclanthology.org/2025.coling-main.366/`](https://aclanthology.org/2025.coling-main.366/)

**Abstract:** *As Large Language Models (LLMs) become available in a wider range of domains and applications, evaluating the truthfulness of multilingual LLMs is an issue of increasing relevance. TruthfulQA (Lin et al., 2022) is one of few benchmarks designed to evaluate how models imitate widespread falsehoods. However, it is strongly English-centric and starting to become outdated. We present VeritasQA, a context- and time-independent truthfulness benchmark built with multilingual transferability in mind, and available in Spanish, Catalan, Galician and English. VeritasQA comprises a set of 353 questions and answers inspired by common misconceptions and falsehoods that are not tied to any particular country or recent event. We release VeritasQA under an open license and present the evaluation results of 15 models of various architectures and sizes.*

**GitHub repository:** [`https://github.com/langtech-bsc/veritasQA`](https://github.com/langtech-bsc/veritasQA)


### Citation

```
@inproceedings{aula-blasco-etal-2025-veritasqa,
    title = "{V}eritas{QA}: A Truthfulness Benchmark Aimed at Multilingual Transferability",
    author = "Aula-Blasco, Javier  and  Falc{\~a}o, J{\'u}lia  and  Sotelo, Susana  and  Paniagua, Silvia  and  Gonzalez-Agirre, Aitor  and  Villegas, Marta",
    editor = "Rambow, Owen  and  Wanner, Leo  and  Apidianaki, Marianna  and  Al-Khalifa, Hend  and  Eugenio, Barbara Di  and  Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.366/",
    pages = "5463--5474"
}
```

### Languages

* Catalan (`ca`)
* English (`en`)
* Galician (`gl`)
* Spanish (`es`)

The datasets are fully parallel across languages.

### Groups and Tasks


#### Tasks

* `veritasqa_gen_{es,ca,gl,en}`: Generative task
* `veritasqa_mc_{es,ca,gl,en}`: Multiple-choice task across all correct and incorrect reference answers

#### Groups

For lang in [`es`, `ca`, `gl`, `en`]:

* `veritasqa`: both `gen` and `mc` tasks in all languages
* `veritasqa_{lang}`: both `gen` and `mc` tasks in the given language

There is no default task.

### Metrics

Performance in the generation task is measured by BLEU (`bleu_{acc,max,diff}`), and in the multiple-choice task, by log probabilities (`lprob_{max,diff}`) and by three custom multiple-choice metrics (`{mc{1,2,3}`). Below is the explanation of these metrics, extracted from the [paper](https://aclanthology.org/2025.coling-main.366/):

> **Log probabilities.** We append each correct answer to the prompt and calculate their probabilities, and also calculate the probabilities using the incorrect answers. We report the maximum log probability amongst correct answers (`lprob_max`) and the difference between maximum correct and incorrect answers’ log probabilities (`lprob_diff`). \
> **Multiple-choice.** We also calculate the 3 multiple-choice (MC) metrics proposed by Lin et al. (2022) (also used in Kai et al., 2024): MC1 evaluates whether the model assigns the highest score to the best correct answer; MC2 is the normalized probability mass for all correct answers over all available answers, and MC3 assesses whether each correct answer receives a higher score than the incorrect answers. \
> **Generation.** We pass the prompt alone as input for the model to fill in the answer. (...) These responses are then evaluated against correct and incorrect reference
answers with BLEU scores (Papineni et al., 2002), using the SacreBLEU library (Post, 2018). We report the highest BLEU across all correct answers (`bleu_max`), the difference between BLEU scores for correct and incorrect answers (`bleu_diff`), and the accuracy based on whether the highest BLEU across correct answers is better than the highest BLEU across incorrect answers (`bleu_acc`).


### ⚠️ Warning: VeritasQA is a zero-shot benchmark

VeritasQA is designed as a true zero-shot benchmark. There is only a test split and it should be evaluated without any gradient updates and with no few-shot examples. Therefore, the task implementations for both the generative task and the multiple-choice task follow this design and **they may not work properly in a few-shot setting**.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
