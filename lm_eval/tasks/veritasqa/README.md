# VeritasQA

### Paper

Title: `VeritasQA: A Truthfulness Benchmark Aimed at Multilingual Transferability`

URL: [`https://aclanthology.org/2025.coling-main.366/`](https://aclanthology.org/2025.coling-main.366/)

Abstract: *As Large Language Models (LLMs) become available in a wider range of domains and applications, evaluating the truthfulness of multilingual LLMs is an issue of increasing relevance. TruthfulQA (Lin et al., 2022) is one of few benchmarks designed to evaluate how models imitate widespread falsehoods. However, it is strongly English-centric and starting to become outdated. We present VeritasQA, a context- and time-independent truthfulness benchmark built with multilingual transferability in mind, and available in Spanish, Catalan, Galician and English. VeritasQA comprises a set of 353 questions and answers inspired by common misconceptions and falsehoods that are not tied to any particular country or recent event. We release VeritasQA under an open license and present the evaluation results of 15 models of various architectures and sizes.*

Homepage: `https://github.com/sylinrl/TruthfulQA`


### Citation

```
@inproceedings{aula-blasco-etal-2025-veritasqa,
    title = "{V}eritas{QA}: A Truthfulness Benchmark Aimed at Multilingual Transferability",
    author = "Aula-Blasco, Javier  and
      Falc{\~a}o, J{\'u}lia  and
      Sotelo, Susana  and
      Paniagua, Silvia  and
      Gonzalez-Agirre, Aitor  and
      Villegas, Marta",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.366/",
    pages = "5463--5474"
}
```

### Groups and Tasks

#### Groups

* `veritasqa`

#### Tasks

* `veritasqa_gen_{es,ca,gl,en}`: Multiple-choice, single answer
* `truthfulqa_mc2_{es,ca,gl,en}`: Multiple-choice, multiple answers
* `veritasqa_gen_{es,ca,gl,en}`: Answer generation

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

### Changelog
