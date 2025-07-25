import argparse
import os
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import mlflow

METRICS_TO_TRACK = [
    "acc",
    "acc_ambig",
    "acc_disambig",
    "bias_score_ambig",
    "bias_score_disambig",
    "bleu",
    "bleu_acc",
    "bleu_diff",
    "bleu_max",
    "em",
    "exact_match_get-answer",
    "exact_match_remove_whitespace",
    "f1",
    "lprob_diff",
    "lprob_max",
    "mcc",
    "mc1",
    "mc2",
    "mc3",
    "rouge1",
]

TASK_SCHEME = {
    # ca
    "belebele_cat_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "ca",
    },
    "xnli_ca": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "ca",
    },
    "catcola": {
        "num_labels": "2 (mcc)",
        "metric": "mcc",
        "category": "Linguistic Acceptability",
        "language": "ca",
    },
    "copa_ca": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "ca",
    },
    "openbookqa_ca": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "ca",
    },
    "parafraseja": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "ca",
    },
    "paws_ca": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "ca",
    },
    "piqa_ca": {"num_labels": "2", "metric": "acc", "category": "QA", "language": "ca"},
    "siqa_ca": {"num_labels": "3", "metric": "acc", "category": "QA", "language": "ca"},
    "teca": {"num_labels": "3", "metric": "acc", "category": "NLI", "language": "ca"},
    "wnli_ca": {
        "num_labels": "2",
        "metric": "acc",
        "category": "NLI",
        "language": "ca",
    },
    "arc_ca_easy": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "ca",
    },
    "arc_ca_challenge": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "ca",
    },
    "xstorycloze_ca": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "ca",
    },
    "xquad_ca": {
        "num_labels": "gen_task",
        "metric": "f1",
        "category": "QA",
        "language": "ca",
    },
    "catalanqa": {
        "num_labels": "gen_task",
        "metric": "f1",
        "category": "QA",
        "language": "ca",
    },
    "coqcat": {
        "num_labels": "gen_task",
        "metric": "f1",
        "category": "QA",
        "language": "ca",
    },
    "flores_ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "ca",
    },
    "flores_de-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_en-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_es-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_eu-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_fr-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_gl-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_it-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_pt-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-de": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-en": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-fr": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-it": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "flores_ca-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "ca",
    },
    "cabreu_extractive": {
        "num_labels": "gen_task",
        "metric": "rouge1",
        "category": "Summarization",
        "language": "ca",
    },
    "cabreu_abstractive": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Summarization",
        "language": "ca",
    },
    "cabreu_extreme": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Summarization",
        "language": "ca",
    },
    "mgsm_direct_ca": {
        "num_labels": "gen_task",
        "metric": "exact match",
        "category": "Math",
        "language": "ca",
    },
    "phrases_va": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "ca",
    },
    "phrases_ca-va": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "ca",
    },
    "phrases_va-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "ca",
    },
    "phrases_es-va": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "es",
    },
    "phrases_va-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "es",
    },
    "veritasqa_gen_ca": {
        "num_labels": "gen_task",
        "metric": "bleu_max,bleu_acc,bleu_diff",
        "category": "Truthfulness",
        "language": "ca",
    },
    "veritasqa_mc_ca": {
        "num_labels": "varying",
        "metric": "lprob_max,lprob_diff,mc1,mc2,mc3",
        "category": "Truthfulness",
        "language": "ca",
    },
    # From alexandrainst/m_mmlu
    "m_mmlu_ca": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "ca",
    },
    # es
    "belebele_spa_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "es",
    },
    "wnli_es": {
        "num_labels": "2",
        "metric": "acc",
        "category": "NLI",
        "language": "es",
    },
    "xnli_es_spanish_bench": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "es",
    },
    "paws_es": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "es",
    },
    "escola": {
        "num_labels": "2 (mcc)",
        "metric": "mcc",
        "category": "Linguistic Acceptability",
        "language": "es",
    },
    "xstorycloze_es": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "es",
    },
    "mgsm_direct_es_v2": {
        "num_labels": "gen_task",
        "metric": "exact match",
        "category": "Math",
        "language": "es",
    },
    "xquad_es": {
        "num_labels": "gen_task",
        "metric": "f1",
        "category": "QA",
        "language": "es",
    },
    "xlsum_es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Summarization",
        "language": "es",
    },
    "flores_es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "es",
    },
    "flores_de-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_en-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_ca-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_eu-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_fr-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_gl-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_it-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_pt-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-de": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-en": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-fr": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-it": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "flores_es-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "es",
    },
    "esbbq": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_age": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_disability_status": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_gender": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_lgbtqia": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_nationality": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_physical_appearance": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_race_ethnicity": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_religion": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_ses": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "esbbq_spanish_region": {
        "num_labels": "3",
        "metric": "acc_ambig,acc_disambig,bias_score_ambig,bias_score_disambig",
        "category": "Social Bias",
        "language": "es",
    },
    "veritasqa_gen_es": {
        "num_labels": "gen_task",
        "metric": "bleu_max,bleu_acc,bleu_diff",
        "category": "Truthfulness",
        "language": "es",
    },
    "veritasqa_mc_es": {
        "num_labels": "varying",
        "metric": "lprob_max,lprob_diff,mc1,mc2,mc3",
        "category": "Truthfulness",
        "language": "es",
    },
    "cocoteros_es": {
        "num_labels": "gen_task",
        "metric": "bleu,rouge1",
        "category": "Constrained Text Generation",
        "language": "es",
    },
    # From alexandrainst/m_mmlu
    "m_mmlu_es": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "es",
    },
    # From openai/MMMLU
    "mmmlu_es": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "es",
    },
    # eu
    "belebele_eus_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "eu",
    },
    "xstorycloze_eu": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "eu",
    },
    "eus_reading": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "eu",
    },
    "eus_proficiency": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "eu",
    },
    "eus_trivia": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "eu",
    },
    "eus_exams_eu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "eu",
    },
    "eus_exams_eu_ejadministrari": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_ejlaguntza": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_ejlaguntzaile": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_ejteknikari": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opebilbaoeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehuadmineu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehuauxeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehubiblioeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehuderechoeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehueconomicaseu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehuempresarialeseu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehusubalternoeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehutecnicoeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeehuteknikarib": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opegasteizkoudala": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakiadmineu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakiauxenfeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakiauxeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakiceladoreu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakienfeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakioperarioeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakitecnicoeu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_opeosakivarioseu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza1e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza2e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza3e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza5e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza6e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "eus_exams_eu_osakidetza7e": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA (subtask)",
        "language": "eu",
    },
    "qnlieu": {"num_labels": "2", "metric": "acc", "category": "NLI", "language": "eu"},
    "xnli_eu": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "eu",
    },
    "xnli_eu_native": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "eu",
    },
    "wnli_eu": {
        "num_labels": "2",
        "metric": "acc",
        "category": "NLI",
        "language": "eu",
    },
    "xcopa_eu": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "eu",
    },
    "mgsm_direct_eu": {
        "num_labels": "gen_task",
        "metric": "exact match",
        "category": "Math",
        "language": "eu",
    },
    "mgsm_native_cot_eu": {
        "num_labels": "gen_task",
        "metric": "exact_match,get-answer",
        "category": "Math (subtask)",
        "language": "eu",
    },
    "flores_eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "eu",
    },
    "flores_de-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_en-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_ca-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_es-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_fr-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_gl-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_it-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_pt-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-de": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-en": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-fr": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-it": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    "flores_eu-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "eu",
    },
    # From alexandrainst/m_mmlu
    "m_mmlu_eu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "eu",
    },
    # gl
    "belebele_glg_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "gl",
    },
    "galcola": {
        "num_labels": "2 (mcc)",
        "metric": "mcc",
        "category": "Linguistic Acceptability",
        "language": "gl",
    },
    "parafrases_gl": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "gl",
    },
    "paws_gl": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "gl",
    },
    "summarization_gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Summarization",
        "language": "gl",
    },
    "mgsm_direct_gl": {
        "num_labels": "gen_task",
        "metric": "exact match",
        "category": "Math",
        "language": "gl",
    },
    "openbookqa_gl": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "gl",
    },
    "truthfulqa_gl_gen": {
        "num_labels": "gen_task",
        "metric": "bleu_max",
        "category": "Truthfulness",
        "language": "gl",
    },
    "truthfulqa_gl_mc1": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Truthfulness",
        "language": "gl",
    },
    "truthfulqa_gl_mc2": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Truthfulness",
        "language": "gl",
    },
    "flores_gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "gl",
    },
    "flores_de-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_en-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_ca-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_es-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_fr-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_eu-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_it-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_pt-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-de": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-en": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-fr": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-it": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "flores_gl-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "gl",
    },
    "veritasqa_gen_gl": {
        "num_labels": "gen_task",
        "metric": "bleu_max,bleu_acc,bleu_diff",
        "category": "Truthfulness",
        "language": "gl",
    },
    "veritasqa_mc_gl": {
        "num_labels": "varying",
        "metric": "lprob_max,lprob_diff,mc1,mc2,mc3",
        "category": "Truthfulness",
        "language": "gl",
    },
    # en
    "belebele_eng_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "en",
    },
    "arc_easy": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "en",
    },
    "arc_challenge": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "en",
    },
    "hellaswag": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "en",
    },
    "copa": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "en",
    },
    "xstorycloze_en": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Commonsense Reasoning",
        "language": "en",
    },
    "xnli_en_iberobench": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "en",
    },
    "openbookqa": {
        "num_labels": "4",
        "metric": "acc",
        "category": "QA",
        "language": "en",
    },
    "piqa": {"num_labels": "2", "metric": "acc", "category": "QA", "language": "en"},
    "social_iqa": {
        "num_labels": "3",
        "metric": "acc",
        "category": "QA",
        "language": "en",
    },
    "cola": {
        "num_labels": "2 (mcc)",
        "metric": "mcc",
        "category": "Linguistic Acceptability",
        "language": "en",
    },
    "wnli": {"num_labels": "2", "metric": "acc", "category": "NLI", "language": "en"},
    "truthfulqa_gen": {
        "num_labels": "gen_task",
        "metric": "bleu_max",
        "category": "Truthfulness",
        "language": "en",
    },
    "truthfulqa_mc1": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Truthfulness",
        "language": "en",
    },
    "truthfulqa_mc2": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Truthfulness",
        "language": "en",
    },
    "paws_en": {
        "num_labels": "2",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "en",
    },
    "xquad_en": {
        "num_labels": "gen_task",
        "metric": "f1",
        "category": "QA",
        "language": "en",
    },
    "mgsm_direct_en": {
        "num_labels": "gen_task",
        "metric": "exact match",
        "category": "Math",
        "language": "en",
    },
    "veritasqa_gen_en": {
        "num_labels": "gen_task",
        "metric": "bleu_max,bleu_acc,bleu_diff",
        "category": "Truthfulness",
        "language": "en",
    },
    "veritasqa_mc_en": {
        "num_labels": "varying",
        "metric": "lprob_max,lprob_diff,mc1,mc2,mc3",
        "category": "Truthfulness",
        "language": "en",
    },
    # From alexandrainst/m_mmlu
    "m_mmlu_en": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "en",
    },
    # From cais/mmlu
    "mmlu": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "en",
    },
    # pt
    "assin_entailment": {
        "num_labels": "3",
        "metric": "acc",
        "category": "NLI",
        "language": "pt",
    },
    "assin_paraphrase": {
        "num_labels": "3",
        "metric": "acc",
        "category": "Paraphrasing",
        "language": "pt",
    },
    "flores_pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation - Adaptation",
        "language": "pt",
    },
    "flores_de-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_en-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_ca-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_es-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_fr-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_eu-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_it-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_gl-pt": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-ca": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-de": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-en": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-fr": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-eu": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-it": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-gl": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "flores_pt-es": {
        "num_labels": "gen_task",
        "metric": "bleu",
        "category": "Translation (subtask)",
        "language": "pt",
    },
    "belebele_por_Latn": {
        "num_labels": "4",
        "metric": "acc",
        "category": "Reading Comprehension",
        "language": "pt",
    },
    # From alexandrainst/m_mmlu
    "m_mmlu_pt": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "pt",
    },
    # From openai/MMMLU
    "mmmlu_pt": {
        "num_labels": "4",
        "metric": "acc",
        "category": "General Knowledge",
        "language": "pt",
    },
}

RANDOM_RESULTS = {
    "veritasqa_mc_ca": 15.0,
    "veritasqa_mc_en": 15.0,
    "veritasqa_mc_gl": 15.0,
    "veritasqa_mc_es": 15.0,
}


def get_random(task, num_labels) -> Optional[float]:
    if task in RANDOM_RESULTS.keys():
        return RANDOM_RESULTS[task]
    if num_labels == "2":
        return 50.00
    elif num_labels == "3":
        return 33.33
    elif num_labels == "4":
        return 25.00
    elif num_labels == "12":
        return 8.33
    elif num_labels in ["gen_task", "2 (mcc)"]:
        return 0.0
    elif num_labels == "varying":
        return np.nan


def get_max(metric_name: str) -> int:
    if metric_name in ["exact_match", "exact_match_get-answer", "mcc"]:
        max = 1
    else:
        max = 100
    return max


def get_model_name_from_model_args(model_args: str) -> str:
    model_arg_parts = model_args.split(",")

    for model_arg in model_arg_parts:
        if model_arg.startswith("pretrained") or model_arg.startswith("path"):
            model_name_parts = model_arg.split("=")
            model_name = model_name_parts[1]
            model_name = shorten_model_name(model_name)
            return model_name

    return "UNKNOWN"


def shorten_model_name(model_name: str) -> str:
    model_name_parts = model_name.split("/")
    return model_name_parts[-1]


def convert_metric_name(metric_name: str) -> str:
    if metric_name.endswith(",none"):
        metric_name = metric_name.replace(",none", "")
    if "," in metric_name:
        metric_name = metric_name.replace(",", "_")

    return metric_name


def get_task_parameters(task: str) -> Tuple[str, str, str, str, float]:
    num_labels = TASK_SCHEME[task]["num_labels"]
    lang = TASK_SCHEME[task]["language"]
    category = TASK_SCHEME[task]["category"]
    metric = TASK_SCHEME[task]["metric"]
    random = get_random(task, num_labels)
    return num_labels, lang, category, metric, random


def normalize_score(score: float, random: float, max: int) -> float:
    return ((score - random) / (max - random)) * 100


def log_to_mlflow(
    results: Union[Dict[str, Any], None],
    args: Union[argparse.Namespace, None] = None,
    model_name_str: Union[str, None] = None,
    experiment_name: Union[str, None] = None,
) -> None:
    mlflow.set_tracking_uri("/gpfs/projects/bsc88/evaluation/mlflow")
    if model_name_str:
        model_name = model_name_str
    else:
        model_name = get_model_name_from_model_args(args.model_args)

    if experiment_name:
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(model_name)

    task_results = results.get("results")
    for task in task_results.keys():
        if task in TASK_SCHEME.keys():
            with mlflow.start_run(run_name=task):
                num_labels, lang, category, metric, random = get_task_parameters(task)
                mlflow.log_param("num_labels", num_labels)
                mlflow.log_param("language", lang)
                mlflow.log_param("category", category)
                mlflow.log_param("metric", metric)
                mlflow.log_param("random", random)
                mlflow.log_param("dataset", task)
                for metric_name, score in task_results[task].items():
                    if score != "N/A":
                        metric_name = convert_metric_name(metric_name)
                        if (
                            task in ["catcola", "escola", "galcola"]
                            and metric_name == "acc"
                        ):
                            continue
                        if metric_name in METRICS_TO_TRACK:
                            if metric_name in ["f1", "acc", "rouge1"]:
                                score = 100 * score
                            score = float(score)
                            score = round(score, 2)
                            mlflow.log_metric(str(metric_name), score)

                            if (
                                task.startswith("veritasqa")
                                or task.startswith("esbbq")
                                or task.startswith("cocoteros")
                            ):
                                mlflow.log_metric("score", np.nan)
                                mlflow.log_param("max", np.nan)
                                mlflow.log_metric(f"normalized_{metric_name}", np.nan)
                                mlflow.log_metric(f"normalized_score", np.nan)
                            else:
                                mlflow.log_metric("score", score)
                                max = get_max(metric_name)
                                mlflow.log_param("max", max)
                                normalized_score = normalize_score(score, random, max)
                                mlflow.log_metric(
                                    f"normalized_{metric_name}", normalized_score
                                )
                                mlflow.log_metric(f"normalized_score", normalized_score)

                            mlflow.log_param("model", model_name)
