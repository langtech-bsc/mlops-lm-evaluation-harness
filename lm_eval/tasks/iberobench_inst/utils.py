import re

import evaluate

from lm_eval.tasks.catalan_bench.utils import process_doc_cabreu
from lm_eval.tasks.xquad.utils import process_results_qa as process_results_xquad
from lm_eval.tasks.catalan_bench.utils import process_results_qa as process_results_xquad_ca
from lm_eval.tasks.galician_bench.utils import process_summarization
from lm_eval.tasks.spanish_bench.utils import process_xlsum

def process_xsum(dataset):
    def _process_doc(doc):
        # Remove double spaces
        doc["document"] = re.sub(r" +", " ", doc["document"])
        doc["summary"] = re.sub(r" +", " ", doc["summary"])
        return doc

    return dataset.map(lambda doc: _process_doc(doc))

def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items

def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]