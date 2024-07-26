import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring
from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import re

def process_results_gen(doc, results):
    completion = results[0]
    target =  doc["summary"]

    # normalize all text
    completion = completion.lower()
    # drop markdown formatting
    completion = completion.replace("*", "")
    # clean up spaces and newlines
    completion = re.sub(r'\s+', ' ', completion).strip()
    completion = re.sub(r'\n+', ' ', completion).strip()

    target = target.lower()
    target = re.sub(r'\s+', ' ', target).strip()
    target = re.sub(r'\n+', ' ', target).strip()


    # BLEU
    bleu_score = bleu([[target]], [completion])

    # ROUGE-N
    rouge_scores = rouge([target], [completion])

    # Rouge1_scores
    rouge1_score = rouge_scores["rouge1"]

    # ROUGE-2
    rouge2_score = rouge_scores["rouge2"]

    # ROUGE-L
    rougeL_score = rouge_scores["rougeLsum"]

    bert_precision, bert_recall, bert_f1 = bert_score(target, completion)

    return {
        "bleu": bleu_score,
        "rouge1": rouge1_score,
        "rouge2": rouge2_score,
        "rougeLsum": rougeL_score,
        "bert_precision": bert_precision.mean().item(),
        "bert_recall": bert_recall.mean().item(),
        "bert_f1": bert_f1.mean().item()
    }


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)

    # Add newlines between sentences to correctly compute `rougeLsum`.
    # treatseach sentence as a separate unit for the calculation of the longest common subsequence.
    # newlines enforces this boundary
    def _prepare_summary(summary):
        summary = summary.replace(". ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure for type in rouge_types}


def bert_score(refs, pred):
    # BERTScore calculation

    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([pred], [refs])
    return P, R, F1
