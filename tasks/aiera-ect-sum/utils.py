import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring
from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score


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
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    rouge_scores = scorer.score(target, completion)
    # Rouge1_scores
    rouge1_score = rouge_scores["rouge1"]

    # ROUGE-2
    rouge2_score = rouge_scores["rouge2"]

    # ROUGE-L
    rougeL_score = rouge_scores["rougeLsum"]

    bert_precision, bert_recall, bert_f1 = bert_score(target, completion)

    # add no instruct
    bert_no_instruct_precision, bert_no_instruct_recall, bert_no_instruct_f1 = bert_no_instruct_score(target, completion)

    return {
        "bleu": bleu_score,
        "rouge1_precision": rouge1_score.precision,
        "rouge1_recall": rouge1_score.recall,
        "rouge1_f1": rouge1_score.fmeasure,
        "rouge2_precision": rouge2_score.precision,
        "rouge2_recall": rouge2_score.recall,
        "rouge2_f1": rouge2_score.fmeasure,
        "rougeLsum_precision": rougeL_score.precision,
        "rougeLsum_recall": rougeL_score.recall,
        "rougeLsum_f1": rougeL_score.fmeasure,
        "bert_precision": bert_precision.mean().item(),
        "bert_recall": bert_recall.mean().item(),
        "bert_f1": bert_f1.mean().item(),
        "bert_no_instruct_precision": bert_no_instruct_precision,
        "bert_no_instruct_recall": bert_no_instruct_recall,
        "bert_no_instruct_f1": bert_no_instruct_f1
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


def bert_score(refs, pred):
    # BERTScore calculation

    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([pred], [refs])
    return P, R, F1

def bert_no_instruct_score(refs, pred):
    # BERTScore calculation

    model_name = "avsolatorio/NoInstruct-small-Embedding-v0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    references = [refs.lower()]
    candidates = [pred.lower()]


    P, R, F1 = custom_bert_scorer([pred], [refs], model, tokenizer)

    return P.item(), R.item(), F1.item()


# Function to compute embeddings (at token level)
def get_token_embeddings(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # We use the last hidden state (token embeddings)
        token_embeddings = outputs.last_hidden_state
    return token_embeddings, inputs.attention_mask

# Custom scorer to calculate precision, recall, and F1 using embeddings
def custom_bert_scorer(candidates, references, model, tokenizer):
    candidate_embeddings, candidate_mask = get_token_embeddings(candidates, model, tokenizer)
    reference_embeddings, reference_mask = get_token_embeddings(references, model, tokenizer)
    
    # Convert embeddings to numpy for cosine similarity
    candidate_embeddings = candidate_embeddings.numpy()
    reference_embeddings = reference_embeddings.numpy()
    
    precision_scores = []
    recall_scores = []

    # Iterate over each candidate-reference pair
    for i in range(len(candidates)):
        # Compute cosine similarity between token embeddings
        sim_matrix = cosine_similarity(candidate_embeddings[i], reference_embeddings[i])

        # Precision: max similarity for each token in the candidate summary
        precision_per_token = np.max(sim_matrix, axis=1)
        precision = precision_per_token.mean()  # Averaging across all candidate tokens
        precision_scores.append(precision)

        # Recall: max similarity for each token in the reference summary
        recall_per_token = np.max(sim_matrix, axis=0)
        recall = recall_per_token.mean()  # Averaging across all reference tokens
        recall_scores.append(recall)

    # Precision and recall averaged over all candidate-reference pairs
    precision = np.mean(precision_scores)
    recall = np.mean(recall_scores)

    # F1 score based on precision and recall
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add epsilon to avoid division by zero

    return precision, recall, f1