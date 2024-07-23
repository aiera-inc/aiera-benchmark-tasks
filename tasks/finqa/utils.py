import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring


def preprocess_docs(dataset):
    def _process_doc(doc):
        # breakpoint()

        query_list = doc["query"].split(".")
        # insert additional instruction. Otherwise we get verbose answer from gpt-4-turbo etc.
        query_list.insert(1, " Respond with only the numeric answer in the appropriate units.")
        query = ".".join(query_list)

        out_doc = {
            "id": doc["id"],
            "query": query,
            "answer": doc["answer"],
        }
        return out_doc

    return dataset.map(_process_doc)

def process_results_gen(doc, results):
    completion = results[0]
    target =  str(doc["answer"])

    if "formatted answer:" in completion.lower():
        completion_splits = completion.split(":")
        completion = completion_splits[-1].strip()

    # hack fix for string formatting
    if target[-2] == ".":
        target = target + "0"
    
    elif "." not in target:
        target = target + ".00"

    exact_match_manual = 1 if completion == target else 0

    return {
        "exact_match_manual": exact_match_manual
    }

