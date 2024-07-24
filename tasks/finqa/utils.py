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
