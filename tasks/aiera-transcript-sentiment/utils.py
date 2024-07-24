def process_results_gen(doc, results):
    completion = results[0].lower().strip()
    target =  doc["sentiment"]

    if len(completion.split()) > 1:
        if "positive" in completion:
            completion = "positive"

        elif "negative" in completion:
            completion = "negative"

        elif "neutral" in completion:
            completion = "neutral"

    negative_scored_positive=0
    negative_scored_neutral=0
    positive_scored_negative=0
    positive_scored_neutral=0
    neutral_scored_positive=0
    neutral_scored_negative=0
    accuracy = 0

    if target == completion:
        accuracy = 1

    if target == "positive":
        if completion == "neutral":
            positive_scored_neutral = 1

        elif completion == "negative":
            positive_scored_negative = 1

    elif target == "neutral":
        if completion == "positive":
            neutral_scored_positive = 1

        elif completion == "negative":
            neutral_scored_negative = 1

    elif target == "negative":
        if completion == "positive":
            negative_scored_positive = 1

        elif completion == "neutral":
            negative_scored_neutral = 1


    return {
        "accuracy": accuracy,
        "negative_scored_positive": negative_scored_positive,
        "negative_scored_neutral": negative_scored_neutral,
        "positive_scored_negative": positive_scored_negative,
        "positive_scored_neutral": positive_scored_neutral,
        "neutral_scored_positive": neutral_scored_positive,
        "neutral_scored_negative": neutral_scored_negative
    }
