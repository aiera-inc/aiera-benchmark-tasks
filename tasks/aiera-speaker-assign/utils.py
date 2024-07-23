import json 


def process_results_gen(doc, results):
    completion = results[0].strip()
    speaker_name =  doc["speaker"]
    change = doc["change"]

    # json formatting varies across models, some will be properly formatted
    # the below strips extra text
    if "```json\n" in completion:
        completion_splits = completion.split("```json\n")
        completion = "".join(completion_splits[1:])
        completion_splits = completion.split("```")
        completion = completion_splits[0]

    if "\n" in completion:
        if "{" in completion:
            completion_splits = completion.split("{")
            completion = "".join(completion_splits[1:])
            completion = "{" + completion
            if "}" in completion:
                completion_splits = completion.split("}")
                completion = "".join(completion_splits[:-1])
                completion = completion + "}"

    completion = completion.strip()
    # try to load completion
    speakers_result = {}
    try:
        speakers_result = json.loads(completion)
    except:
        return {
            "accuracy": 0
        }

    correct = 0
    # if keys aren't present
    if "change" not in speakers_result or not "speaker" in speakers_result:
        return {
            "accuracy": 0
        }

    change_res = int(speakers_result["change"])
    speaker_res = speakers_result["speaker"]

    if change_res == change and speaker_name == speaker_res:
        correct = 1

    return {
        "accuracy": correct
    }
