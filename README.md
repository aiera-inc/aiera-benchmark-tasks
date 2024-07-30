# aiera-benchmark-tasks
This repository holds public-facing LLM benchmark tasks for use with EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). A leaderboard for these tasks is available on huggingface [here](https://huggingface.co/spaces/Aiera/aiera-finance-leaderboard).

Tasks included:
- **aiera_speaker_assign**: Assignments of speakers to event transcript segments and identification of speaker changes. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/aiera-speaker-assign).
* **aiera_ect_sum**: Abstractive summarizations of earnings call transcripts. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/aiera-ect-sum).
* **finqa**: Calculation-based Q&A over financial text. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/finqa-verified).
* **aiera_transcript_sentiment**: Event transcript segments with labels indicating the financial sentiment. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/aiera-transcript-sentiment).

## Note

The evaluation criteria was designed to be extremely permissive, accounting for verbosity of chat model output by stripping extraneous data in post processing functions defined in the `utils.py` files of each task. You may find that this is not adequate for your evaluation use case and rewrite to evaluate the model's ability to enforce formatting.

## How to use

Set up the environment with conda:

```bash
conda env create -f environment.yml
conda activate aiera-benchmarking-tasks
```

Next set up the `lm-evaluation-harness`.
```bash
git submodule init
pip install -e lm-evaluation-harness
pip install -e lm-evaluation-harness"[api]"
```

Now you can run individual tasks using the standard `lm_eval` command line:

```bash
lm_eval --model openai-chat-completions \
    --model_args model=gpt-4-turbo-2024-04-09 \
    --tasks aiera_ect_sum,aiera_speaker_assign,aiera_transcript_sentiment,finqa\
    --include_path tasks
```

Or programatically with python using:
```python
from lm_eval import tasks, evaluator, simple_evaluate, evaluate
from lm_eval.models.openai_completions import OpenaiChatCompletionsLM

model = OpenaiChatCompletionsLM("gpt-4-turbo-2024-04-09")

task_manager = tasks.TaskManager(include_path="tasks", include_defaults=False)

results = simple_evaluate( # call simple_evaluate
    model=model,
    tasks=["aiera_ect_sum","aiera_speaker_assign","aiera_transcript_sentiment","finqa"],
    num_fewshot=0,
    task_manager=task_manager,
    write_out = True,
)
```

Or alternatively, can run all tasks using

```bash
lm_eval --model openai-chat-completions \
    --model_args model=gpt-4-turbo-2024-04-09 \
    --tasks aiera_benchmark \
    --include_path tasks
```