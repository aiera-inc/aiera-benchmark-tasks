# aiera-benchmark-tasks
This repository holds public-facing LLM benchmark tasks for use with EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). A leaderboard for these tasks is available on huggingface [here](https://huggingface.co/spaces/Aiera/aiera-leaderboard).

Tasks included:
* **finqa**: Calculation-based Q&A over financial text. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/finqa-verified).
* **aiera_ect_sum**: Abstractive summarizations of earnings call transcripts. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/aiera-ect-sum).
* **aiera_transcript_sentiment**: Event transcript segments with labels indicating the financial sentiment. Dataset available on [huggingface](https://huggingface.co/datasets/Aiera/aiera-transcript-sentiment).

## The Aiera Score

The [leaderboard](https://huggingface.co/spaces/Aiera/aiera-leaderboard) ranks models by a
weighted **Aiera Score**. Its central measure is **Research** (answering proprietary,
analyst-grade research questions while connected to Aiera's [MCP server](https://mcp-pub.aiera.com)),
combined with the capability tasks in this repo:

| Component | Weight |
| --- | --- |
| Research (model + Aiera MCP) | 60% |
| Q&A (`finqa`) | 24% |
| Summary (`aiera_ect_sum`) | 10% |
| Sentiment (`aiera_transcript_sentiment`) | 6% |

The board is **research-gated**: a model must have a Research score to be listed. The Research
measure is evaluated separately (it is not one of the lm-eval tasks in this repo); the three
capability tasks above are what this repository defines and runs.

## Note

The evaluation criteria was designed to be extremely permissive, accounting for verbosity of chat model output by stripping extraneous data in post processing functions defined in the `utils.py` files of each task. You may find that this is not adequate for your evaluation use case and rewrite to evaluate the model's ability to enforce formatting.

## How to use

Set up the environment with conda:

```bash
conda env create -f environment.yml
conda activate aiera-benchmarking-tasks
```

This installs `lm_eval==0.4.3` (pinned in `environment.yml`) along with the metric
dependencies.

Now you can run individual tasks using the standard `lm_eval` command line:

```bash
lm_eval --model openai-chat-completions \
    --model_args model=gpt-4-turbo-2024-04-09 \
    --tasks aiera_ect_sum,aiera_transcript_sentiment,finqa\
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
    tasks=["aiera_ect_sum","aiera_transcript_sentiment","finqa"],
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

## Publishing models to the leaderboard

The commands above run a single model and print results, but they don't get a model
onto the [leaderboard](https://huggingface.co/spaces/Aiera/aiera-leaderboard).
The board only renders models that have a complete results file in the
[`Aiera/aiera-leaderboard-results`](https://huggingface.co/datasets/Aiera/aiera-leaderboard-results)
dataset (plus a matching entry in the queue dataset). The `runner/` package wraps that
end-to-end: it runs all capability tasks for a reviewed set of models, writes results in the
schema the Space expects, and publishes to both the results and queue datasets. (The Research
component of the Aiera Score is scored separately, not by this runner.)

The model list (including the correct, current provider model ids) lives in
`runner/models.py`. **Model ids must be exact**: e.g. Anthropic ids from the 4.6
generation on are *dateless* (`claude-opus-4-6`, not `claude-opus-4-6-20250725`),
otherwise the provider returns a 404 and the run is recorded as `FAILED`.

```bash
# Validate the registry and see the plan (no API calls, no cost):
python -m runner.run --models all --dry-run

# Smoke test one model against 2 samples/task without publishing:
python -m runner.run --models openai/gpt-5.5 --limit 2 --no-upload

# Full run + publish (needs the provider keys and an HF write token):
export ANTHROPIC_API_KEY=...        # per-provider keys, see runner/models.py `requires`
export OPENAI_API_KEY=...
export HF_TOKEN=...                  # write access to the Aiera datasets
python -m runner.run --models anthropic/claude-opus-4-8,anthropic/claude-opus-4-7,openai/gpt-5.5
```

Models missing their required key are recorded as `FAILED` (with the reason) and
skipped; a model that errors mid-run is recorded as `FAILED` without aborting the rest
of the batch. To add a new model, append a `ModelSpec` to `runner/models.py` after
verifying its id against the provider's `/v1/models` endpoint.