"""Run the Aiera leaderboard benchmark for one or more models and publish results.

This is the orchestration that the public task definitions in ``tasks/`` were always
missing: it takes a reviewed model registry (``runner/models.py``), runs the four
benchmark tasks via lm-evaluation-harness, writes each result in the exact schema the
leaderboard Space expects, and publishes to BOTH datasets the Space reads:

* ``Aiera/aiera-leaderboard-results`` -> ``<org>/<model>/results_<ts>.json``
* ``Aiera/aiera-leaderboard-queue``   -> ``<org>/<model>.json`` (status FINISHED/FAILED)

The queue entry is not optional: ``src/leaderboard/read_evals.py`` in the Space does
``glob("<requests>/<org/model>*.json")[0]`` and crashes the whole board if a results
file has no matching queue file. So every published result gets a queue entry too.

Usage
-----
    # validate everything offline — no API calls, no network, no cost:
    python -m runner.run --models all --dry-run

    # evaluate the latest frontier models (needs the provider keys set):
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
    python -m runner.run --models anthropic/claude-opus-4-8,openai/gpt-5.5

    # smoke test against 2 samples/task before a full (paid) run:
    python -m runner.run --models openai/gpt-5.5 --limit 2 --no-upload

A failed model is recorded as FAILED in the queue (with the error as ``reason``,
mirroring the existing queue convention) and does NOT abort the rest of the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from runner.models import (
    PROVIDER_KEY_FOR_OPENAI_COMPAT,
    REGISTRY,
    ModelSpec,
    by_path,
)

TASKS = ["aiera_ect_sum", "aiera_transcript_sentiment", "finqa"]
RESULTS_REPO = "Aiera/aiera-leaderboard-results"
QUEUE_REPO = "Aiera/aiera-leaderboard-queue"
REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "tasks"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _missing_keys(spec: ModelSpec) -> list[str]:
    return [k for k in spec.requires if not os.environ.get(k)]


def _select(selectors: list[str]) -> list[ModelSpec]:
    if selectors == ["all"]:
        return list(REGISTRY)
    index = by_path()
    chosen, unknown = [], []
    for s in selectors:
        (chosen.append(index[s]) if s in index else unknown.append(s))
    if unknown:
        sys.exit(f"Unknown model path(s): {', '.join(unknown)}\nKnown: {', '.join(index)}")
    return chosen


def _build_results_doc(spec: ModelSpec, lm_results: dict) -> dict:
    """Wrap lm-eval's ``results`` block in the leaderboard's file schema."""
    return {"config": {"model_name": spec.path}, "results": lm_results}


def _queue_entry(spec: ModelSpec, status: str, reason: str = "") -> dict:
    entry = {
        "model": spec.path,
        "status": status,
        "submitted_time": _now(),
        "model_type": "pretrained",
        "revision": "main",
        "private": False,
        "likes": 0,
        "params": 0,
        "license": spec.license,
    }
    if reason:
        entry["reason"] = reason
    return entry


def _run_one(spec: ModelSpec, tasks: list[str], limit: int | None) -> dict:
    """Run lm-eval for a single model. Imported lazily so --dry-run needs no deps."""
    from lm_eval import simple_evaluate, tasks as lm_tasks

    from runner import compat  # patch lm-eval 0.4.3 for current frontier models
    compat.apply()

    # local-chat-completions reads its key from OPENAI_API_KEY; map the provider key in.
    env_key = spec.key_env or PROVIDER_KEY_FOR_OPENAI_COMPAT.get(spec.org)
    if spec.backend == "local-chat-completions" and env_key:
        os.environ["OPENAI_API_KEY"] = os.environ[env_key]

    task_manager = lm_tasks.TaskManager(include_path=str(TASKS_PATH), include_defaults=False)
    out = simple_evaluate(
        model=spec.backend,
        model_args=spec.model_args,
        tasks=tasks,
        num_fewshot=0,
        task_manager=task_manager,
        limit=limit,
        write_out=False,
    )
    return out["results"]


def _publish(api, spec: ModelSpec, doc: dict, out_dir: Path, status: str, reason: str = "") -> None:
    """Write the results file to the results repo and the queue entry to the queue repo."""
    # Results file
    if doc is not None:
        local = out_dir / spec.path / f"results_{_now()}.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(doc, indent=2))
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{spec.path}/{local.name}",
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Add results for {spec.path}",
        )
    # Queue entry (required by the Space, even for FAILED runs)
    q_local = out_dir / "queue" / f"{spec.path}.json"
    q_local.parent.mkdir(parents=True, exist_ok=True)
    q_local.write_text(json.dumps(_queue_entry(spec, status, reason)))
    api.upload_file(
        path_or_fileobj=str(q_local),
        path_in_repo=f"{spec.path}.json",
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Set {spec.path} -> {status}",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run + publish Aiera leaderboard evals.")
    p.add_argument("--models", default="all", help="comma-separated org/model paths, or 'all'")
    p.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    p.add_argument("--limit", type=int, default=None, help="cap samples per task (smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="validate + print plan; no API calls, no upload")
    p.add_argument("--no-upload", action="store_true", help="run evals but do not publish to the datasets")
    p.add_argument("--output-dir", default=".eval-output", help="local dir for generated files")
    args = p.parse_args()

    specs = _select([s.strip() for s in args.models.split(",") if s.strip()])
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.output_dir)

    print(f"Tasks:        {', '.join(tasks)}")
    print(f"Models ({len(specs)}):")
    for s in specs:
        miss = _missing_keys(s)
        flag = f"  !! missing {', '.join(miss)}" if miss else ""
        print(f"  - {s.path:<42} [{s.backend}] model={s.model_id}{flag}")

    if args.dry_run:
        blocked = [s.path for s in specs if _missing_keys(s)]
        print(f"\nDRY RUN: registry valid. {len(specs)} model(s) selected.")
        if blocked:
            print(f"WOULD FAIL (missing keys): {', '.join(blocked)}")
        print(f"Would publish results -> {RESULTS_REPO} and queue status -> {QUEUE_REPO}.")
        return

    api = None
    if not args.no_upload:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN must be set to publish (or pass --no-upload).")
        api = HfApi(token=token)

    summary: list[tuple[str, str, str]] = []
    for spec in specs:
        miss = _missing_keys(spec)
        if miss:
            reason = f"{miss[0]} not set"
            print(f"\n[SKIP] {spec.path}: {reason}")
            if api:
                _publish(api, spec, None, out_dir, "FAILED", reason)
            summary.append((spec.path, "FAILED", reason))
            continue
        print(f"\n[RUN ] {spec.path} ({spec.model_id}) ...")
        try:
            lm_results = _run_one(spec, tasks, args.limit)
            doc = _build_results_doc(spec, lm_results)
            if api:
                _publish(api, spec, doc, out_dir, "FINISHED")
            else:
                (out_dir / spec.path).mkdir(parents=True, exist_ok=True)
                (out_dir / spec.path / f"results_{_now()}.json").write_text(json.dumps(doc, indent=2))
            print(f"[ OK ] {spec.path}")
            summary.append((spec.path, "FINISHED", ""))
        except Exception as exc:  # noqa: BLE001 — record + continue, never abort the batch
            reason = f"{type(exc).__name__}: {exc}"[:500]
            traceback.print_exc()
            if api:
                _publish(api, spec, None, out_dir, "FAILED", reason)
            summary.append((spec.path, "FAILED", reason))

    print("\n==== summary ====")
    for path, status, reason in summary:
        print(f"  {status:<9} {path}" + (f"  :: {reason}" if reason else ""))


if __name__ == "__main__":
    main()
