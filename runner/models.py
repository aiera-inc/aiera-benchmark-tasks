"""Model registry for the Aiera Finance Leaderboard.

Each entry maps a *leaderboard path* (``org/model`` — how the model is displayed on
the board and where its files live in the results/queue datasets) to the concrete
provider backend + API model id used to run the evaluation.

Why this file exists
--------------------
The leaderboard only ever shows models that have a complete results file in
``Aiera/aiera-leaderboard-results``. Historically, runs were kicked off ad-hoc with
hand-typed model ids, which produced two recurring failures visible in the queue:

* **Wrong model id (404 from the provider).** e.g. ``claude-opus-4-6-20250725``.
  Starting with the Claude 4.6 generation, Anthropic ids are *dateless* pinned
  snapshots (``claude-opus-4-6``), so the dated form 404s.
* **Missing provider key in the eval environment.** e.g. ``OPENAI_API_KEY not set``.

Keeping the ids in one reviewed place fixes the first class of failure; ``requires``
makes the second fail loudly *before* a run instead of silently mid-run.

Verify ids against each provider's models endpoint before adding new ones:
* Anthropic: ``GET https://api.anthropic.com/v1/models``
* OpenAI:    ``GET https://api.openai.com/v1/models``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    path: str  # leaderboard "org/model" — display name + dataset path
    backend: str  # lm-eval model name (see lm_eval/models/*; registered names)
    model_id: str  # provider API model id passed to the backend
    requires: tuple[str, ...]  # env vars that MUST be set for this model to run
    extra_args: str = ""  # extra lm-eval model_args, comma-joined (e.g. base_url=...)
    key_env: str = ""  # env var holding the key to feed an OpenAI-compatible backend (overrides PROVIDER_KEY_FOR_OPENAI_COMPAT)
    license: str = "proprietary"

    @property
    def org(self) -> str:
        return self.path.split("/", 1)[0]

    @property
    def model_args(self) -> str:
        """The ``--model_args`` string handed to lm-eval."""
        args = f"model={self.model_id}"
        return f"{args},{self.extra_args}" if self.extra_args else args


# Google/DeepSeek expose OpenAI-compatible endpoints, so they reuse lm-eval's
# ``local-chat-completions`` backend with a base_url override. That backend reads
# its key from OPENAI_API_KEY, so the runner maps the provider key into it.
# API roots only — the OpenAI client appends "/chat/completions" itself.
_GEMINI_BASE = "base_url=https://generativelanguage.googleapis.com/v1beta/openai"
_DEEPSEEK_BASE = "base_url=https://api.deepseek.com"
# Together AI hosts current open-weight models on one OpenAI-compatible endpoint.
# (Serving config/quantization is provider-specific — footnote "served via Together"
# on the board for apples-to-apples comparison.)
_TOGETHER_BASE = "base_url=https://api.together.xyz/v1"
# Mistral's own API (la Plateforme) hosts Mistral Large, which Together does not.
_MISTRAL_BASE = "base_url=https://api.mistral.ai/v1"


def _together(path: str, model_id: str, license: str) -> "ModelSpec":
    return ModelSpec(path, "local-chat-completions", model_id, ("TOGETHER_API_KEY",),
                     _TOGETHER_BASE, key_env="TOGETHER_API_KEY", license=license)


REGISTRY: list[ModelSpec] = [
    # --- OpenAI -----------------------------------------------------------------
    ModelSpec("openai/gpt-5.5", "openai-chat-completions", "gpt-5.5", ("OPENAI_API_KEY",)),
    ModelSpec("openai/gpt-5.2", "openai-chat-completions", "gpt-5.2", ("OPENAI_API_KEY",)),
    ModelSpec("openai/o3-2025-04-16", "openai-chat-completions", "o3-2025-04-16", ("OPENAI_API_KEY",)),
    ModelSpec("openai/o4-mini-2025-04-16", "openai-chat-completions", "o4-mini-2025-04-16", ("OPENAI_API_KEY",)),
    ModelSpec("openai/gpt-4.1-nano-2025-04-14", "openai-chat-completions", "gpt-4.1-nano-2025-04-14", ("OPENAI_API_KEY",)),
    # --- Anthropic (dateless ids from the 4.6 generation onward) ----------------
    ModelSpec("anthropic/claude-opus-4-8", "anthropic-chat", "claude-opus-4-8", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-opus-4-7", "anthropic-chat", "claude-opus-4-7", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-opus-4-6", "anthropic-chat", "claude-opus-4-6", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-sonnet-4-6", "anthropic-chat", "claude-sonnet-4-6", ("ANTHROPIC_API_KEY",)),
    # --- Google (OpenAI-compatible endpoint; verify model ids before running) ---
    ModelSpec("google/gemini-2.5-pro", "local-chat-completions", "gemini-2.5-pro", ("GEMINI_API_KEY",), _GEMINI_BASE),
    ModelSpec("google/gemini-2.5-flash", "local-chat-completions", "gemini-2.5-flash", ("GEMINI_API_KEY",), _GEMINI_BASE),
    # --- DeepSeek (OpenAI-compatible endpoint) ----------------------------------
    ModelSpec("deepseek/DeepSeek-V3-0324", "local-chat-completions", "deepseek-chat", ("DEEPSEEK_API_KEY",), _DEEPSEEK_BASE),
    # --- Open-weight, served via Together AI ------------------------------------
    _together("meta-llama/Llama-4-Maverick-17B-128E-Instruct", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama 4 Community"),
    _together("meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Community"),
    _together("Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "Apache-2.0"),
    _together("Qwen/Qwen3-32B", "Qwen/Qwen3-32B", "Apache-2.0"),
    _together("mistralai/Mistral-Small-24B-Instruct-2501", "mistralai/Mistral-Small-24B-Instruct-2501", "Apache-2.0"),
    _together("google/gemma-3-27b-it", "google/gemma-3-27b-it", "Gemma"),
    _together("openai/gpt-oss-120b", "openai/gpt-oss-120b", "Apache-2.0"),
    _together("zai-org/GLM-4.6", "zai-org/GLM-4.6", "MIT"),
    _together("moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.6", "Modified-MIT"),
    # --- Mistral Large (Mistral's own API; not hosted on Together) ---------------
    ModelSpec("mistralai/mistral-large-2512", "local-chat-completions", "mistral-large-2512",
              ("MISTRAL_API_KEY",), _MISTRAL_BASE, key_env="MISTRAL_API_KEY", license="Mistral Research"),
]


# Which env var feeds the OpenAI-compatible backend's key for each provider. The
# runner copies the provider's key into OPENAI_API_KEY for local-chat-completions.
PROVIDER_KEY_FOR_OPENAI_COMPAT = {
    "google": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def by_path() -> dict[str, ModelSpec]:
    return {m.path: m for m in REGISTRY}
