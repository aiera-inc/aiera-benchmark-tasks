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
# OpenRouter: one OpenAI-compatible endpoint with broad serverless coverage of
# open-weight models (routes to underlying hosts — footnote "via OpenRouter").
_OPENROUTER_BASE = "base_url=https://openrouter.ai/api/v1"
# Azure OpenAI deployments (gpt-5.6-*): the OpenAI-compatible /openai/v1 surface works with the
# standard client (unlike the deployment-path API), so local-chat-completions can drive them.
_AZURE_BASE = "base_url=https://dev-mj1qg7dd-eastus2.cognitiveservices.azure.com/openai/v1"


def _openrouter(path: str, model_id: str, license: str) -> "ModelSpec":
    return ModelSpec(path, "local-chat-completions", model_id, ("OPENROUTER_API_KEY",),
                     _OPENROUTER_BASE, key_env="OPENROUTER_API_KEY", license=license)


def _together(path: str, model_id: str, license: str) -> "ModelSpec":
    return ModelSpec(path, "local-chat-completions", model_id, ("TOGETHER_API_KEY",),
                     _TOGETHER_BASE, key_env="TOGETHER_API_KEY", license=license)


def _azure(path: str, model_id: str, license: str = "proprietary") -> "ModelSpec":
    return ModelSpec(path, "local-chat-completions", model_id, ("AZURE_OPENAI_API_KEY",),
                     _AZURE_BASE, key_env="AZURE_OPENAI_API_KEY", license=license)


REGISTRY: list[ModelSpec] = [
    # --- OpenAI -----------------------------------------------------------------
    # gpt-5.5-pro is a Responses-API reasoning model (rejects v1/chat/completions),
    # so its capability run goes via OpenRouter (chat-normalized). Its research run
    # uses the native OpenAI Responses API (lift.py "openai" answerer) regardless.
    _openrouter("openai/gpt-5.5-pro", "openai/gpt-5.5-pro", "proprietary"),
    ModelSpec("openai/gpt-5.5", "openai-chat-completions", "gpt-5.5", ("OPENAI_API_KEY",)),
    ModelSpec("openai/gpt-5.2", "openai-chat-completions", "gpt-5.2", ("OPENAI_API_KEY",)),
    # gpt-5.6 cohort — Azure deployments (capability via the /openai/v1 surface; research via
    # the lift.py "azure" answerer). Reasoning models -> compat shim maps max_completion_tokens.
    _azure("openai/gpt-5.6-sol", "gpt-5.6-sol"),
    _azure("openai/gpt-5.6-luna", "gpt-5.6-luna"),
    _azure("openai/gpt-5.6-terra", "gpt-5.6-terra"),
    ModelSpec("openai/o3-2025-04-16", "openai-chat-completions", "o3-2025-04-16", ("OPENAI_API_KEY",)),
    ModelSpec("openai/o4-mini-2025-04-16", "openai-chat-completions", "o4-mini-2025-04-16", ("OPENAI_API_KEY",)),
    ModelSpec("openai/gpt-4.1-nano-2025-04-14", "openai-chat-completions", "gpt-4.1-nano-2025-04-14", ("OPENAI_API_KEY",)),
    # --- Anthropic (dateless ids from the 4.6 generation onward) ----------------
    ModelSpec("anthropic/claude-sonnet-5", "anthropic-chat", "claude-sonnet-5", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-fable-5", "anthropic-chat", "claude-fable-5", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-opus-4-8", "anthropic-chat", "claude-opus-4-8", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-opus-4-7", "anthropic-chat", "claude-opus-4-7", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-opus-4-6", "anthropic-chat", "claude-opus-4-6", ("ANTHROPIC_API_KEY",)),
    ModelSpec("anthropic/claude-sonnet-4-6", "anthropic-chat", "claude-sonnet-4-6", ("ANTHROPIC_API_KEY",)),
    # --- Google (OpenAI-compatible endpoint; verify model ids before running) ---
    ModelSpec("google/gemini-2.5-pro", "local-chat-completions", "gemini-2.5-pro", ("GEMINI_API_KEY",), _GEMINI_BASE),
    ModelSpec("google/gemini-2.5-flash", "local-chat-completions", "gemini-2.5-flash", ("GEMINI_API_KEY",), _GEMINI_BASE),
    # --- DeepSeek (OpenAI-compatible endpoint) ----------------------------------
    ModelSpec("deepseek/DeepSeek-V3-0324", "local-chat-completions", "deepseek-chat", ("DEEPSEEK_API_KEY",), _DEEPSEEK_BASE),
    # --- Open-weight + Mistral Large, served via OpenRouter (broad serverless) ---
    _openrouter("meta-llama/Llama-4-Maverick-17B-128E-Instruct", "meta-llama/llama-4-maverick", "Llama 4 Community"),
    _openrouter("meta-llama/Llama-4-Scout-17B-16E-Instruct", "meta-llama/llama-4-scout", "Llama 4 Community"),
    _openrouter("Qwen/Qwen3-235B-A22B", "qwen/qwen3-235b-a22b-2507", "Apache-2.0"),
    _openrouter("Qwen/Qwen3-32B", "qwen/qwen3-32b", "Apache-2.0"),
    _openrouter("mistralai/Mistral-Small-24B-Instruct-2501", "mistralai/mistral-small-24b-instruct-2501", "Apache-2.0"),
    _openrouter("google/gemma-3-27b-it", "google/gemma-3-27b-it", "Gemma"),
    _openrouter("openai/gpt-oss-120b", "openai/gpt-oss-120b", "Apache-2.0"),
    _openrouter("zai-org/GLM-5.2", "z-ai/glm-5.2", "MIT"),
    _openrouter("zai-org/GLM-4.6", "z-ai/glm-4.6", "MIT"),
    _openrouter("moonshotai/Kimi-K3", "moonshotai/kimi-k3", "Modified-MIT"),
    _openrouter("moonshotai/Kimi-K2.7-Code", "moonshotai/kimi-k2.7-code", "Modified-MIT"),
    _openrouter("moonshotai/Kimi-K2.6", "moonshotai/kimi-k2.6", "Modified-MIT"),
    _openrouter("Qwen/Qwen3.7-Max", "qwen/qwen3.7-max", "proprietary"),
    _openrouter("Qwen/Qwen3.7-Plus", "qwen/qwen3.7-plus", "proprietary"),
    _openrouter("deepseek/DeepSeek-V4-Pro", "deepseek/deepseek-v4-pro", "MIT"),
    _openrouter("MiniMaxAI/MiniMax-M3", "minimax/minimax-m3", "Apache-2.0"),
    _openrouter("nvidia/Nemotron-3-Ultra-550B-A55B", "nvidia/nemotron-3-ultra-550b-a55b", "NVIDIA Open Model License"),
    _openrouter("google/gemma-4-31b-it", "google/gemma-4-31b-it", "Gemma"),
    # Closed frontier routed via OpenRouter (no native GEMINI/XAI keys in env; served via OpenRouter)
    _openrouter("google/gemini-3.1-pro", "google/gemini-3.1-pro-preview", "proprietary"),
    _openrouter("google/gemini-3.5-flash", "google/gemini-3.5-flash", "proprietary"),
    _openrouter("x-ai/grok-4.3", "x-ai/grok-4.3", "proprietary"),
    _openrouter("mistralai/Mistral-Medium-3.5", "mistralai/mistral-medium-3-5", "proprietary"),
    _openrouter("mistralai/mistral-large-2512", "mistralai/mistral-large-2512", "Mistral Research"),
    # Liquid LFM2-24B-A2B: OpenRouter serves it chat-only (no tool endpoint); Together AI serves
    # it with native function calling, which the research eval requires. Served via Together.
    _together("LiquidAI/LFM2-24B-A2B", "LiquidAI/LFM2-24B-A2B", "LFM Open License"),
]


# Which env var feeds the OpenAI-compatible backend's key for each provider. The
# runner copies the provider's key into OPENAI_API_KEY for local-chat-completions.
PROVIDER_KEY_FOR_OPENAI_COMPAT = {
    "google": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def by_path() -> dict[str, ModelSpec]:
    return {m.path: m for m in REGISTRY}
