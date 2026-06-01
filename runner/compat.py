"""Compatibility shims that let lm-eval 0.4.3 drive current frontier models.

lm-eval 0.4.3 (pinned in environment.yml) predates GPT-5.x / o-series / Claude 4.x
and breaks on all of them in ways that manifest as a silent multi-minute hang:

* OpenAI chat: sends the now-rejected ``max_tokens`` (reasoning models require
  ``max_completion_tokens``) and forces ``temperature=0`` (reasoning models reject
  any non-default temperature).
* Anthropic chat: ``AnthropicChatLM.generate_until`` passes ``stop=until`` straight
  into ``messages.create(**kwargs)``, but the Messages API param is ``stop_sequences``.
* Both wrap the call in ``@retry_on_specific_exceptions(max_retries=None)`` with no
  request timeout, so any of the above retries *forever* instead of surfacing.
* Both use the task's tiny ``max_gen_toks`` (256-500) as the cap. A reasoning model
  spends that entirely on hidden reasoning tokens and returns an empty answer.

``apply()`` monkeypatches the two module-level functions where the API call actually
happens (``oa_completion`` and ``anthropic_chat``). Because lm-eval looks those names
up as module globals at call time, replacing them is sufficient — no subclassing or
re-registration needed. Tunables are env-overridable for smoke-test iteration.
"""

from __future__ import annotations

import os

# Reasoning models emit hidden reasoning tokens before the visible answer, so the
# task's 256-500 budget comes back empty. Give the completion room; you only pay
# for tokens actually generated, and non-reasoning models stop well before this.
REASONING_BUDGET = int(os.environ.get("LB_MAX_COMPLETION_TOKENS", "16000"))
ANTHROPIC_MIN_BUDGET = int(os.environ.get("LB_ANTHROPIC_MIN_TOKENS", "2048"))
# Open-weight models behind OpenAI-compatible hosts (Together etc.) often have a
# thinking mode on by default; give them room so the answer isn't truncated by
# reasoning tokens. You only pay for tokens actually generated.
COMPAT_MIN_BUDGET = int(os.environ.get("LB_COMPAT_MIN_TOKENS", "8000"))
REQUEST_TIMEOUT = float(os.environ.get("LB_REQUEST_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("LB_MAX_RETRIES", "4"))

_applied = False


def _is_openai_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith(("o1", "o3", "o4")) or m.startswith("gpt-5")


def _COMPAT_REASONING(model: str) -> bool:
    # Reasoning models behind OpenAI-compatible endpoints (Gemini 2.5, DeepSeek-R1)
    # spend hidden tokens; DeepSeek-V3 (chat) does not.
    m = (model or "").lower()
    return "gemini-2.5" in m or "deepseek-r1" in m or "deepseek-reasoner" in m


def apply() -> None:
    """Idempotently patch the OpenAI and Anthropic call sites in lm-eval 0.4.3."""
    global _applied
    if _applied:
        return

    # --- OpenAI chat completions ------------------------------------------------
    import lm_eval.models.openai_completions as oc

    def patched_oa_completion(client, chat: bool = False, **kwargs):
        if not chat:
            return client.completions.create(**kwargs)
        model = kwargs.get("model", "")
        base = str(getattr(client, "base_url", "") or "")
        requested = kwargs.pop("max_tokens", None) or 0
        if "openai.com" in base:
            # OpenAI native: reasoning models need max_completion_tokens and reject
            # temperature/stop; the `until` truncation is re-applied by lm-eval post-hoc.
            if _is_openai_reasoning(model):
                kwargs["max_completion_tokens"] = max(requested, REASONING_BUDGET)
                kwargs.pop("temperature", None)
                kwargs.pop("stop", None)
            else:
                kwargs["max_completion_tokens"] = requested or 256
        else:
            # OpenAI-compatible endpoint (Gemini/DeepSeek/Together): keep plain max_tokens
            # and leave temperature/stop alone; give all of them headroom since many
            # open-weight models emit hidden reasoning before the answer.
            kwargs["max_tokens"] = max(requested, COMPAT_MIN_BUDGET)
        client = client.with_options(timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
        return client.chat.completions.create(**kwargs)

    oc.oa_completion = patched_oa_completion

    # --- Anthropic messages -----------------------------------------------------
    import anthropic
    import lm_eval.models.anthropic_llms as al

    # The anthropic SDK removed Anthropic.get_tokenizer(), which AnthropicLM.__init__
    # calls. It only feeds the loglikelihood path (unused for our generate_until
    # tasks), so a no-op stub is safe and avoids the AttributeError at construction.
    if not hasattr(anthropic.Anthropic, "get_tokenizer"):
        anthropic.Anthropic.get_tokenizer = lambda self: None

    def patched_anthropic_chat(client, model, prompt, max_tokens, temperature, stop=None, **kwargs):
        kwargs.pop("until", None)
        budget = max(int(max_tokens or 0), ANTHROPIC_MIN_BUDGET)
        create_kwargs = {
            "model": model,
            "max_tokens": budget,
            "messages": [{"role": "user", "content": f"{prompt}"}],
            **kwargs,
        }
        # Claude 4.x+ deprecated the temperature param; 2.x/3.x still accept it.
        keeps_temp = "claude-2" in model.lower() or "claude-3" in model.lower()
        if temperature is not None and keeps_temp:
            create_kwargs["temperature"] = temperature
        # Anthropic rejects whitespace-only stop sequences (OpenAI tolerates them).
        seqs = [s for s in (stop or []) if isinstance(s, str) and s.strip()]
        if seqs:
            create_kwargs["stop_sequences"] = seqs  # correct Messages-API param name
        client = client.with_options(timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
        resp = client.messages.create(**create_kwargs)
        for block in resp.content:  # skip any non-text (thinking/tool) blocks
            if getattr(block, "type", None) == "text":
                return block.text
        return resp.content[0].text if resp.content else ""

    al.anthropic_chat = patched_anthropic_chat

    _applied = True
