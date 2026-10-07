from typing import Any

from openhands.sdk import LLM

from ng_automate.models.llm import LLMConfig


def build_llm(cfg: LLMConfig) -> LLM:
    """The one place that turns our config into an OpenHands LLM (LiteLLM inside)."""
    options: dict[str, Any] = {}
    if cfg.num_ctx:
        options["litellm_extra_body"] = {"options": {"num_ctx": cfg.num_ctx}}
    if not cfg.thinking:
        # None = don't send any reasoning setting (the SDK default "high" turns thinking on).
        options["reasoning_effort"] = None

    return LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_mode=cfg.api_mode,
        **options,
    )
