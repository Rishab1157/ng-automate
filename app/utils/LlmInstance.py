"""Build the LLM an OpenHands agent talks to."""

from typing import Any

from openhands.sdk import LLM

from app.config import settings
from app.models.llmModel import LlmConfigModel


def get_default_llm_config() -> LlmConfigModel:
    """The default LLM from settings (e.g. Devstral on the innovation server)."""
    return LlmConfigModel(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        api_mode=settings.LLM_API_MODE,
        num_ctx=settings.LLM_NUM_CTX,
        thinking=settings.LLM_THINKING,
    )


def build_llm(config: LlmConfigModel) -> LLM:
    options: dict[str, Any] = {}
    if config.num_ctx:
        options["litellm_extra_body"] = {"options": {"num_ctx": config.num_ctx}}
    if not config.thinking:
        # None = send no reasoning setting at all (the SDK default "high" turns thinking on).
        options["reasoning_effort"] = None

    return LLM(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        api_mode=config.api_mode,
        **options,
    )
