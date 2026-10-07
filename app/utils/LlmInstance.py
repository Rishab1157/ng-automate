"""Build the LLM an OpenHands agent talks to."""

from typing import Any

from openhands.sdk import LLM

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.models.modelConnectionModel import ModelConnectionSourceModel

# QXcel provider_code -> (LiteLLM prefix, keep QXcel's base URL?, api_mode).
# Native providers use LiteLLM's own endpoints; QXcel's stored base URLs include paths LiteLLM does not expect.
_PROVIDER_ROUTES: dict[str, tuple[str, bool, str]] = {
    "OPENAI": ("openai", True, "auto"),
    "AGENTICQE": ("openai", True, "chat"),  # gateway that only allows /chat/completions
    "ANTHROPIC": ("anthropic", False, "auto"),
    "GEMINI": ("gemini", False, "auto"),
    "GROQ": ("groq", False, "auto"),
    "GROK(XAI)": ("xai", False, "auto"),
    "XAI": ("xai", False, "auto"),
}
# Unknown providers are treated as OpenAI-compatible endpoints.
_FALLBACK_ROUTE = ("openai", True, "chat")


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


def llm_config_from_connection(source: ModelConnectionSourceModel) -> LlmConfigModel:
    """LLM config for a QXcel model connection."""
    provider = source.provider_code.strip().upper()
    prefix, keep_base_url, api_mode = _PROVIDER_ROUTES.get(provider, _FALLBACK_ROUTE)
    model_name = source.model_name.lower() if provider == "GEMINI" else source.model_name
    base_url = source.api_base_url if keep_base_url else None
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = None
    return LlmConfigModel(
        model=f"{prefix}/{model_name}",
        base_url=base_url,
        api_key=source.api_key,
        api_mode=api_mode,
    )
