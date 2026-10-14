"""Build the LLM an OpenHands agent talks to."""

from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.models.modelConnectionModel import ModelConnectionSourceModel

# Hosts that mean "the machine this process runs on". Inside a sandbox container they point at the container.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

if TYPE_CHECKING:
    # The OpenHands SDK (and LiteLLM under it) takes seconds to import: only agents load it, not the API.
    from openhands.sdk import LLM

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
        native_tool_calling=settings.LLM_NATIVE_TOOL_CALLING,
    )


def sandbox_base_url(base_url: str | None) -> str | None:
    """The base URL a sandboxed agent must use to reach the model.

    Agents run in Docker containers, where localhost/127.0.0.1 is the container itself, so a model server on
    the machine that runs the API is unreachable by that name. Sandboxes are started with a host-gateway alias
    (SANDBOX_HOST_ALIAS), so a loopback URL is rewritten to it. A URL that already points elsewhere is left
    untouched: a hosted model is reachable from anywhere.
    """
    if not base_url:
        return base_url
    parts = urlsplit(base_url)
    if parts.hostname not in _LOOPBACK_HOSTS:
        return base_url
    host = settings.SANDBOX_HOST_ALIAS
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def build_llm(config: LlmConfigModel) -> "LLM":
    from openhands.sdk import LLM

    options: dict[str, Any] = {}
    if config.num_ctx:
        # Ollama's own options; its length limit goes here too, as these options replace the ones LiteLLM would send.
        options["litellm_extra_body"] = {
            "options": {"num_ctx": config.num_ctx, "num_predict": settings.LLM_MAX_OUTPUT_TOKENS}
        }
    if not config.thinking:
        # None = send no reasoning setting at all (the SDK default "high" turns thinking on).
        options["reasoning_effort"] = None

    return LLM(
        model=config.model,
        api_key=config.api_key,
        base_url=sandbox_base_url(config.base_url),
        api_mode=config.api_mode,
        native_tool_calling=config.native_tool_calling,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        num_retries=settings.LLM_NUM_RETRIES,
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
