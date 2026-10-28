"""Build the LLM an OpenHands agent talks to, and check that a self-hosted one answers before an agent uses it."""

import ipaddress
import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.models.modelConnectionModel import ModelConnectionSourceModel

logger = logging.getLogger(__name__)

# Hosts that mean "the machine this process runs on". Inside a sandbox container they point at the container.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
MODEL_NOT_ANSWERING = (
    "the model server is up but did not answer a 1-token request within {seconds} s: it is busy or stuck, try again "
    "later"
)

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

    if config.model.startswith("ollama") and config.base_url:
        # The SDK looks up an Ollama model's info on the host without its base URL, so LiteLLM asks localhost:11434
        # (waiting up to 100 minutes when something there never answers). Point that lookup at the model's server.
        os.environ.setdefault("OLLAMA_API_BASE", config.base_url)
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
        # Pauses before the next tries: min, 2 x min, 4 x min, ... at most max (the SDK doubles its multiplier per try).
        retry_min_wait=settings.LLM_RETRY_MIN_WAIT_SECONDS,
        retry_max_wait=settings.LLM_RETRY_MAX_WAIT_SECONDS,
        retry_multiplier=settings.LLM_RETRY_MIN_WAIT_SECONDS,
        **options,
    )


def is_self_hosted(base_url: str | None) -> bool:
    """A model server on this machine or its local network: a private or loopback address, or a host name without a
    domain (localhost, a container's name, the sandbox's host alias). Asking it costs nothing, unlike a hosted gateway,
    where every request is paid for."""
    host = urlsplit(base_url).hostname if base_url else None
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host or host == settings.SANDBOX_HOST_ALIAS
    return address.is_private or address.is_loopback


def check_model_answers(config: LlmConfigModel) -> str | None:
    """None when a self-hosted model answers a 1-token request within MODEL_ANSWER_CHECK_SECONDS (0: not checked),
    else why not.

    Asked once before an agent starts or test data is read: that its server answers HTTP shows only that it is up, and
    a stuck one held a real run for the full timeout on every try. A hosted model is never asked (every request to it
    is paid for). An error other than no answer in time is left to the real calls, which report it.
    """
    seconds = settings.MODEL_ANSWER_CHECK_SECONDS
    if seconds <= 0 or not is_self_hosted(config.base_url):
        return None
    from openhands.sdk.llm import Message, TextContent
    from openhands.sdk.llm.exceptions import LLMTimeoutError

    one_token: dict[str, Any] = {"max_output_tokens": 1, "timeout": seconds, "num_retries": 1}
    if config.num_ctx:
        # Ollama: the context size the agents use (another one makes it load the model again), and one token.
        one_token["litellm_extra_body"] = {"options": {"num_ctx": config.num_ctx, "num_predict": 1}}
    # Asked from this process: the configured URL, not the sandbox's alias for it.
    llm = build_llm(config).model_copy(update={"base_url": config.base_url, **one_token})
    try:
        llm.completion([Message(role="user", content=[TextContent(text="Reply with OK.")])])
    except LLMTimeoutError:
        return MODEL_NOT_ANSWERING.format(seconds=seconds)
    except Exception as error:
        logger.info("The model check got %s: left to the agents' own calls", type(error).__name__)
    return None


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
