"""Get JSON that matches a schema from an LLM, using the provider's schema-constrained output mode.

A plain LLM call (no agent, no tools): the provider is asked for `response_format: json_schema` (Ollama
structured outputs, OpenAI strict JSON schema), so the reply can only be JSON of that shape. The caller's parser
still validates it; a reply that fails is retried once more with the exact error. A rate-limited call (HTTP 429) is
tried again after the pauses the agents' model calls make (LLM_NUM_RETRIES, LLM_RETRY_MIN/MAX_WAIT_SECONDS).
"""

import logging
from collections.abc import Callable
from time import sleep  # this module's own name for it: tests replace it here, never time.sleep itself
from typing import Any, TypeVar

import litellm
from openhands.sdk.llm.exceptions import is_quota_exhaustion_error

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.models.tokenUsageModel import TokenUsageModel
from app.utils.TokenCost import priced

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ERROR_CHARS = 500


def complete_json(
    llm_config: LlmConfigModel,
    instructions: str,
    text: str,
    schema_name: str,
    schema: dict[str, Any],
    parse: Callable[[str], T],
    attempts: int = settings.STRUCTURED_OUTPUT_ATTEMPTS,
    timeout_seconds: int = 300,
    on_usage: Callable[[TokenUsageModel], None] | None = None,
) -> T:
    """Ask for JSON matching `schema`, validated by `parse` (which raises ValueError when it is not usable).

    Raises ValueError with a short reason when no attempt produced a usable reply. `on_usage` gets what each answered
    call used (tokens, estimated input and output cost).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": text},
    ]
    reason = "no attempt was made"
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            reply = _complete_with_pauses(llm_config, messages, schema_name, schema, timeout_seconds, on_usage)
        except Exception as error:
            # Provider errors can quote endpoints or keys: log the type only.
            logger.warning("Structured output call failed (attempt %d): %s", attempt, type(error).__name__)
            reason = f"the model call failed ({type(error).__name__})"
            if _rate_limited(error):
                break  # it was already tried again after pauses: never once more right away
            continue
        try:
            return parse(reply)
        except ValueError as error:
            reason = str(error)[:MAX_ERROR_CHARS]
            logger.info("Structured output attempt %d was not usable: %s", attempt, reason)
            messages = [
                *messages,
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That JSON is not valid: {reason}. Reply with the corrected JSON only."},
            ]
    raise ValueError(reason)


def _complete_with_pauses(*args: Any) -> str:
    """_complete, tried again while the provider rate-limits (HTTP 429), as the agents' model calls are: up to
    LLM_NUM_RETRIES tries in all, the pause doubling from LLM_RETRY_MIN_WAIT_SECONDS up to LLM_RETRY_MAX_WAIT_SECONDS.
    A used-up quota is not waited for (as the SDK does): no pause brings it back."""
    for attempt in range(1, settings.LLM_NUM_RETRIES):
        try:
            return _complete(*args)
        except Exception as error:
            if not _rate_limited(error) or is_quota_exhaustion_error(error):
                raise
            pause = min(settings.LLM_RETRY_MIN_WAIT_SECONDS * 2 ** (attempt - 1), settings.LLM_RETRY_MAX_WAIT_SECONDS)
            logger.warning("The model is rate limited: trying again in %d s", pause)
            sleep(pause)
    return _complete(*args)  # the last try: its error goes to the caller


def _rate_limited(error: Exception) -> bool:
    """HTTP 429 (LiteLLM's RateLimitError carries it too)."""
    return getattr(error, "status_code", None) == 429


def _complete(
    llm_config: LlmConfigModel,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    timeout_seconds: int,
    on_usage: Callable[[TokenUsageModel], None] | None = None,
) -> str:
    options: dict[str, Any] = {}
    if llm_config.num_ctx:
        options["num_ctx"] = llm_config.num_ctx
    response = litellm.completion(
        model=llm_config.model,
        messages=messages,
        api_base=llm_config.base_url,
        api_key=llm_config.api_key.get_secret_value() if llm_config.api_key else None,
        response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}},
        temperature=0,
        timeout=timeout_seconds,
        **options,
    )
    if on_usage is not None:
        on_usage(priced(_usage(response), llm_config.model))
    return response.choices[0].message.content or ""


def _usage(response: Any) -> TokenUsageModel:
    """What one call used, as LiteLLM reports it. Never raises: a count it does not report stays 0."""
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return TokenUsageModel(
        input_tokens=_count(getattr(usage, "prompt_tokens", 0)),
        cached_input_tokens=_count(getattr(prompt_details, "cached_tokens", 0)),
        cache_write_tokens=_count(getattr(usage, "cache_creation_input_tokens", 0)),
        output_tokens=_count(getattr(usage, "completion_tokens", 0)),
        reasoning_tokens=_count(getattr(completion_details, "reasoning_tokens", 0)),
        calls=1,
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
