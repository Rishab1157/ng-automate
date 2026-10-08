"""Get JSON that matches a schema from an LLM, using the provider's schema-constrained output mode.

A plain LLM call (no agent, no tools): the provider is asked for `response_format: json_schema` (Ollama
structured outputs, OpenAI strict JSON schema), so the reply can only be JSON of that shape. The caller's parser
still validates it; a reply that fails is retried once more with the exact error.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import litellm

from app.config import settings
from app.models.llmModel import LlmConfigModel

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
) -> T:
    """Ask for JSON matching `schema`, validated by `parse` (which raises ValueError when it is not usable).

    Raises ValueError with a short reason when no attempt produced a usable reply.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": text},
    ]
    reason = "no attempt was made"
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            reply = _complete(llm_config, messages, schema_name, schema, timeout_seconds)
        except Exception as error:
            # Provider errors can quote endpoints or keys: log the type only.
            logger.warning("Structured output call failed (attempt %d): %s", attempt, type(error).__name__)
            reason = f"the model call failed ({type(error).__name__})"
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


def _complete(
    llm_config: LlmConfigModel,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    timeout_seconds: int,
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
    return response.choices[0].message.content or ""
