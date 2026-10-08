"""Plumbing shared by the OpenHands agents (analyzer, healer, test generator).

- EventForwarder: the conversation callback; maps SDK events for the run timeline, masks secrets, never raises.
- AgentControl: lets the user steer the agent that is working (messages, pause, resume, stop). Thread-safe.
- ask(): send one message, let the agent work within a deadline, return its final answer; waits while paused.
- close_conversation(), build_condenser(), secret_values().
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.event.error_classification import classify_error

from app.agents.AnalyzerAgent.AgentEventMapper import (
    AGENT_TIMED_OUT,
    MAX_ITERATIONS_CODE,
    AgentEvent,
    describe_agent_failure,
    map_openhands_event,
)
from app.core.exceptions import NgAutomateException
from app.models.llmModel import LlmConfigModel
from app.utils.AgentControl import AgentControl

logger = logging.getLogger(__name__)

# Summarize old steps before the history fills this share of the context window. Ollama silently drops
# the start of a prompt that is too long, which would lose the task and the answer format.
CONTEXT_WINDOW_SHARE = 0.6
# Shorter values are not real secrets, and masking them would mangle ordinary words in the timeline.
MIN_SECRET_LENGTH = 8
SECRET_MASK = "***"
# run() calls after a resume: one may return at once while a step from before the pause is still finishing.
MAX_RESUMES = 50
_PAUSED_STATUS = "paused"


class EventForwarder:
    """The conversation callback: maps SDK events and hands them to on_event.

    It never raises. The SDK calls its callbacks one after another without a guard, and its own event
    bookkeeping (which the final answer is read from) runs after ours.
    """

    def __init__(self, on_event: Callable[[AgentEvent], None] | None, secrets: tuple[str, ...]) -> None:
        self.on_event = on_event
        self.secrets = secrets
        self._failed = False

    def __call__(self, event: Any) -> None:
        if self.on_event is not None:
            self.notify(map_openhands_event(event))

    def notify(self, event: AgentEvent | None) -> None:
        if event is None or self.on_event is None:
            return
        try:
            self.on_event(_mask_event(event, self.secrets))
        except Exception:
            if not self._failed:
                logger.exception("Agent on_event callback failed; the agent continues")
            self._failed = True


def ask(
    conversation: BaseConversation,
    message: str,
    deadline: float,
    failure: Callable[[str], NgAutomateException],
    control: AgentControl | None = None,
) -> str:
    """Send one message, let the agent work, return its final answer ("" when it gave none).

    Stopping at a step limit is not a failure (the agent may still have answered); any other run error raises
    `failure(reason)` with a short, safe reason. With `control`, the user can message, pause and resume the agent:
    while paused, this waits (the pause does not count against the deadline) and then lets the agent continue.
    """
    conversation.send_message(message)
    if control is not None:
        control.attach(conversation)
    try:
        resumes = 0
        while True:
            _run(conversation, deadline, failure)
            if control is None or _status(conversation) != _PAUSED_STATUS or control.is_stopped:
                break
            # The user paused the agent (or a step from before the pause just finished): wait, then go on.
            deadline += control.wait_while_paused()
            resumes += 1
            if control.is_stopped or resumes > MAX_RESUMES:
                break
    finally:
        if control is not None:
            control.detach(conversation)
    return get_agent_final_response(conversation.state.events)


def _run(conversation: BaseConversation, deadline: float, failure: Callable[[str], NgAutomateException]) -> None:
    try:
        if isinstance(conversation, RemoteConversation):
            # Without a timeout a remote run keeps polling for up to an hour, even after its sandbox is gone.
            conversation.run(timeout=max(deadline - time.monotonic(), 1.0))
        else:
            conversation.run()
    except ConversationRunError as error:
        if not _stopped_at_a_limit(error):
            logger.warning("Agent conversation failed", exc_info=True)
            raise failure(_run_failure_reason(error)) from error
        logger.info("Agent stopped at a step limit; reading what it answered")


def _status(conversation: BaseConversation) -> str | None:
    status = getattr(getattr(conversation, "state", None), "execution_status", None)
    return getattr(status, "value", status)


def close_conversation(conversation: BaseConversation) -> None:
    try:
        conversation.close()
    except Exception:
        logger.warning("Could not close the agent conversation", exc_info=True)


def build_condenser(llm: LLM, num_ctx: int | None) -> LLMSummarizingCondenser:
    return LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_tokens=int(num_ctx * CONTEXT_WINDOW_SHARE) if num_ctx else None,
    )


def secret_values(config: LlmConfigModel) -> tuple[str, ...]:
    key = config.api_key.get_secret_value() if config.api_key else ""
    return (key,) if len(key) >= MIN_SECRET_LENGTH else ()


def _stopped_at_a_limit(error: ConversationRunError) -> bool:
    """Out of steps or stuck: the agent can still answer with what it found (a remote run raises for these)."""
    if error.conversation_error is not None:
        return error.conversation_error.code == MAX_ITERATIONS_CODE
    return "stuck" in str(error.original_exception).lower()


def _run_failure_reason(error: ConversationRunError) -> str:
    event = error.conversation_error
    if event is not None:
        return describe_agent_failure(event.code, event.classification.kind if event.classification else None)
    original = error.original_exception
    if isinstance(original, TimeoutError):
        return AGENT_TIMED_OUT
    code = type(original).__name__
    return describe_agent_failure(code, classify_error(code, str(original)).kind)


def _mask_event(event: AgentEvent, secrets: tuple[str, ...]) -> AgentEvent:
    """Defense in depth: the API key never reaches the timeline, even if the agent read it somewhere."""
    if not secrets:
        return event
    return AgentEvent(event.type, event.level, _mask(event.message, secrets), _mask(event.data, secrets))


def _mask(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, SECRET_MASK)
        return value
    if isinstance(value, list):
        return [_mask(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _mask(item, secrets) for key, item in value.items()}
    return value
