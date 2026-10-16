"""Plumbing shared by the OpenHands agents (analyzer, healer, test generator).

- EventForwarder: the conversation callback; maps SDK events for the run timeline, masks secrets, never raises.
- AgentControl: lets the user steer the agent that is working (messages, pause, resume, stop). Thread-safe.
- ask(): send one message, let the agent work within a deadline, return its final answer; waits while paused, and
  says in the timeline when the agent makes no progress (waiting for the model, or a long command).
- close_conversation() (returns the conversation's token usage), build_condenser(), secret_values().
"""

import logging
import shlex
import time
from collections.abc import Callable
from typing import Any

from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.error_classification import classify_error
from openhands.sdk.workspace import BaseWorkspace

from app.agents.AnalyzerAgent.AgentEventMapper import (
    AGENT_TIMED_OUT,
    MAX_ITERATIONS_CODE,
    AgentEvent,
    describe_agent_failure,
    map_openhands_event,
)
from app.core.exceptions import NgAutomateException
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventLevel, RunEventType
from app.models.tokenUsageModel import TokenUsageModel
from app.utils.AgentControl import AgentControl
from app.utils.TokenCost import priced

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
# A remote run is waited for in slices of this many seconds. Between slices a stop, the deadline and the agent's
# progress are checked: one long wait kept polling a removed sandbox (logging "connection refused" every few seconds)
# until its deadline, up to an hour after the run was stopped.
RUN_SLICE_SECONDS = 30.0
# After this long without a new agent step the timeline says what the agent waits for, then again as often.
NO_PROGRESS_NOTICE_SECONDS = 60.0
WAITING_FOR_MODEL = (
    "No new step for {minutes} min: the agent is waiting for the model's answer (the model server may be busy with "
    "another request, or slow on a long prompt)"
)
WAITING_FOR_ACTION = "No new step for {minutes} min: the agent is still running {action}"
# describe_last_steps(): how many of the last steps, and how much of a failed step's error.
MAX_LAST_STEPS = 6
MAX_STEP_DETAIL_CHARS = 200
LAST_STEPS_SUMMARY = "The agent gave no summary. Its last steps: {steps}"
MODEL_UNREACHABLE_FROM_SANDBOX = (
    "the sandbox cannot reach the model server ({reason}); check the network or VPN between Docker and the model server"
)
# check_model_reachable(): seconds curl may take, and its exit code when the image has no curl.
MODEL_CHECK_SECONDS = 10
CURL_NOT_FOUND = 127


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
    forwarder: EventForwarder | None = None,
) -> str:
    """Send one message, let the agent work, return its final answer ("" when it gave none).

    Stopping at a step limit is not a failure (the agent may still have answered); any other run error raises
    `failure(reason)` with a short, safe reason. With `control`, the user can message, pause and resume the agent:
    while paused, this waits (the pause does not count against the deadline) and then lets the agent continue.
    With `forwarder`, the timeline says when the agent makes no progress and what it is waiting for.
    """
    conversation.send_message(message)
    if control is not None:
        control.attach(conversation)
    try:
        resumes = 0
        while True:
            _run(conversation, deadline, failure, control, forwarder)
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


def _run(
    conversation: BaseConversation,
    deadline: float,
    failure: Callable[[str], NgAutomateException],
    control: AgentControl | None = None,
    forwarder: EventForwarder | None = None,
) -> None:
    try:
        if isinstance(conversation, RemoteConversation):
            _run_remote(conversation, deadline, control, forwarder)
        else:
            conversation.run()
    except ConversationRunError as error:
        if not _stopped_at_a_limit(error):
            logger.warning("Agent conversation failed", exc_info=True)
            raise failure(_run_failure_reason(error)) from error
        logger.info("Agent stopped at a step limit; reading what it answered")


def _run_remote(
    conversation: RemoteConversation,
    deadline: float,
    control: AgentControl | None,
    forwarder: EventForwarder | None,
) -> None:
    """Start the remote run once, then wait for it in slices (RUN_SLICE_SECONDS). Raises ConversationRunError like
    run(): when the run fails, at the deadline, or, once the user stopped the run, as soon as a slice ends without the
    agent finishing (its sandbox may be gone, so nothing else would end the wait).

    The later slices use the SDK's own wait (the one run() uses after posting /run): posting /run again blocks behind
    a long model call on the agent server until the request times out, which failed a real generation.
    """
    wait = getattr(conversation, "_wait_for_run_completion", None)
    armed = getattr(conversation, "_run_armed", None)
    if wait is None or armed is None:  # another SDK version: one wait until the deadline, as before
        conversation.run(timeout=max(deadline - time.monotonic(), 1.0))
        return
    since = len(conversation.state.events)
    watch = _ProgressWatch(conversation)
    started = False
    while True:
        timeout = max(min(deadline - time.monotonic(), RUN_SLICE_SECONDS), 1.0)
        try:
            if not started:
                conversation.run(timeout=timeout)
            else:
                armed.set()  # as run() does: the run's final state snapshot now counts as its end
                try:
                    wait(1.0, timeout, since)
                finally:
                    armed.clear()
            return
        except ConversationRunError as error:
            still_working = isinstance(error.original_exception, TimeoutError) and error.conversation_error is None
            if not still_working or time.monotonic() >= deadline or (control is not None and control.is_stopped):
                raise
        started = True
        if forwarder is not None:
            forwarder.notify(watch.notice())


class _ProgressWatch:
    """Notices an agent that has made no new step for a while, and says what it is waiting for."""

    def __init__(self, conversation: BaseConversation) -> None:
        self.conversation = conversation
        self.count = self._count()
        self.since = self.noticed = time.monotonic()

    def notice(self) -> AgentEvent | None:
        now, count = time.monotonic(), self._count()
        if count != self.count:
            self.count, self.since, self.noticed = count, now, now
            return None
        if now - self.since < NO_PROGRESS_NOTICE_SECONDS or now - self.noticed < NO_PROGRESS_NOTICE_SECONDS:
            return None
        self.noticed = now
        minutes = round((now - self.since) / 60)
        last = self._last()
        action = map_openhands_event(last) if isinstance(last, ActionEvent) else None
        message = (
            WAITING_FOR_ACTION.format(minutes=minutes, action=action.message)
            if action is not None
            else WAITING_FOR_MODEL.format(minutes=minutes)
        )
        return AgentEvent(RunEventType.AGENT_MESSAGE, RunEventLevel.INFO, message, {"waiting_seconds": round(now - self.since)})

    def _count(self) -> int:
        try:
            return len(self.conversation.state.events)
        except Exception:
            return 0

    def _last(self) -> Any:
        try:
            events = self.conversation.state.events
            return events[-1] if len(events) else None
        except Exception:
            return None


def _status(conversation: BaseConversation) -> str | None:
    status = getattr(getattr(conversation, "state", None), "execution_status", None)
    return getattr(status, "value", status)


def close_conversation(conversation: BaseConversation, model: str) -> TokenUsageModel | None:
    """Close the conversation; what its LLM calls used (read just before, priced for `model`), or None when that could
    not be read."""
    usage = read_token_usage(conversation, model)
    try:
        conversation.close()
    except Exception:
        logger.warning("Could not close the agent conversation", exc_info=True)
    return usage


def read_token_usage(conversation: BaseConversation, model: str) -> TokenUsageModel | None:
    """What the conversation's LLM calls used (the agent and its condenser), priced call by call for `model`, or None.
    Never raises: a failure to read the numbers must not fail the agent.

    A remote conversation is refreshed first: its cached state can miss the last calls, and holds only totals (priced
    as one call then, which can overestimate a model that costs more above a prompt size).
    """
    refresh = getattr(getattr(conversation, "state", None), "refresh_from_server", None)
    if refresh is not None:
        try:
            refresh()
        except Exception:
            logger.info("Could not refresh the agent's state; its token counts may miss the last calls", exc_info=True)
    try:
        stats = getattr(conversation, "conversation_stats", None)
        if stats is None:
            return None
        metrics = stats.get_combined_metrics()
        if not metrics.token_usages:
            total = metrics.accumulated_token_usage
            return priced(_sdk_usage(total), model) if total is not None else TokenUsageModel()
        usage = TokenUsageModel()
        for call in metrics.token_usages:
            usage = usage.add(priced(_sdk_usage(call, calls=1), model))
        return usage
    except Exception:
        logger.warning("Could not read the agent's token usage; going on without it", exc_info=True)
        return None


def _sdk_usage(usage: Any, calls: int | None = None) -> TokenUsageModel:
    return TokenUsageModel(
        input_tokens=usage.prompt_tokens,
        cached_input_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        output_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        calls=calls,
    )


def build_condenser(llm: LLM, num_ctx: int | None) -> LLMSummarizingCondenser:
    return LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_tokens=int(num_ctx * CONTEXT_WINDOW_SHARE) if num_ctx else None,
    )


def secret_values(config: LlmConfigModel) -> tuple[str, ...]:
    key = config.api_key.get_secret_value() if config.api_key else ""
    return (key,) if len(key) >= MIN_SECRET_LENGTH else ()


def describe_last_steps(conversation: BaseConversation, limit: int = MAX_LAST_STEPS) -> str:
    """The agent's last steps as "action -> result", the same step repeated shown once with its count, e.g.
    "str_replace BaseTest.java -> file_editor failed: No replacement was performed ... (x15)". "" when it took none.

    For an agent that ends without an answer (stuck, out of steps): what it tried still reaches the next attempt and
    the healer's memory, instead of "no summary".
    """
    try:
        events = list(conversation.state.events)
    except Exception:
        return ""
    steps: list[list[Any]] = []  # [text, count]
    action: AgentEvent | None = None
    for event in events:
        mapped = map_openhands_event(event)
        if mapped is None:
            continue
        if mapped.type == RunEventType.AGENT_ACTION:
            action = mapped
        elif mapped.type == RunEventType.AGENT_OBSERVATION and action is not None:
            result = mapped.message
            if mapped.data.get("error"):
                detail = next((line.strip() for line in str(mapped.data.get("output", "")).splitlines()
                               if line.strip() and not line.startswith("[An error")), "")
                result += f": {detail[:MAX_STEP_DETAIL_CHARS]}" if detail else ""
            text = f"{action.message} -> {result}"
            action = None
            if steps and steps[-1][0] == text:
                steps[-1][1] += 1
            else:
                steps.append([text, 1])
    return "; ".join(text + (f" (x{count})" if count > 1 else "") for text, count in steps[-limit:])


def check_model_reachable(workspace: BaseWorkspace, base_url: str | None) -> str | None:
    """None when the sandbox reaches the model server (any HTTP answer counts), else curl's reason (e.g. "Failed to
    connect to 192.168.3.90 port 11435"). Without this, an agent whose sandbox cannot reach the model retries its call
    silently for minutes while the run looks busy. Hosted models (no base URL) and images without curl are not checked.
    """
    if not base_url:
        return None
    try:
        result = workspace.execute_command(
            f"curl -sS -m {MODEL_CHECK_SECONDS} -o /dev/null {shlex.quote(base_url)}", timeout=MODEL_CHECK_SECONDS + 10
        )
    except Exception:
        logger.warning("Could not check whether the sandbox reaches the model server", exc_info=True)
        return None
    if result.exit_code in (0, CURL_NOT_FOUND):
        return None
    lines = [line.strip() for line in f"{result.stderr}\n{result.stdout}".splitlines() if line.strip()]
    return lines[-1][:MAX_STEP_DETAIL_CHARS] if lines else f"curl exited with {result.exit_code}"


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
