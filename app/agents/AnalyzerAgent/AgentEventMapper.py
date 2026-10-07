"""Turn OpenHands SDK events into run events for the run timeline.

Only what a user needs to follow the agent: tool calls, short tool output, the agent's own messages and
errors. Never the system prompt, our prompts, the LLM config or raw provider errors.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.event.error_classification import FailureKind
from openhands.sdk.llm import content_to_str
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobObservation, GlobTool
from openhands.tools.grep import GrepObservation, GrepTool

from app.agents.AnalyzerAgent.AnalyzerPrompts import project_relative_path
from app.models.runModel import RunEventLevel, RunEventType

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 200
MAX_MESSAGE_CHARS = 1000
MAX_OUTPUT_CHARS = 2000

MAX_ITERATIONS_CODE = "MaxIterationsReached"

# Why the agent stopped, in words a user can act on. Raw error details can contain endpoints and
# provider responses, so they stay in the server log.
AGENT_OUT_OF_STEPS = "the agent used all its steps before giving an answer"
AGENT_TIMED_OUT = "the agent did not finish in time"
AGENT_STOPPED = "the agent stopped because of an unexpected error"
MODEL_AUTH_FAILED = "the model rejected the API key of the model connection"
MODEL_QUOTA_EXCEEDED = "the usage quota or budget of the model is used up"
MODEL_RATE_LIMITED = "the model is rate limited, try again later"
MODEL_MISCONFIGURED = "the model is not available or the model connection is misconfigured"
MODEL_UNREACHABLE = "the model could not be reached, try again later"

_FAILURE_REASONS: dict[FailureKind, str] = {
    FailureKind.AUTH: MODEL_AUTH_FAILED,
    FailureKind.QUOTA: MODEL_QUOTA_EXCEEDED,
    FailureKind.RATE_LIMIT: MODEL_RATE_LIMITED,
    FailureKind.CONFIG: MODEL_MISCONFIGURED,
    FailureKind.TRANSIENT: MODEL_UNREACHABLE,
}


@dataclass(frozen=True)
class AgentEvent:
    type: RunEventType
    level: RunEventLevel
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def map_openhands_event(event: Any) -> AgentEvent | None:
    """None for everything the timeline does not show (system prompt, our messages, state updates, ...)."""
    try:
        if isinstance(event, ActionEvent):
            return _map_action(event)
        if isinstance(event, ObservationEvent):
            return _map_observation(event)
        if isinstance(event, AgentErrorEvent):
            return _map_agent_error(event)
        if isinstance(event, ConversationErrorEvent):
            return _map_conversation_error(event)
        if isinstance(event, MessageEvent) and event.source == "agent":
            return _map_agent_message(event)
    except Exception:
        # Called from the SDK's event thread: a mapping bug must cost one timeline entry, not the run.
        logger.warning("Could not map agent event %s", type(event).__name__, exc_info=True)
    return None


def describe_agent_failure(code: str | None, kind: FailureKind | None) -> str:
    """A short, safe reason for an agent that stopped with an error."""
    if code == MAX_ITERATIONS_CODE:
        return AGENT_OUT_OF_STEPS
    return _FAILURE_REASONS.get(kind, AGENT_STOPPED) if kind else AGENT_STOPPED


def _map_action(event: ActionEvent) -> AgentEvent:
    args = _action_args(event)
    data: dict[str, Any] = {"tool": event.tool_name}
    if args:
        data["args"] = args
    thought = "".join(content_to_str(event.thought)).strip()
    if thought:
        data["thought"] = clip(thought, MAX_MESSAGE_CHARS)
    return AgentEvent(RunEventType.AGENT_ACTION, RunEventLevel.INFO, _describe_action(event.tool_name, args), data)


def _map_observation(event: ObservationEvent) -> AgentEvent:
    observation = event.observation
    data: dict[str, Any] = {
        "tool": event.tool_name,
        "output": clip("".join(content_to_str(observation.to_llm_content)), MAX_OUTPUT_CHARS),
    }
    if observation.is_error:
        data["error"] = True
        message = f"{event.tool_name} failed"
    elif isinstance(observation, GlobObservation):
        message = f"glob found {len(observation.files)} file(s){' (truncated)' if observation.truncated else ''}"
    elif isinstance(observation, GrepObservation):
        message = f"grep matched {len(observation.matches)} file(s){' (truncated)' if observation.truncated else ''}"
    else:
        message = f"{event.tool_name} result"
    return AgentEvent(RunEventType.AGENT_OBSERVATION, RunEventLevel.DETAIL, message, data)


def _map_agent_error(event: AgentErrorEvent) -> AgentEvent:
    first_line = event.error.strip().splitlines()[0] if event.error.strip() else "unknown error"
    return AgentEvent(
        RunEventType.AGENT_ERROR,
        RunEventLevel.INFO,
        clip(f"{event.tool_name} error: {first_line}", MAX_SUMMARY_CHARS),
        {"tool": event.tool_name, "error": clip(event.error, MAX_OUTPUT_CHARS)},
    )


def _map_conversation_error(event: ConversationErrorEvent) -> AgentEvent:
    kind = event.classification.kind if event.classification else None
    reason = describe_agent_failure(event.code, kind)
    data: dict[str, Any] = {"code": event.code}
    if kind:
        data["kind"] = kind.value
    return AgentEvent(RunEventType.AGENT_ERROR, RunEventLevel.INFO, reason[0].upper() + reason[1:], data)


def _map_agent_message(event: MessageEvent) -> AgentEvent | None:
    text = "".join(content_to_str(event.llm_message.content)).strip()
    if not text:
        return None
    return AgentEvent(RunEventType.AGENT_MESSAGE, RunEventLevel.INFO, clip(text, MAX_MESSAGE_CHARS))


def _action_args(event: ActionEvent) -> dict[str, Any]:
    if event.action is not None:
        raw: Any = event.action.model_dump(mode="json", exclude_none=True)
    else:
        # The model's call could not be turned into an action (bad arguments): show what it sent.
        try:
            raw = json.loads(event.tool_call.arguments)
        except (TypeError, ValueError):
            raw = {"arguments": event.tool_call.arguments}
    if not isinstance(raw, dict):
        raw = {"arguments": raw}
    raw.pop("kind", None)
    return {key: _clip_value(value) for key, value in raw.items()}


def _describe_action(tool: str, args: dict[str, Any]) -> str:
    if tool == GlobTool.name:
        text = f"glob {args.get('pattern', '')}{_in_path(args)}"
    elif tool == GrepTool.name:
        include = f" ({args['include']})" if args.get("include") else ""
        text = f'grep "{args.get("pattern", "")}"{_in_path(args)}{include}'
    elif tool == FileEditorTool.name:
        text = f"{args.get('command', 'file_editor')} {project_relative_path(str(args.get('path', '')))}"
        view_range = args.get("view_range")
        if isinstance(view_range, list) and len(view_range) == 2:
            text += f" lines {view_range[0]}-{view_range[1]}"
    else:
        text = tool
    return clip(" ".join(text.split()), MAX_SUMMARY_CHARS)


def _in_path(args: dict[str, Any]) -> str:
    path = args.get("path")
    if not path:
        return ""
    relative = project_relative_path(str(path))
    return "" if relative == "." else f" in {relative}"


def _clip_value(value: Any) -> Any:
    if isinstance(value, str):
        return clip(value, MAX_MESSAGE_CHARS)
    if isinstance(value, list):
        return [_clip_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clip_value(item) for key, item in value.items()}
    return value


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
