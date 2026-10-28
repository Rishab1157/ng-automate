"""A note on a tool call that keeps failing the same way (an NG Automate change).

A model sometimes sends a failing tool call again and again without changing it:
gpt-4o sent the same `file_editor create` 6 times in a row in one run, and smaller
models did the same with other calls. For every call (its tool and arguments) the
agent counts how many of its results in a row were errors; from the third on, the
error the model reads ends with a note that repeating the call will not help.

Only that text changes: every call still runs, nothing is skipped, blocked or
stopped, and OpenHands' StuckDetector is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, ObservationEvent
from openhands.sdk.llm import TextContent
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

#: The model is told from this failure in a row on (the 3rd, the 4th, ...).
NOTE_FROM_FAILURE = 3
#: Longer string arguments (a file's text, a long command) count by their sha1.
MAX_PLAIN_STRING_CHARS = 200
REPEATED_FAILURE_NOTE = (
    "You already sent exactly this call {n} times and it failed the same way each "
    "time. Repeating it will not help: change the call or do something else first "
    "(for example check the path, create a missing folder, or read the file)."
)


class RepeatedFailures:
    """How many results in a row of each tool call were errors, per conversation.

    A successful result of a call starts its count again.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def add_note(
        self,
        conversation_id: object,
        action_event: ActionEvent,
        events: Sequence[Event],
    ) -> list[Event]:
        """The results of one executed call, with the note added when it is due.

        Never raises: any problem here leaves the results as they are.
        """
        try:
            return self._add_note(str(conversation_id), action_event, list(events))
        except Exception:
            logger.debug("No repeated-failure note", exc_info=True)
            return list(events)

    def _add_note(
        self, conversation_id: str, action_event: ActionEvent, events: list[Event]
    ) -> list[Event]:
        failed = _failed(events)
        if failed is None:  # no tool result in it: nothing to count
            return events
        key = (conversation_id, fingerprint(action_event))
        if not failed:
            self._counts.pop(key, None)
            return events
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count < NOTE_FROM_FAILURE:
            return events
        logger.info(
            "Tool call %s failed %d times in a row; the model is told",
            action_event.tool_name,
            count,
        )
        note = REPEATED_FAILURE_NOTE.format(n=count)
        return [_with_note(event, note) for event in events]


def fingerprint(action_event: ActionEvent) -> str:
    """The tool name and the call's arguments as canonical JSON (sorted keys).

    The arguments are the ones the tool gets (without the security_risk and summary
    every call carries); string values longer than MAX_PLAIN_STRING_CHARS are
    replaced by their sha1.
    """
    action = action_event.action
    if action is not None:
        arguments: Any = action.model_dump(mode="json")
    else:
        arguments = json.loads(action_event.tool_call.arguments)
    canonical = json.dumps(
        _shortened(arguments),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{action_event.tool_name} {canonical}"


def _shortened(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_PLAIN_STRING_CHARS:
        data = value.encode("utf-8", "surrogatepass")
        return f"sha1:{hashlib.sha1(data).hexdigest()}"
    if isinstance(value, dict):
        return {key: _shortened(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shortened(item) for item in value]
    return value


def _failed(events: Sequence[Event]) -> bool | None:
    """True if a result is an error, False if none is, None without any result."""
    results = [e for e in events if isinstance(e, ObservationEvent | AgentErrorEvent)]
    if not results:
        return None
    return any(
        isinstance(e, AgentErrorEvent) or e.observation.is_error for e in results
    )


def _with_note(event: Event, note: str) -> Event:
    """The event with the note after the error text the model reads."""
    if isinstance(event, AgentErrorEvent):
        return event.model_copy(update={"error": f"{event.error}\n\n{note}"})
    if isinstance(event, ObservationEvent) and event.observation.is_error:
        extended = [*event.extended_content, TextContent(text=f"\n{note}")]
        return event.model_copy(update={"extended_content": extended})
    return event
