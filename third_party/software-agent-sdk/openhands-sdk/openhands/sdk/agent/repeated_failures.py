"""Notes on tool calls that keep failing (an NG Automate change).

A model sometimes sends a failing tool call again and again without changing it:
gpt-4o sent the same `file_editor create` 6 times in a row in one run, and smaller
models did the same with other calls. For every call (its tool and arguments) the
agent counts how many of its results in a row were errors; from the third on, the
error the model reads ends with a note that repeating the call will not help.

A model can also change every call a little and still be stuck on one file:
qwen2.5-coder:14b put the same text into a file many times, then made 78 failed
file_editor calls on that file in 21 minutes, and the rule above never fired, as the
calls kept changing. So for a tool that can change files (one that does not say it only
reads, as grep and glob do) the agent also counts the failed calls in a row on each
file (the call's `path`; a relative path starts at the working directory); a
successful call of that tool on that file starts again. After every 5th failure the
error ends with a note to view the whole file again before the next edit.

Only that text changes: every call still runs, nothing is skipped, blocked or
stopped, and OpenHands' StuckDetector is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, ObservationEvent
from openhands.sdk.llm import TextContent
from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.tool import ToolDefinition


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
#: The model is told after every this many failures in a row on one file.
FILE_NOTE_EVERY = 5
FILE_FAILURE_NOTE = (
    "Your last {n} {tool} calls on {path} failed. Change your approach: view the "
    "whole file again before the next edit. If text you added is in it more than "
    "once, remove the extra copies first; then replace one complete block at a time."
)


class RepeatedFailures:
    """How many results in a row of each tool call were errors, per conversation.

    A successful result of a call starts its count again. The same is counted per
    file for the calls of a tool that can change files.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}
        self._file_counts: dict[tuple[str, str, str], int] = {}

    def add_note(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
        events: Sequence[Event],
        tool: ToolDefinition | None = None,
    ) -> list[Event]:
        """The results of one executed call, with the notes added when they are due.

        ``tool`` is the called tool, when known. Never raises: any problem here
        leaves the results as they are.
        """
        try:
            return self._add_note(conversation, action_event, list(events), tool)
        except Exception:
            logger.debug("No repeated-failure note", exc_info=True)
            return list(events)

    def _add_note(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
        events: list[Event],
        tool: ToolDefinition | None,
    ) -> list[Event]:
        failed = _failed(events)
        if failed is None:  # no tool result in it: nothing to count
            return events
        conversation_id = str(conversation.id)
        notes = [
            note
            for note in (
                self._same_call_note(conversation_id, action_event, failed),
                self._same_file_note(conversation, action_event, failed, tool),
            )
            if note
        ]
        if not notes:
            return events
        note = "\n\n".join(notes)
        return [_with_note(event, note) for event in events]

    def _same_call_note(
        self, conversation_id: str, action_event: ActionEvent, failed: bool
    ) -> str | None:
        key = (conversation_id, fingerprint(action_event))
        if not failed:
            self._counts.pop(key, None)
            return None
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count < NOTE_FROM_FAILURE:
            return None
        logger.info(
            "Tool call %s failed %d times in a row; the model is told",
            action_event.tool_name,
            count,
        )
        return REPEATED_FAILURE_NOTE.format(n=count)

    def _same_file_note(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
        failed: bool,
        tool: ToolDefinition | None,
    ) -> str | None:
        """The note after every FILE_NOTE_EVERY failures in a row on one file."""
        annotations = tool.annotations if tool is not None else None
        if annotations is not None and annotations.readOnlyHint:
            return None
        path = _path(action_event)
        if path is None:
            return None
        working_dir = str(conversation.workspace.working_dir)
        file = os.path.normpath(os.path.join(working_dir, path))
        key = (str(conversation.id), action_event.tool_name, file)
        if not failed:
            self._file_counts.pop(key, None)
            return None
        count = self._file_counts.get(key, 0) + 1
        self._file_counts[key] = count
        if count % FILE_NOTE_EVERY:
            return None
        logger.info(
            "Tool %s failed %d times in a row on %s; the model is told",
            action_event.tool_name,
            count,
            file,
        )
        return FILE_FAILURE_NOTE.format(n=count, tool=action_event.tool_name, path=path)


def fingerprint(action_event: ActionEvent) -> str:
    """The tool name and the call's arguments as canonical JSON (sorted keys).

    The arguments are the ones the tool gets (without the security_risk and summary
    every call carries); string values longer than MAX_PLAIN_STRING_CHARS are
    replaced by their sha1.
    """
    canonical = json.dumps(
        _shortened(_arguments(action_event)),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{action_event.tool_name} {canonical}"


def _arguments(action_event: ActionEvent) -> Any:
    action = action_event.action
    if action is not None:
        return action.model_dump(mode="json")
    return json.loads(action_event.tool_call.arguments)


def _path(action_event: ActionEvent) -> str | None:
    """The call's `path` argument, if it has one that is not empty."""
    arguments = _arguments(action_event)
    path = arguments.get("path") if isinstance(arguments, dict) else None
    if isinstance(path, str) and path.strip():
        return path.strip()
    return None


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
