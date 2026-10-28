"""The note on a tool call that keeps failing the same way (agent/repeated_failures.py).

The tests run the real agent loop (Agent, TestLLM and a small tool) and read what the
model reads: the tool results as LLM messages.
"""

import asyncio
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

import pytest
from pydantic import Field

from openhands.sdk.agent import Agent
from openhands.sdk.agent.repeated_failures import REPEATED_FAILURE_NOTE, fingerprint
from openhands.sdk.conversation import Conversation
from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, ObservationEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent, content_to_str
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Action, Observation, Tool, ToolExecutor, register_tool
from openhands.sdk.tool.tool import DeclaredResources, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


NOTE = "You already sent exactly this call"

# The paths the tool finds. Any other path fails with an error observation, and a
# path under /raise makes the tool raise (an AgentErrorEvent).
EXISTING: set[str] = set()


class PathAction(Action):
    path: str = Field(default="")
    text: str = Field(default="")


class PathObservation(Observation):
    pass


class PathExecutor(ToolExecutor[PathAction, PathObservation]):
    def __call__(
        self, action: PathAction, conversation: "BaseConversation | None" = None
    ) -> PathObservation:
        if action.path.startswith("/raise"):
            raise ValueError(f"cannot read {action.path}")
        found = action.path in EXISTING
        return PathObservation.from_text(
            text=f"read {action.path}" if found else f"No file at {action.path}",
            is_error=not found,
        )


class PathTool(ToolDefinition[PathAction, PathObservation]):
    name = "path_tool"

    def declared_resources(self, action: Action) -> DeclaredResources:
        return DeclaredResources(keys=(), declared=True)  # may run in parallel

    @classmethod
    def create(cls, conv_state: "ConversationState | None" = None) -> Sequence[Self]:
        return [
            cls(
                description="Reads a path.",
                action_type=PathAction,
                observation_type=PathObservation,
                executor=PathExecutor(),
            )
        ]


register_tool("RepeatedFailuresPathTool", PathTool)


@pytest.fixture(autouse=True)
def _no_existing_paths():
    EXISTING.clear()
    yield
    EXISTING.clear()


def _call(call_id: str, path: str, text: str = "") -> MessageToolCall:
    arguments = json.dumps({"path": path, "text": text})
    return MessageToolCall(
        id=call_id, name="path_tool", arguments=arguments, origin="completion"
    )


def _read(event: Event) -> str:
    """The text the model reads for a tool result."""
    assert isinstance(event, ObservationEvent | AgentErrorEvent)
    return "".join(content_to_str(event.to_llm_message().content))


class _Loop:
    """An agent whose LLM sends the given tool calls: one list per step."""

    def __init__(self, steps: list[list[MessageToolCall]], concurrency: int = 1):
        llm = TestLLM.from_messages(
            [
                Message(
                    role="assistant", content=[TextContent(text="")], tool_calls=calls
                )
                for calls in steps
            ]
        )
        self.agent = Agent(
            llm=llm,
            tools=[Tool(name="RepeatedFailuresPathTool")],
            tool_concurrency_limit=concurrency,
        )
        self.conversation = Conversation(agent=self.agent)
        self.conversation.send_message(
            Message(role="user", content=[TextContent(text="Go")])
        )

    def step(self) -> list[Event]:
        """Runs one step; its tool results, in the order of the calls."""
        events: list[Event] = []
        self.agent.step(self.conversation, on_event=events.append)
        return [e for e in events if isinstance(e, ObservationEvent | AgentErrorEvent)]

    def texts(self) -> list[str]:
        return [_read(event) for event in self.step()]


def test_the_note_comes_from_the_third_identical_failure_on():
    loop = _Loop([[_call(f"c{i}", "/w/missing.txt")] for i in range(4)])

    first, second, third, fourth = (loop.texts()[0] for _ in range(4))

    assert NOTE not in first and NOTE not in second
    assert third.endswith(REPEATED_FAILURE_NOTE.format(n=3))
    assert (
        "You already sent exactly this call 3 times and it failed the same way each "
        "time. Repeating it will not help: change the call or do something else first "
        "(for example check the path, create a missing folder, or read the file)."
    ) in third
    assert "No file at /w/missing.txt" in third  # after the tool's own error
    assert NOTE + " 4 times" in fourth


def test_only_the_text_the_model_reads_changes():
    loop = _Loop([[_call(f"c{i}", "/w/missing.txt")] for i in range(3)])
    loop.step()
    loop.step()

    (third,) = loop.step()

    assert isinstance(third, ObservationEvent)
    assert third.observation.is_error
    assert NOTE not in third.observation.text  # the tool's result is unchanged
    assert NOTE in _read(third)


def test_a_success_starts_the_count_again():
    loop = _Loop([[_call(f"c{i}", "/w/a.txt")] for i in range(6)])
    assert NOTE not in loop.texts()[0]
    assert NOTE not in loop.texts()[0]
    EXISTING.add("/w/a.txt")
    assert loop.texts() == ["read /w/a.txt"]
    EXISTING.clear()

    fourth, fifth, sixth = (loop.texts()[0] for _ in range(3))

    assert NOTE not in fourth and NOTE not in fifth
    assert NOTE + " 3 times" in sixth


def test_different_calls_get_no_note():
    loop = _Loop(
        [
            [_call("c1", "/w/a.txt")],
            [_call("c2", "/w/b.txt")],
            [_call("c3", "/w/a.txt", text="other")],
            [_call("c4", "/w/c.txt")],
            [_call("c5", "/w/b.txt", text="other")],
        ]
    )

    texts = [loop.texts()[0] for _ in range(5)]

    assert all(text.startswith("[An error occurred") for text in texts)
    assert not any(NOTE in text for text in texts)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_parallel_calls_in_one_step_are_counted_separately(concurrency):
    loop = _Loop(
        [[_call(f"a{i}", "/w/a.txt"), _call(f"b{i}", "/w/b.txt")] for i in range(3)],
        concurrency=concurrency,
    )
    for _ in range(2):
        assert not any(NOTE in text for text in loop.texts())

    a, b = loop.texts()

    assert "No file at /w/a.txt" in a and NOTE + " 3 times" in a
    assert "No file at /w/b.txt" in b and NOTE + " 3 times" in b


@pytest.mark.parametrize("concurrency", [1, 4])
def test_the_same_call_twice_in_one_step_counts_twice(concurrency):
    loop = _Loop(
        [
            [_call("c1", "/w/a.txt"), _call("c2", "/w/a.txt")],
            [_call("c3", "/w/a.txt")],
        ],
        concurrency=concurrency,
    )

    assert not any(NOTE in text for text in loop.texts())
    assert NOTE + " 3 times" in loop.texts()[0]


def test_errors_of_a_raising_tool_get_the_note_too():
    loop = _Loop([[_call(f"c{i}", "/raise/x")] for i in range(3)])
    loop.step()
    loop.step()

    (third,) = loop.step()

    assert isinstance(third, AgentErrorEvent)
    assert third.error.startswith("Error executing tool 'path_tool': cannot read")
    assert third.error.endswith(REPEATED_FAILURE_NOTE.format(n=3))


def test_the_async_loop_adds_the_note_too():
    loop = _Loop([[_call(f"c{i}", "/w/missing.txt")] for i in range(3)])
    results: list[Event] = []
    for _ in range(3):
        results.clear()
        asyncio.run(loop.agent.astep(loop.conversation, on_event=results.append))

    errors = [e for e in results if isinstance(e, ObservationEvent)]
    assert len(errors) == 1 and NOTE + " 3 times" in _read(errors[0])


def _action_event(arguments: dict) -> ActionEvent:
    call = MessageToolCall(
        id="c1",
        name="path_tool",
        arguments=json.dumps(arguments),
        origin="completion",
    )
    return ActionEvent(
        thought=[],
        action=PathAction(**arguments),
        tool_name="path_tool",
        tool_call_id="c1",
        tool_call=call,
        llm_response_id="r1",
    )


def test_fingerprint_sorts_keys_and_hashes_long_strings():
    long_text = "x" * 201
    same = _action_event({"path": "/w/a", "text": "hi"})
    reordered = _action_event({"text": "hi", "path": "/w/a"})
    long = _action_event({"path": "/w/a", "text": long_text})

    assert fingerprint(same) == fingerprint(reordered)
    assert fingerprint(same).startswith('path_tool {"')
    assert '"path":"/w/a"' in fingerprint(same)
    assert long_text not in fingerprint(long) and '"text":"sha1:' in fingerprint(long)
    assert '"text":"' + "x" * 200 + '"' in fingerprint(
        _action_event({"path": "/w/a", "text": "x" * 200})
    )
