"""AgentControl and ask(): the user steers the agent that is working (messages, pause, resume, stop).

The fake conversation behaves like the real one (verified against the agent server): send_message during a run is
seen at the next step, interrupt() makes run() return with status "paused", run() again continues.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any

from app.core.exceptions import AnalysisError
from app.utils.AgentConversationUtils import ask
from app.utils.AgentControl import AgentControl


class FakeConversation:
    def __init__(self, steps: int = 3, step_seconds: float = 0.05) -> None:
        self.steps_left = steps
        self.step_seconds = step_seconds
        self.messages: list[str] = []
        self.runs = 0
        self.interrupted = threading.Event()
        self.state = SimpleNamespace(events=[], execution_status=SimpleNamespace(value="idle"))

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def interrupt(self) -> None:
        self.interrupted.set()

    def pause(self) -> None:
        self.interrupted.set()

    def run(self) -> None:
        self.runs += 1
        self.state.execution_status.value = "running"
        while self.steps_left > 0:
            if self.interrupted.is_set():
                self.interrupted.clear()
                self.state.execution_status.value = "paused"
                return
            time.sleep(self.step_seconds)
            self.steps_left -= 1
        self.state.execution_status.value = "finished"


def _failure(reason: str) -> AnalysisError:
    return AnalysisError(reason)


def _ask_in_thread(conversation: FakeConversation, control: AgentControl, deadline: float) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def run() -> None:
        result["answer"] = ask(conversation, "do the task", deadline, _failure, control)
        result["ended_at"] = time.monotonic()

    thread = threading.Thread(target=run)
    thread.start()
    result["thread"] = thread
    return result


def _wait(condition: Any, timeout: float = 5) -> None:
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.01)


def test_message_reaches_the_working_agent() -> None:
    conversation, control = FakeConversation(steps=20), AgentControl()
    result = _ask_in_thread(conversation, control, time.monotonic() + 30)
    _wait(lambda: control.is_working)

    assert control.deliver("use the existing LoginPage") is True
    result["thread"].join(5)

    assert conversation.messages == ["do the task", "use the existing LoginPage"]
    assert not control.is_working  # detached after the agent finished


def test_message_without_a_working_agent_waits_for_the_next_one() -> None:
    control = AgentControl()

    assert control.deliver("prefer data-test locators") is False
    conversation = FakeConversation(steps=1)
    ask(conversation, "write the tests", time.monotonic() + 30, _failure, control)

    assert conversation.messages == ["write the tests", "prefer data-test locators"]


def test_pause_stops_the_agent_until_resume_and_the_pause_does_not_eat_the_deadline() -> None:
    conversation, control = FakeConversation(steps=10), AgentControl()
    result = _ask_in_thread(conversation, control, time.monotonic() + 0.6)
    _wait(lambda: control.is_working)

    assert control.pause() is True
    assert control.is_paused and control.paused_since is not None
    time.sleep(0.8)  # longer than the agent's whole deadline
    assert result["thread"].is_alive() and conversation.steps_left > 0
    assert control.resume() is True
    result["thread"].join(5)

    assert conversation.steps_left == 0  # it finished its work after the resume
    assert conversation.runs >= 2
    assert 0.7 < control.paused_seconds < 5  # the healer's time budget leaves this out


def test_pause_without_a_working_agent_is_refused_and_resume_needs_a_pause() -> None:
    control = AgentControl()

    assert control.pause() is False
    assert control.resume() is False


def test_stop_releases_a_paused_agent() -> None:
    conversation, control = FakeConversation(steps=50), AgentControl()
    result = _ask_in_thread(conversation, control, time.monotonic() + 30)
    _wait(lambda: control.is_working)
    control.pause()
    _wait(lambda: conversation.state.execution_status.value == "paused")

    control.stop()
    result["thread"].join(5)

    assert not result["thread"].is_alive()
    assert conversation.steps_left > 0  # it did not go on working


def test_ask_without_control_works_as_before() -> None:
    conversation = FakeConversation(steps=2)

    ask(conversation, "task", time.monotonic() + 30, _failure)

    assert conversation.messages == ["task"] and conversation.runs == 1


class StatuslessConversation:
    """A conversation that reports no execution status (e.g. an older SDK or a test double)."""

    def __init__(self) -> None:
        self.runs = 0
        self.state = SimpleNamespace(events=[])

    def send_message(self, message: str) -> None:
        pass

    def run(self) -> None:
        self.runs += 1


def test_conversations_without_a_status_are_not_resumed() -> None:
    conversation = StatuslessConversation()

    ask(conversation, "task", time.monotonic() + 30, _failure, AgentControl())  # type: ignore[arg-type]

    assert conversation.runs == 1
