"""AgentControl and ask(): the user steers the agent that is working (messages, pause, resume, stop).

The fake conversation behaves like the real one (verified against the agent server): send_message during a run is
seen at the next step, interrupt() makes run() return with status "paused", run() again continues.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.core.exceptions import AnalysisError
from app.models.llmModel import LlmConfigModel
from app.utils import AgentConversationUtils as conversation_utils
from app.utils.AgentConversationUtils import RUN_SLICE_SECONDS, EventForwarder, ask
from app.utils.AgentControl import EARLIER_INSTRUCTIONS, USER_MESSAGE, AgentControl


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

    assert conversation.messages == ["do the task", USER_MESSAGE.format(text="use the existing LoginPage")]
    assert not control.is_working  # detached after the agent finished


def test_message_without_a_working_agent_waits_for_the_next_one() -> None:
    control = AgentControl()

    assert control.deliver("prefer data-test locators") is False
    conversation = FakeConversation(steps=1)
    ask(conversation, "write the tests", time.monotonic() + 30, _failure, control)

    assert conversation.messages == ["write the tests", USER_MESSAGE.format(text="prefer data-test locators")]


class SilentRemote(RemoteConversation):
    """A remote conversation whose sandbox never answers (removed, or the model never replies): every wait times out."""

    def __init__(self) -> None:  # no super().__init__: nothing to connect to
        self.timeouts: list[float] = []
        self.posted_runs = 0
        self.fake_state = SimpleNamespace(events=[], execution_status=SimpleNamespace(value="running"))
        self._run_armed = threading.Event()

    @property
    def state(self) -> Any:  # type: ignore[override]
        return self.fake_state

    def send_message(self, message: Any, sender: str | None = None) -> None:
        pass

    def run(self, blocking: bool = True, poll_interval: float = 1.0, timeout: float = 3600.0) -> None:
        self.posted_runs += 1
        self._wait_for_run_completion(poll_interval, timeout)

    def _wait_for_run_completion(self, poll_interval: float = 1.0, timeout: float = 1800.0, since: int = 0) -> None:
        self.timeouts.append(timeout)
        time.sleep(0.01)
        raise ConversationRunError("c", TimeoutError(f"Run timed out after {timeout} seconds"))


def test_a_stopped_run_stops_waiting_for_an_agent_that_never_answers() -> None:
    # A real run: the user stopped it, its sandbox was removed, and the old wait polled the dead sandbox ("connection
    # refused" every 3 s) until its deadline, up to an hour. Now the wait ends with the slice it is in.
    control, remote = AgentControl(), SilentRemote()
    control.stop()

    with pytest.raises(AnalysisError):
        ask(remote, "heal it", time.monotonic() + 3600, _failure, control)

    assert remote.timeouts == [RUN_SLICE_SECONDS]


def test_an_agent_without_progress_is_reported_in_the_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversation_utils, "NO_PROGRESS_NOTICE_SECONDS", 0.0)
    events: list[AgentEvent] = []
    remote = SilentRemote()

    with pytest.raises(AnalysisError):
        ask(remote, "heal it", time.monotonic() + 0.2, _failure, None, EventForwarder(events.append, ()))

    assert len(remote.timeouts) > 1  # it kept waiting until the deadline
    # The run was started once: posting /run again blocks behind a long model call until the request times out.
    assert remote.posted_runs == 1
    assert events and events[0].message.startswith("No new step for 0 min: the agent is waiting for the model's answer")


def test_a_new_agent_of_the_run_gets_the_instructions_given_earlier() -> None:
    # Each heal is a new conversation: what the user told the first one must still hold for the next one.
    control = AgentControl()
    control.deliver("use Cucumber BDD")  # no agent works yet: queued
    first = FakeConversation(steps=1)
    ask(first, "heal 1", time.monotonic() + 30, _failure, control)
    second = FakeConversation(steps=1)
    ask(second, "heal 2", time.monotonic() + 30, _failure, control)
    ask(second, "nudge", time.monotonic() + 30, _failure, control)  # the same conversation is not told twice

    assert first.messages == ["heal 1", USER_MESSAGE.format(text="use Cucumber BDD")]
    assert second.messages == ["heal 2", EARLIER_INSTRUCTIONS.format(instructions="- use Cucumber BDD"), "nudge"]
    assert control.instructions == ["use Cucumber BDD"]


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


# ---------- before an agent starts: can it use its model? ----------

class _Sandbox:
    """A sandbox whose curl to the model server ends with `exit_code`."""

    def __init__(self, exit_code: int, stderr: str = "") -> None:
        self.exit_code, self.stderr = exit_code, stderr

    def execute_command(self, command: str, cwd: str | None = None, timeout: float = 30.0) -> Any:
        return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr=self.stderr, timeout_occurred=False)


def test_an_agent_starts_only_when_its_sandbox_reaches_the_model_and_the_model_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LlmConfigModel(model="ollama_chat/qwen2.5-coder:14b", base_url="http://192.168.3.90:11435")
    asked: list[LlmConfigModel] = []
    monkeypatch.setattr(conversation_utils, "check_model_answers", lambda llm_config: asked.append(llm_config))
    refused = "curl: (7) Failed to connect to 192.168.3.90 port 11435"

    assert conversation_utils.check_model(_Sandbox(0), config) is None
    assert conversation_utils.check_model(_Sandbox(7, refused), config) == (
        conversation_utils.MODEL_UNREACHABLE_FROM_SANDBOX.format(reason=refused)
    )
    assert asked == [config]  # a server the sandbox cannot reach is not asked anything more

    monkeypatch.setattr(conversation_utils, "check_model_answers", lambda llm_config: "it did not answer")
    assert conversation_utils.check_model(_Sandbox(0), config) == "it did not answer"
