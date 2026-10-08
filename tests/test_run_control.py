"""RunControl: applies user commands to the run and its working agent; the executor turns a stop into "cancelled"."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from app.agents.MasterAgent.RunControl import (
    DELIVERED,
    NOT_PAUSED,
    NOTHING_TO_PAUSE,
    PAUSED,
    QUEUED,
    RESUMED,
    STOPPING,
    RunControl,
    agent_control,
)
from app.core.exceptions import ErrorMessages
from app.models.runCommandModel import RunCommandModel, RunCommandStatus, RunCommandType
from app.models.runModel import RunEventType, RunStatus
from tests.test_run_executor import FakeCommands, FakeGraph, FakeRunService, make_executor, make_run, settle


def _command(type: RunCommandType, text: str | None = None) -> RunCommandModel:
    return RunCommandModel(
        id=f"c-{type.value}", run_id="r-1", org_id="o-1", created_by="u-1", type=type, text=text,
        status=RunCommandStatus.PENDING, created_at=datetime.now(UTC),
    )


class FakeConversation:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.interrupts = 0

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def interrupt(self) -> None:
        self.interrupts += 1


def _control(commands: FakeCommands | None = None, **options: Any) -> tuple[RunControl, FakeCommands, list[str], list[float]]:
    commands = commands or FakeCommands()
    stops: list[str] = []
    extensions: list[float] = []
    control = RunControl("r-1", commands, stops.append, extensions.append, **options)  # type: ignore[arg-type]
    return control, commands, stops, extensions


@pytest.mark.anyio
async def test_message_goes_to_the_working_agent_or_waits_for_the_next() -> None:
    control, commands, _, _ = _control()
    conversation = FakeConversation()

    await control.apply(_command(RunCommandType.MESSAGE, "too early"))
    control.agent.attach(conversation)  # the next agent starts: it gets the queued message
    await control.apply(_command(RunCommandType.MESSAGE, "use LoginPage"))

    assert conversation.messages == ["too early", "use LoginPage"]
    assert [(m[1], m[2]) for m in commands.marked] == [(RunCommandStatus.APPLIED, QUEUED), (RunCommandStatus.APPLIED, DELIVERED)]


@pytest.mark.anyio
async def test_pause_and_resume() -> None:
    control, commands, _, _ = _control()
    conversation = FakeConversation()

    await control.apply(_command(RunCommandType.PAUSE))  # no agent working yet
    control.agent.attach(conversation)
    await control.apply(_command(RunCommandType.RESUME))  # not paused
    await control.apply(_command(RunCommandType.PAUSE))
    await control.apply(_command(RunCommandType.RESUME))

    assert [(m[1], m[2]) for m in commands.marked] == [
        (RunCommandStatus.REJECTED, NOTHING_TO_PAUSE),
        (RunCommandStatus.REJECTED, NOT_PAUSED),
        (RunCommandStatus.APPLIED, PAUSED),
        (RunCommandStatus.APPLIED, RESUMED),
    ]
    assert conversation.interrupts == 1


@pytest.mark.anyio
async def test_stop_ends_the_run_once() -> None:
    control, commands, stops, _ = _control()

    await control.apply(_command(RunCommandType.STOP))
    await control.apply(_command(RunCommandType.STOP))

    assert stops == [ErrorMessages.RUN_STOPPED_BY_USER]
    assert control.stop_reason == ErrorMessages.RUN_STOPPED_BY_USER
    assert commands.marked[0][2] == STOPPING


@pytest.mark.anyio
async def test_paused_time_is_given_back_to_the_run_and_a_long_pause_stops_it() -> None:
    control, commands, stops, extensions = _control(poll_seconds=0.01, max_pause_seconds=0.05)
    control.agent.attach(FakeConversation())
    control.agent.pause()

    poller = asyncio.create_task(control.poll())
    await asyncio.sleep(0.2)
    poller.cancel()
    await asyncio.gather(poller, return_exceptions=True)

    assert extensions and all(seconds == 0.01 for seconds in extensions)
    assert stops == [ErrorMessages.RUN_PAUSED_TOO_LONG.format(minutes=0)]


def test_agents_find_the_control_of_their_run_while_it_executes() -> None:
    control, *_ = _control()

    with control:
        assert agent_control("r-1") is control.agent
    assert agent_control("r-1") is None
    assert control.agent.is_stopped  # a paused agent thread is released when the run ends


@pytest.mark.anyio
async def test_stop_command_cancels_the_run() -> None:
    run = make_run()
    runs = FakeRunService(run)
    commands = FakeCommands()
    started = asyncio.Event()

    async def long_graph(state: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(30)
        return state

    executor, sandbox = make_executor(runs, FakeGraph(long_graph), command_service=commands)
    await executor.submit(run.id)
    await asyncio.wait_for(started.wait(), 5)
    commands.pending.append(_command(RunCommandType.STOP).model_copy(update={"run_id": run.id}))
    await settle(executor)

    done = runs.runs[run.id]
    assert done.status == RunStatus.CANCELLED
    assert done.error is not None and done.error.message == ErrorMessages.RUN_STOPPED_BY_USER
    assert [event[1] for event in runs.events][-1] == RunEventType.RUN_CANCELLED
    assert sandbox.removed == [run.id]  # a finished run's folder is cleaned up
