"""Applies the commands users send to a running run (message, pause, resume, stop).

The API stores commands in MongoDB; while the run executes, its RunControl polls the pending ones and applies them to
the agent that is working (through AgentControl) or to the run itself (stop). Every outcome goes to the timeline.
While the agent is paused, the run's timeout is pushed back, up to MAX_PAUSE_SECONDS; then the run is stopped.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.config import settings
from app.core.exceptions import ErrorMessages
from app.models.runCommandModel import RunCommandModel, RunCommandStatus, RunCommandType
from app.utils.AgentControl import AgentControl

if TYPE_CHECKING:
    from app.services.runCommandService import RunCommandService

logger = logging.getLogger(__name__)

DELIVERED = "Your message was given to the agent"
QUEUED = "No agent is working right now: your message goes to the next agent"
PAUSED = "Paused: the agent waits until you resume or stop the run"
NOTHING_TO_PAUSE = "Nothing to pause: no agent is working right now (tests may be running)"
RESUMED = "Resumed: the agent goes on"
NOT_PAUSED = "The agent is not paused"
STOPPING = "Stopping the run"

# Runs executing in this process, so the master's stages find the control of their run.
_CONTROLS: dict[str, "RunControl"] = {}


def agent_control(run_id: str) -> AgentControl | None:
    """The working-agent handle of a run executing here, or None (e.g. in tests, or no RunControl)."""
    control = _CONTROLS.get(run_id)
    return control.agent if control is not None else None


class RunControl:
    def __init__(
        self,
        run_id: str,
        command_service: "RunCommandService",
        on_stop: Callable[[str], None],
        extend_deadline: Callable[[float], None],
        poll_seconds: float = settings.COMMAND_POLL_SECONDS,
        max_pause_seconds: float = settings.MAX_PAUSE_SECONDS,
    ) -> None:
        self.run_id = run_id
        self.command_service = command_service
        self.on_stop = on_stop
        self.extend_deadline = extend_deadline
        self.poll_seconds = poll_seconds
        self.max_pause_seconds = max_pause_seconds
        self.agent = AgentControl()
        self.stop_reason: str | None = None

    def __enter__(self) -> "RunControl":
        _CONTROLS[self.run_id] = self
        return self

    def __exit__(self, *_: object) -> None:
        if _CONTROLS.get(self.run_id) is self:
            del _CONTROLS[self.run_id]
        self.agent.stop()  # a paused agent thread must not wait forever

    async def poll(self) -> None:
        """Until cancelled: apply pending commands; keep the run's clock still while the agent is paused."""
        while True:
            try:
                for command in await self.command_service.get_pending(self.run_id):
                    await self.apply(command)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Could not read or apply the commands of run %s", self.run_id, exc_info=True)
            if self.agent.is_paused:
                self.extend_deadline(self.poll_seconds)
                since = self.agent.paused_since
                if since is not None and time.monotonic() - since > self.max_pause_seconds:
                    self.stop(ErrorMessages.RUN_PAUSED_TOO_LONG.format(minutes=int(self.max_pause_seconds // 60)))
            await asyncio.sleep(self.poll_seconds)

    async def apply(self, command: RunCommandModel) -> None:
        if command.type == RunCommandType.MESSAGE:
            delivered = await asyncio.to_thread(self.agent.deliver, command.text or "")
            await self.command_service.mark(command, RunCommandStatus.APPLIED, DELIVERED if delivered else QUEUED)
        elif command.type == RunCommandType.PAUSE:
            paused = await asyncio.to_thread(self.agent.pause)
            await self._mark(command, paused, PAUSED, NOTHING_TO_PAUSE)
        elif command.type == RunCommandType.RESUME:
            await self._mark(command, self.agent.resume(), RESUMED, NOT_PAUSED)
        else:
            await self.command_service.mark(command, RunCommandStatus.APPLIED, STOPPING)
            self.stop(ErrorMessages.RUN_STOPPED_BY_USER)

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.agent.stop()
            self.on_stop(reason)

    async def _mark(self, command: RunCommandModel, done: bool, applied: str, rejected: str) -> None:
        if done:
            await self.command_service.mark(command, RunCommandStatus.APPLIED, applied)
        else:
            await self.command_service.mark(command, RunCommandStatus.REJECTED, rejected)
