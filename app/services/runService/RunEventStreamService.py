"""Follow a run's timeline live: new events as they are saved, then the run's final state.

Events are written to MongoDB by the run (from any worker); a follower polls for events after the last seq it
sent, so a client that reconnects with that seq misses nothing and gets nothing twice.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from app.config import settings
from app.models.runModel import RunEventModel, RunModel, RunStatus

from .RunService import RunService

POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0
PAGE_SIZE = 200

_FINISHED = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


@dataclass(frozen=True)
class RunStreamItem:
    """One thing to send: an event, a keep-alive heartbeat, or the end of the run (with its final state)."""

    event: RunEventModel | None = None
    run: RunModel | None = None

    @property
    def is_heartbeat(self) -> bool:
        return self.event is None and self.run is None


class RunEventStreamService:
    def __init__(
        self,
        run_service: RunService | None = None,
        poll_seconds: float = POLL_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        max_stream_seconds: float = settings.RUN_TIMEOUT_SECONDS + 300,
    ) -> None:
        self.run_service = run_service or RunService()
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.max_stream_seconds = max_stream_seconds

    async def follow(
        self,
        run_id: str,
        org_id: str,
        after_seq: int,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[RunStreamItem]:
        """Yield events after `after_seq` until the run is finished and all its events were sent.

        The caller must have checked the run belongs to the org (so a 404 is a normal response, not a broken stream).
        """
        started = last_sent = time.monotonic()
        seq = after_seq
        while time.monotonic() - started < self.max_stream_seconds:
            if await is_disconnected():
                return
            # Read the status BEFORE the events: events saved after this read are fetched on the next round,
            # so the final page is never missed.
            run = await self.run_service.get(run_id, org_id)
            events = await self.run_service.get_events(run_id, org_id, after_seq=seq, limit=PAGE_SIZE)
            for event in events:
                seq = event.seq
                yield RunStreamItem(event=event)
            if events:
                last_sent = time.monotonic()
                if len(events) == PAGE_SIZE:
                    continue  # more are waiting: no sleep
            if run.status in _FINISHED and not events:
                yield RunStreamItem(run=run)
                return
            if time.monotonic() - last_sent >= self.heartbeat_seconds:
                last_sent = time.monotonic()
                yield RunStreamItem()
            await asyncio.sleep(self.poll_seconds)
