"""The user's handle on the agent that is working: messages, pause, resume, stop (thread-safe).

Kept apart from AgentConversationUtils so the API process can use it without importing the OpenHands SDK.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation

logger = logging.getLogger(__name__)

# How often a paused agent's thread checks whether the user resumed it.
PAUSE_POLL_SECONDS = 0.5


class AgentControl:
    """The user's handle on the agent that is working right now. Thread-safe.

    The agent's thread attaches its conversation while it works; the run's command poller (another thread or the event
    loop) delivers messages, pauses, resumes or stops it. A message that arrives while no agent works is kept and
    handed to the next agent that attaches.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversation: "BaseConversation | None" = None
        self._queued: list[str] = []
        self._resumed = threading.Event()
        self._resumed.set()
        self._paused_since: float | None = None
        self._paused_total = 0.0
        self._stopped = False

    @property
    def is_paused(self) -> bool:
        return not self._resumed.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def paused_since(self) -> float | None:
        """time.monotonic() when the agent was paused, or None."""
        return self._paused_since

    @property
    def is_working(self) -> bool:
        return self._conversation is not None

    def attach(self, conversation: "BaseConversation") -> None:
        with self._lock:
            self._conversation = conversation
            queued, self._queued = self._queued, []
        for text in queued:
            conversation.send_message(text)

    def detach(self, conversation: "BaseConversation") -> None:
        with self._lock:
            if self._conversation is conversation:
                self._conversation = None

    def deliver(self, text: str) -> bool:
        """Give the message to the working agent (True) or keep it for the next one (False)."""
        with self._lock:
            conversation = self._conversation
            if conversation is None:
                self._queued.append(text)
                return False
        conversation.send_message(text)  # reaches the agent at its next step
        return True

    def pause(self) -> bool:
        """Stop the working agent at once (its current model call is cancelled). False when no agent is working."""
        with self._lock:
            conversation = self._conversation
            if conversation is None or self.is_paused:
                return False
            self._resumed.clear()
            self._paused_since = time.monotonic()
        try:
            conversation.interrupt()
        except Exception:
            logger.warning("Could not interrupt the agent; pausing it after its current step", exc_info=True)
            conversation.pause()
        return True

    def resume(self) -> bool:
        """False when the agent was not paused."""
        with self._lock:
            if not self.is_paused:
                return False
            self._paused_since = None
            self._resumed.set()
        return True

    def stop(self) -> None:
        """The run is ending: a paused agent stops waiting."""
        self._stopped = True
        with self._lock:
            self._paused_since = None
            self._resumed.set()

    @property
    def paused_seconds(self) -> float:
        """All the time agents spent waiting in wait_while_paused (time limits do not count it)."""
        return self._paused_total

    def wait_while_paused(self) -> float:
        """Agent thread: block while the user has the agent paused; returns the seconds waited."""
        started = time.monotonic()
        while not self._resumed.wait(PAUSE_POLL_SECONDS):
            pass
        waited = time.monotonic() - started
        self._paused_total += waited
        return waited
