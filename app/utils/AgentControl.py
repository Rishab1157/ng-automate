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
# How the user's messages reach an agent: marked as the user's, so the agent treats them as instructions to follow,
# before its own default rules (but it never deletes or weakens an existing test).
USER_MESSAGE = (
    "Message from the user running this job. Follow it; where it differs from your task's rules, the user's message "
    "wins (but never delete or weaken an existing test): {text}"
)
EARLIER_INSTRUCTIONS = (
    "Earlier in this run the user gave these instructions. Follow them; where they differ from your task's rules, the "
    "user's instructions win (but never delete or weaken an existing test):\n{instructions}"
)


class AgentControl:
    """The user's handle on the agent that is working right now. Thread-safe.

    The agent's thread attaches its conversation while it works; the run's command poller (another thread or the event
    loop) delivers messages, pauses, resumes or stops it. A message that arrives while no agent works is kept and
    handed to the next agent that attaches. Messages hold for the whole run: a new agent conversation of the run (the
    next heal, the generator after the analyzer) first gets the instructions the user gave earlier.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversation: "BaseConversation | None" = None
        self._queued: list[str] = []
        self._delivered: list[str] = []  # the user's messages some agent of the run has received, in order
        self._informed: set[str] = set()  # conversations that have every delivered message
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

    @property
    def instructions(self) -> list[str]:
        """Every message the user sent during the run, in order."""
        with self._lock:
            return [*self._delivered, *self._queued]

    def attach(self, conversation: "BaseConversation") -> None:
        with self._lock:
            self._conversation = conversation
            key = _key(conversation)
            earlier = [] if key in self._informed else list(self._delivered)
            queued, self._queued = self._queued, []
            self._delivered.extend(queued)
            self._informed.add(key)
        if earlier:
            conversation.send_message(EARLIER_INSTRUCTIONS.format(instructions="\n".join(f"- {t}" for t in earlier)))
        for text in queued:
            conversation.send_message(USER_MESSAGE.format(text=text))

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
            self._delivered.append(text)
        conversation.send_message(USER_MESSAGE.format(text=text))  # reaches the agent at its next step
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


def _key(conversation: "BaseConversation") -> str:
    return str(getattr(conversation, "id", "") or id(conversation))
