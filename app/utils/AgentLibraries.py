"""The agents' heavy libraries (OpenHands, LiteLLM, LangGraph, qdrant-client) load lazily, so the API starts fast.

Two threads importing them at the same moment can deadlock Python's import locks: the startup preload racing a run
that was resumed at startup raised `_DeadlockError` in `openhands.sdk.conversation`. So every load goes through
load_agent_libraries(): the first caller imports, every other caller waits for it, and afterwards each import is only
a dictionary lookup. Code on the event loop awaits agent_libraries_ready(), which never blocks the loop.
"""

import asyncio
import importlib
import logging
import threading

logger = logging.getLogger(__name__)

AGENT_MODULES = (
    "app.agents.MasterAgent.MasterGraph",
    "app.agents.MasterAgent.MasterNodes",
    "app.agents.TestGeneratorAgent.TestCaseExtractor",
    "app.services.sandboxService.SandboxService",
    # The healer's memory (Qdrant) is loaded with the rest, never on the event loop in the middle of a run.
    "qdrant_client",
    "qdrant_client.models",
    "qdrant_client.http.exceptions",
)

_lock = threading.Lock()
_loaded = threading.Event()


def load_agent_libraries() -> None:
    """Blocking: import the agents' libraries once. The first caller imports; the others wait until it is done."""
    if _loaded.is_set():
        return
    with _lock:
        if _loaded.is_set():
            return
        for name in AGENT_MODULES:
            importlib.import_module(name)
        _loaded.set()


async def agent_libraries_ready() -> None:
    """For the event loop: the libraries are loaded (by a worker thread, if nobody has done it yet)."""
    if not _loaded.is_set():
        await asyncio.to_thread(load_agent_libraries)


def agent_libraries_loaded() -> bool:
    return _loaded.is_set()
