"""The agents' libraries load through one guarded loader: never two imports at once (that can deadlock Python's
import locks), and never on the event loop."""

import threading
import time
from dataclasses import dataclass, field

import pytest

from app.utils import AgentLibraries


@dataclass
class ImportRecorder:
    """Stands in for importlib.import_module: slow, and it records order, threads and overlaps."""

    imported: list[str] = field(default_factory=list)
    threads: list[int] = field(default_factory=list)
    most_at_once: int = 0
    _now: int = 0
    _guard: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, name: str) -> None:
        with self._guard:
            self._now += 1
            self.most_at_once = max(self.most_at_once, self._now)
        time.sleep(0.02)
        self.imported.append(name)
        self.threads.append(threading.get_ident())
        with self._guard:
            self._now -= 1


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> ImportRecorder:
    """A loader that has loaded nothing yet, importing fake modules."""
    recorder = ImportRecorder()
    monkeypatch.setattr(AgentLibraries, "_loaded", threading.Event())
    monkeypatch.setattr(AgentLibraries, "_lock", threading.Lock())
    monkeypatch.setattr(AgentLibraries, "AGENT_MODULES", ("fake.one", "fake.two", "fake.three"))
    monkeypatch.setattr(AgentLibraries.importlib, "import_module", recorder)
    return recorder


def test_many_threads_load_once_and_never_at_the_same_time(recorder: ImportRecorder) -> None:
    threads = [threading.Thread(target=AgentLibraries.load_agent_libraries) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert recorder.imported == ["fake.one", "fake.two", "fake.three"]  # once, in order
    assert recorder.most_at_once == 1  # one import at a time
    assert AgentLibraries.agent_libraries_loaded()


@pytest.mark.anyio
async def test_the_event_loop_waits_without_importing_on_it(recorder: ImportRecorder) -> None:
    await AgentLibraries.agent_libraries_ready()

    assert recorder.imported == ["fake.one", "fake.two", "fake.three"]
    assert threading.get_ident() not in recorder.threads  # a worker thread imported them


def test_a_failed_load_is_tried_again_next_time(recorder: ImportRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(AgentLibraries.importlib, "import_module", broken)
    with pytest.raises(ImportError):
        AgentLibraries.load_agent_libraries()
    assert not AgentLibraries.agent_libraries_loaded()

    monkeypatch.setattr(AgentLibraries.importlib, "import_module", recorder)
    AgentLibraries.load_agent_libraries()
    assert AgentLibraries.agent_libraries_loaded()
