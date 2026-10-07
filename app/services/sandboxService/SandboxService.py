"""STUB (interface only) — replaced by the sandbox build step."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from openhands.sdk.workspace import BaseWorkspace


@dataclass(frozen=True)
class SandboxSession:
    workspace: BaseWorkspace
    container_id: str


class SandboxService:
    def prepare_project_dir(self, project_id: str, run_id: str) -> Path:
        raise NotImplementedError

    @contextmanager
    def open_readonly(
        self, project_dir: Path, on_started: Callable[[str], None] | None = None
    ) -> Iterator[SandboxSession]:
        raise NotImplementedError
        yield  # pragma: no cover

    def stop_container(self, container_id: str) -> None:
        raise NotImplementedError

    def remove_run_dir(self, run_id: str) -> None:
        raise NotImplementedError
