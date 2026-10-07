"""Docker sandboxes for agent sessions: one container per session, the project mounted read-only.

The container runs the OpenHands agent server and the agent's tools act inside it. Each run works on
its own unzipped copy of the project archive: <DATA_DIR>/runs/<run_id>/project.
All methods block (unzipping, docker): call them from a worker thread.
"""

import logging
import os
import re
import secrets
import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhands.sdk.workspace import BaseWorkspace
from openhands.workspace import DockerWorkspace

from app.config import settings
from app.core.exceptions import ErrorMessages, NotFoundError, SandboxError
from app.services.projectService import ProjectStorageService
from app.utils.ArchiveUtils import extract_archive

logger = logging.getLogger(__name__)

SANDBOX_PROJECT_DIR = "/workspace/project"
# The agent server in the container only accepts requests carrying this key.
# DockerWorkspace forwards the variable into the container and sends it as the client key.
SESSION_KEY_ENV = "OH_SESSION_API_KEYS_0"
DOCKER_TIMEOUT_SECONDS = 60
# Not in ErrorMessages yet.
SANDBOX_STOP_FAILED = "The sandbox container could not be stopped"

# The SDK logs every docker command line it runs, and `docker run` carries the session key.
_SDK_COMMAND_LOGGER = "openhands.sdk.utils.command"
# Ids become folder names: one safe path segment only.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Docker container ids and names; a leading "-" would read as an option.
_CONTAINER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class SandboxSession:
    workspace: BaseWorkspace
    container_id: str


class _SandboxDockerWorkspace(DockerWorkspace):
    """A DockerWorkspace that stops its container when startup fails half-way.

    DockerWorkspace starts the container inside its constructor. If the health check then fails,
    the caller never gets the object, so it could not stop the container itself.
    """

    def model_post_init(self, context: Any) -> None:
        try:
            super().model_post_init(context)
        except BaseException:
            try:
                self.cleanup()
            except Exception:
                logger.warning("Could not stop a sandbox container that failed to start", exc_info=True)
            raise


class SandboxService:
    def __init__(
        self,
        image: str = settings.SANDBOX_IMAGE,
        cpus: float = settings.SANDBOX_CPUS,
        memory_mb: int = settings.SANDBOX_MEMORY_MB,
        startup_timeout_seconds: int = settings.SANDBOX_STARTUP_TIMEOUT_SECONDS,
        data_dir: Path = settings.DATA_DIR,
        workspace_factory: Callable[..., BaseWorkspace] = _SandboxDockerWorkspace,
    ) -> None:
        # Docker reads 0 as "no limit".
        if cpus <= 0 or memory_mb <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("Sandbox CPUs, memory and startup timeout must be positive")
        self.image = image
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.startup_timeout_seconds = startup_timeout_seconds
        self.workspace_factory = workspace_factory
        self.storage = ProjectStorageService(root=data_dir)
        self.runs_root = self.storage.root / "runs"
        _ensure_session_key()

    def prepare_project_dir(self, project_id: str, run_id: str) -> Path:
        """Unzip the project's archive into a fresh folder for this run; returns its absolute path."""
        if not _SAFE_ID_PATTERN.match(project_id):
            raise NotFoundError(ErrorMessages.PROJECT_NOT_FOUND)
        archive = self.storage.archive_path(project_id)
        if not archive.is_file():
            raise NotFoundError(ErrorMessages.PROJECT_NOT_FOUND)

        project_dir = self._run_dir(run_id) / "project"
        if project_dir.exists():
            shutil.rmtree(project_dir)
        try:
            extract_archive(archive, project_dir, self.storage.max_bytes)
        except BaseException:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        logger.info("Project %s unzipped for run %s", project_id, run_id)
        return project_dir

    @contextmanager
    def open_readonly(
        self, project_dir: Path, on_started: Callable[[str], None] | None = None
    ) -> Iterator[SandboxSession]:
        """A running sandbox with `project_dir` at /workspace/project (read-only), stopped on exit.

        `on_started(container_id)` runs once the container is up and limited, before the session is
        handed out, so the caller can record the container and stop it should this process die.
        """
        workspace = self._start_workspace(project_dir)
        container_id: str | None = getattr(workspace, "_container_id", None)  # DockerWorkspace keeps it private
        try:
            if not container_id:
                raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="the container id is unknown"))
            self._apply_limits(container_id)
            if on_started is not None:
                on_started(container_id)
            logger.info("Sandbox %s started", container_id[:12])
            yield SandboxSession(workspace=workspace, container_id=container_id)
        finally:
            self._dispose(workspace, container_id)

    def stop_container(self, container_id: str) -> None:
        """Stop a sandbox container (it is removed too: sandboxes run with --rm).

        A container that no longer exists is fine. Other failures raise SandboxError.
        """
        if not _CONTAINER_ID_PATTERN.match(container_id):
            raise ValueError(f"Not a container id: {container_id!r}")
        failure = self._docker(["stop", "--", container_id])
        if failure is None or "no such container" in failure.lower():
            return
        logger.error("Could not stop sandbox %s: %s", container_id, failure)
        raise SandboxError(SANDBOX_STOP_FAILED)

    def remove_run_dir(self, run_id: str) -> None:
        """Delete a run's working files. Never fails: leftovers only cost disk space."""
        if not _SAFE_ID_PATTERN.match(run_id):
            logger.warning("Not removing the folder of unexpected run id %r", run_id)
            return
        shutil.rmtree(self.runs_root / run_id, ignore_errors=True)

    def _run_dir(self, run_id: str) -> Path:
        if not _SAFE_ID_PATTERN.match(run_id):
            raise ValueError(f"Unexpected run id: {run_id!r}")
        return self.runs_root / run_id

    def _start_workspace(self, project_dir: Path) -> BaseWorkspace:
        host_dir = project_dir.resolve()
        if not host_dir.is_dir():
            # Docker would quietly create a missing bind-mount source as an empty folder.
            raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="the project folder is missing"))
        try:
            return self.workspace_factory(
                server_image=self.image,
                working_dir=SANDBOX_PROJECT_DIR,
                volumes=[f"{host_dir}:{SANDBOX_PROJECT_DIR}:ro"],
                detach_logs=False,
                health_check_timeout=self.startup_timeout_seconds,
            )
        except Exception as error:
            logger.error("Sandbox failed to start: %s: %s", type(error).__name__, _hide_session_key(str(error)))
            reason = _startup_failure_reason(error, self.startup_timeout_seconds)
            raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason=reason)) from None

    def _apply_limits(self, container_id: str) -> None:
        # DockerWorkspace has no resource options, so the limits are set right after the start.
        memory = f"{self.memory_mb}m"
        failure = self._docker(
            ["update", "--cpus", f"{self.cpus:g}", "--memory", memory, "--memory-swap", memory, "--", container_id]
        )
        if failure is not None:
            logger.error("Could not limit sandbox %s: %s", container_id, failure)
            raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="its CPU and memory limits could not be set"))

    def _dispose(self, workspace: BaseWorkspace, container_id: str | None) -> None:
        """Stop the container without hiding the error that ended the session, if any."""
        try:
            workspace.cleanup()
            return
        except Exception:
            logger.warning("Sandbox %s cleanup failed; stopping the container directly", container_id, exc_info=True)
        if container_id:
            try:
                self.stop_container(container_id)
            except Exception:
                logger.error("Sandbox %s may still be running", container_id, exc_info=True)

    def _docker(self, args: list[str]) -> str | None:
        """Run one docker command: None when it worked, else what went wrong (for the log only)."""
        try:
            result = self._run_docker(args)
        except subprocess.TimeoutExpired:
            return f"docker {args[0]} took longer than {DOCKER_TIMEOUT_SECONDS} seconds"
        except OSError as error:
            return f"docker could not be run: {error}"
        if result.returncode == 0:
            return None
        return result.stderr.strip() or f"docker {args[0]} exited with code {result.returncode}"

    def _run_docker(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        # The one place that runs docker directly (tests replace it).
        return subprocess.run(
            ["docker", *args], capture_output=True, encoding="utf-8", errors="replace",
            timeout=DOCKER_TIMEOUT_SECONDS, check=False,
        )


class _HideSessionKey(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        hidden = _hide_session_key(message)
        if hidden != message:
            record.msg, record.args = hidden, None
        return True


def _ensure_session_key() -> None:
    # An empty key would leave the agent server open to anything that can reach its port.
    if not os.environ.get(SESSION_KEY_ENV):
        os.environ[SESSION_KEY_ENV] = secrets.token_urlsafe(32)
    command_logger = logging.getLogger(_SDK_COMMAND_LOGGER)
    if not any(isinstance(existing, _HideSessionKey) for existing in command_logger.filters):
        command_logger.addFilter(_HideSessionKey())


def _hide_session_key(text: str) -> str:
    key = os.environ.get(SESSION_KEY_ENV)
    return text.replace(key, "***") if key else text


def _startup_failure_reason(error: Exception, timeout_seconds: int) -> str:
    """A short reason for callers: the raw error can hold docker output and server paths."""
    text = str(error).lower()
    if isinstance(error, FileNotFoundError) or "docker is not available" in text:
        return "Docker is not available"
    if "failed to become healthy" in text:
        return f"the agent server did not start within {timeout_seconds} seconds"
    if "stopped unexpectedly" in text:
        return "the container stopped right after starting"
    if "unable to find image" in text or "pull access denied" in text or "no such image" in text:
        return "the sandbox image is not available"
    if text.startswith("port ") and "not available" in text:
        return "no free local port"
    if "failed to run docker container" in text:
        return "Docker could not start the container"
    return "unexpected error"
