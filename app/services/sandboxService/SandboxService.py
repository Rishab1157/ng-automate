"""Docker sandboxes for agent sessions: one container per session.

The container runs the OpenHands agent server and the agent's tools act inside it. Each run works on
its own unzipped copy of the project archive: <DATA_DIR>/runs/<run_id>/project.
All methods block (unzipping, docker): call them from a worker thread.

Sandboxes run untrusted project code (builds, tests), so they are isolated from each other twice over: every
container gets its own random session key for its agent server, and all of them sit on a network where
inter-container traffic is turned off.

Test sandboxes also have a live view: the browsers the tests drive show on a virtual display, and websockify in
the container (published on the host's 127.0.0.1 only) lets a viewer in with a token the API adds per viewer.
"""

import json
import logging
import re
import secrets
import shutil
import subprocess
import threading
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
DOCKER_TIMEOUT_SECONDS = 60
# Docker bridge option that, set to "false", stops containers on the network from reaching each other.
NETWORK_ICC_OPTION = "com.docker.network.bridge.enable_icc"
NETWORK_LABEL = "ngauto.sandboxes=1"
# Build caches inside the test sandbox (docker/sandbox-test/Dockerfile creates these folders).
BUILD_CACHE_PATHS = {
    "maven": "/home/openhands/.m2",
    "npm": "/home/openhands/.npm",
    "pip": "/home/openhands/.cache/pip",
}
# Not in ErrorMessages yet.
SANDBOX_STOP_FAILED = "The sandbox container could not be stopped"
# Test sandboxes: browsers need more shared memory than Docker's default 64 MB (Chrome tabs crash without it).
TEST_SANDBOX_SHM_SIZE = "1g"
# Startup attempts when the free port DockerWorkspace picked, or the one next to it, is taken in the meantime.
STARTUP_PORT_ATTEMPTS = 3

# Live view (docker/sandbox-test/ngauto-entrypoint.sh). websockify listens on this container port; DockerWorkspace
# publishes it on the host's 127.0.0.1, next to the agent server's port. Each token leads to one VNC server.
LIVE_VIEW_CONTAINER_PORT = "8001/tcp"
LIVE_VIEW_VNC_PORT = 5900  # input allowed
LIVE_VIEW_VIEW_ONLY_VNC_PORT = 5901  # input ignored
_LIVE_VIEW_DIR = "/tmp/ngauto-live-view"
# Appends stdin (the token line) to websockify's token file, once the display is up. The token is never part of a
# command line: command lines show in `ps` inside the container and in docker's own logs.
_LIVE_VIEW_NOT_READY_EXIT_CODE = 3

_ADD_LIVE_VIEW_TOKEN_SCRIPT = (
    f"umask 077; "
    f"[ -f {_LIVE_VIEW_DIR}/ready ] || exit {_LIVE_VIEW_NOT_READY_EXIT_CODE}; "
    f"cat >> {_LIVE_VIEW_DIR}/tokens"
)

# The SDK logs every docker command line it runs, and `docker run` carries the session key.
_SDK_COMMAND_LOGGER = "openhands.sdk.utils.command"
# A session key in a logged command line, whatever its value.
_SESSION_KEY_IN_TEXT = re.compile(r"((?:OH_SESSION_API_KEYS_\d+|SESSION_API_KEY)=)\S+")
# Keys of the sandboxes running now, masked wherever they could appear in a log line.
_ACTIVE_SESSION_KEYS: set[str] = set()
_ACTIVE_SESSION_KEYS_LOCK = threading.Lock()
# Ids become folder names: one safe path segment only.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Docker container ids and names; a leading "-" would read as an option.
_CONTAINER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class SandboxSession:
    workspace: BaseWorkspace
    container_id: str


class LiveViewError(Exception):
    """The sandbox has no working live view: not a test sandbox, its display is not up, or it has stopped."""


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
        test_image: str = settings.SANDBOX_TEST_IMAGE,
        build_caches: bool = settings.SANDBOX_BUILD_CACHES,
        cpus: float = settings.SANDBOX_CPUS,
        memory_mb: int = settings.SANDBOX_MEMORY_MB,
        startup_timeout_seconds: int = settings.SANDBOX_STARTUP_TIMEOUT_SECONDS,
        data_dir: Path = settings.DATA_DIR,
        workspace_factory: Callable[..., BaseWorkspace] = _SandboxDockerWorkspace,
        network: str | None = settings.SANDBOX_NETWORK,
    ) -> None:
        # Docker reads 0 as "no limit".
        if cpus <= 0 or memory_mb <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("Sandbox CPUs, memory and startup timeout must be positive")
        self.image = image
        self.test_image = test_image
        self.build_caches = build_caches
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.startup_timeout_seconds = startup_timeout_seconds
        self.workspace_factory = workspace_factory
        # None only in tests that do not run docker: production sandboxes always get the isolated network.
        self.network = network
        self.storage = ProjectStorageService(root=data_dir)
        self.runs_root = self.storage.root / "runs"
        self._network_ready = False
        self._network_lock = threading.Lock()
        _install_log_filter()

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
        with self._open(project_dir, on_started, image=self.image, volumes=[]) as session:
            yield session

    @contextmanager
    def open_writable(
        self, project_dir: Path, org_id: str, on_started: Callable[[str], None] | None = None
    ) -> Iterator[SandboxSession]:
        """A running test sandbox (build tools, browser) where `project_dir` is writable, stopped on exit.

        Only for a run's own copy of the project: changes land in `project_dir` on the host. Build caches are
        per organization, so one org's project can never plant packages another org's build would use.
        """
        if not _SAFE_ID_PATTERN.match(org_id):
            raise ValueError(f"Unexpected org id: {org_id!r}")
        caches = (
            [f"ngauto-{name}-{org_id}:{path}" for name, path in BUILD_CACHE_PATHS.items()] if self.build_caches else []
        )
        with self._open(project_dir, on_started, image=self.test_image, volumes=caches, writable=True) as session:
            yield session

    @contextmanager
    def _open(
        self,
        project_dir: Path,
        on_started: Callable[[str], None] | None,
        *,
        image: str,
        volumes: list[str],
        writable: bool = False,
    ) -> Iterator[SandboxSession]:
        session_key = secrets.token_urlsafe(32)
        with _ACTIVE_SESSION_KEYS_LOCK:
            _ACTIVE_SESSION_KEYS.add(session_key)
        try:
            workspace = self._start_workspace(project_dir, image, volumes, writable, session_key)
        except BaseException:
            _forget_session_key(session_key)
            raise
        container_id: str | None = getattr(workspace, "_container_id", None)  # DockerWorkspace keeps it private
        try:
            if not container_id:
                raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="the container id is unknown"))
            self._apply_limits(container_id)
            if on_started is not None:
                on_started(container_id)
            logger.info("Sandbox %s started (%s)", container_id[:12], "writable" if writable else "read-only")
            yield SandboxSession(workspace=workspace, container_id=container_id)
        finally:
            self._dispose(workspace, container_id)
            _forget_session_key(session_key)

    def stop_container(self, container_id: str) -> None:
        """Stop a sandbox container (it is removed too: sandboxes run with --rm).

        A container that no longer exists is fine. Other failures raise SandboxError.
        """
        _check_container_id(container_id)
        failure = self._docker(["stop", "--", container_id])
        if failure is None or "no such container" in failure.lower():
            return
        logger.error("Could not stop sandbox %s: %s", container_id, failure)
        raise SandboxError(SANDBOX_STOP_FAILED)

    def add_live_view_token(self, container_id: str, interactive: bool) -> str:
        """Let one more viewer into a running test sandbox's display; returns the token websockify will ask for.

        `interactive` viewers may use the mouse and keyboard; the others only watch. The token reaches the
        container on stdin, never on a command line. Raises LiveViewError when the sandbox has no live view.
        """
        _check_container_id(container_id)
        token = secrets.token_urlsafe(32)
        vnc_port = LIVE_VIEW_VNC_PORT if interactive else LIVE_VIEW_VIEW_ONLY_VNC_PORT
        args = ["exec", "-i", "-u", "openhands", "--", container_id, "sh", "-c", _ADD_LIVE_VIEW_TOKEN_SCRIPT]
        try:
            result = self._run_docker(args, input=f"{token}: 127.0.0.1:{vnc_port}\n")
        except subprocess.TimeoutExpired:
            raise LiveViewError(f"docker exec took longer than {DOCKER_TIMEOUT_SECONDS} seconds") from None
        except OSError as error:
            raise LiveViewError(f"docker could not be run: {error}") from None
        if result.returncode == _LIVE_VIEW_NOT_READY_EXIT_CODE:
            raise LiveViewError("the sandbox's display is not running")
        if result.returncode != 0:
            raise LiveViewError(result.stderr.strip() or f"docker exec exited with code {result.returncode}")
        logger.info(
            "Live view viewer added to sandbox %s (%s)", container_id[:12], "interactive" if interactive else "view-only"
        )
        return token

    def live_view_port(self, container_id: str) -> int | None:
        """The host port (on 127.0.0.1) of a sandbox's live view, or None when it has none or is not running."""
        _check_container_id(container_id)
        try:
            result = self._run_docker(["port", "--", container_id, LIVE_VIEW_CONTAINER_PORT])
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        # One "<address>:<port>" line per published address, e.g. "127.0.0.1:30125".
        for line in result.stdout.splitlines():
            port = line.strip().rpartition(":")[2]
            if port.isdigit() and 0 < int(port) < 65536:
                return int(port)
        return None

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

    def _start_workspace(
        self, project_dir: Path, image: str, volumes: list[str], writable: bool, session_key: str
    ) -> BaseWorkspace:
        host_dir = project_dir.resolve()
        if not host_dir.is_dir():
            # Docker would quietly create a missing bind-mount source as an empty folder.
            raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="the project folder is missing"))
        if self.network is not None:
            self._ensure_network()
        options: dict[str, Any] = {}
        if writable:
            # Test sandboxes: the live view's port is published too (on the host's 127.0.0.1, agent server port + 1),
            # and browsers get enough shared memory.
            options = {"extra_ports": True, "extra_run_args": [f"--shm-size={TEST_SANDBOX_SHM_SIZE}"]}
        attempt = 1
        while True:
            try:
                return self.workspace_factory(
                    server_image=image,
                    working_dir=SANDBOX_PROJECT_DIR,
                    volumes=[f"{host_dir}:{SANDBOX_PROJECT_DIR}:{'rw' if writable else 'ro'}", *volumes],
                    detach_logs=False,
                    health_check_timeout=self.startup_timeout_seconds,
                    session_api_key=session_key,
                    network=self.network,
                    **options,
                )
            except Exception as error:
                # DockerWorkspace checks its ports before it starts anything, so trying again with new ports is safe.
                if _is_port_taken(error) and attempt < STARTUP_PORT_ATTEMPTS:
                    logger.info("Sandbox port was taken; starting again with other ports")
                    attempt += 1
                    continue
                logger.error("Sandbox failed to start: %s: %s", type(error).__name__, _hide_session_key(str(error)))
                reason = _startup_failure_reason(error, self.startup_timeout_seconds)
                raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason=reason)) from None

    def _ensure_network(self) -> None:
        """Create the sandbox network once; refuse an existing one that lets containers reach each other."""
        with self._network_lock:
            if self._network_ready:
                return
            name = self.network
            assert name is not None
            options = self._network_options(name)
            if options is None:
                failure = self._docker([
                    "network", "create", "--driver", "bridge", "--opt", f"{NETWORK_ICC_OPTION}=false",
                    "--label", NETWORK_LABEL, "--", name,
                ])
                # "already exists": another process created it in the meantime; it is checked below like any other.
                if failure is not None and "already exists" not in failure:
                    logger.error("Could not create docker network %s: %s", name, failure)
                    raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="its network could not be created"))
                options = self._network_options(name) or {}
            if options.get(NETWORK_ICC_OPTION) != "false":
                logger.error("Docker network %s lets containers reach each other: refusing to use it", name)
                raise SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="its network is not isolated"))
            self._network_ready = True

    def _network_options(self, name: str) -> dict[str, Any] | None:
        """The network's driver options, or None when it does not exist."""
        try:
            result = self._run_docker(["network", "inspect", "--format", "{{json .Options}}", "--", name])
        except (subprocess.TimeoutExpired, OSError):
            return None
        return _json_object(result.stdout) if result.returncode == 0 else None

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

    def _run_docker(self, args: list[str], input: str | None = None) -> subprocess.CompletedProcess[str]:
        # The one place that runs docker directly (tests replace it). `input` goes to docker's stdin.
        return subprocess.run(
            ["docker", *args], input=input, capture_output=True, encoding="utf-8", errors="replace",
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


def _install_log_filter() -> None:
    command_logger = logging.getLogger(_SDK_COMMAND_LOGGER)
    if not any(isinstance(existing, _HideSessionKey) for existing in command_logger.filters):
        command_logger.addFilter(_HideSessionKey())


def _forget_session_key(key: str) -> None:
    with _ACTIVE_SESSION_KEYS_LOCK:
        _ACTIVE_SESSION_KEYS.discard(key)


def _hide_session_key(text: str) -> str:
    text = _SESSION_KEY_IN_TEXT.sub(r"\g<1>***", text)
    with _ACTIVE_SESSION_KEYS_LOCK:
        keys = list(_ACTIVE_SESSION_KEYS)
    for key in keys:
        text = text.replace(key, "***")
    return text


def _check_container_id(container_id: str) -> None:
    if not _CONTAINER_ID_PATTERN.match(container_id):
        raise ValueError(f"Not a container id: {container_id!r}")


def _is_port_taken(error: Exception) -> bool:
    """DockerWorkspace's "Port <n> is not available" (the agent server's port or the live view's next to it)."""
    text = str(error).lower()
    return text.startswith("port ") and "not available" in text


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "null")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


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
    if _is_port_taken(error):
        return "no free local port"
    if "failed to run docker container" in text:
        return "Docker could not start the container"
    return "unexpected error"
