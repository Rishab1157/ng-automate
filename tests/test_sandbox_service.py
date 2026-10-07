import logging
import os
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from bson import ObjectId

from app.core.exceptions import ErrorCode, ErrorMessages, NotFoundError, SandboxError, ValidationError
from app.services.sandboxService import SandboxService, SandboxSession
from app.services.sandboxService.SandboxService import (
    SANDBOX_PROJECT_DIR,
    SESSION_KEY_ENV,
    _SandboxDockerWorkspace,
)
from openhands.workspace import DockerWorkspace
from tests.conftest import make_zip

PROJECT_ID = str(ObjectId())
RUN_ID = str(ObjectId())
CONTAINER_ID = "0123456789abcdef" * 4
SAMPLE_FILES = {"pom.xml": "<project/>", "src/test/java/LoginTest.java": "class LoginTest {}"}

docker_tests = pytest.mark.skipif(
    os.environ.get("NGAUTOMATE_DOCKER_TESTS") != "1", reason="set NGAUTOMATE_DOCKER_TESTS=1 to start real containers"
)


class FakeWorkspace:
    """Stands in for DockerWorkspace: the container id is private there too."""

    def __init__(self, container_id: str | None, cleanup_error: Exception | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._container_id = container_id
        self.cleanup_error = cleanup_error
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_error:
            raise self.cleanup_error
        self._container_id = None


class FakeWorkspaceFactory:
    def __init__(
        self,
        error: Exception | None = None,
        container_id: str | None = CONTAINER_ID,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.error = error
        self.container_id = container_id
        self.cleanup_error = cleanup_error
        self.calls: list[dict[str, Any]] = []
        self.workspaces: list[FakeWorkspace] = []

    def __call__(self, **kwargs: Any) -> FakeWorkspace:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        workspace = FakeWorkspace(self.container_id, self.cleanup_error, **kwargs)
        self.workspaces.append(workspace)
        return workspace

    @property
    def workspace(self) -> FakeWorkspace:
        assert len(self.workspaces) == 1
        return self.workspaces[0]


class FakeDocker:
    """Records docker commands and answers them in order (success once the answers run out)."""

    def __init__(self, *answers: subprocess.CompletedProcess[str] | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        answer = self.answers.pop(0) if self.answers else _completed()
        if isinstance(answer, Exception):
            raise answer
        return answer


def _completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["docker"], returncode=returncode, stdout="", stderr=stderr)


def _service(tmp_path: Path, factory: Callable[..., Any] | None = None) -> SandboxService:
    return SandboxService(
        image="test-image:1",
        cpus=1.5,
        memory_mb=512,
        startup_timeout_seconds=30,
        data_dir=tmp_path / "data",
        workspace_factory=factory or FakeWorkspaceFactory(),
    )


def _store_archive(service: SandboxService, data: bytes, project_id: str = PROJECT_ID) -> None:
    path = service.storage.archive_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _project_dir(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pom.xml").write_text("<project/>")
    return project_dir


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): path.read_text() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def factory() -> FakeWorkspaceFactory:
    return FakeWorkspaceFactory()


@pytest.fixture
def docker() -> FakeDocker:
    return FakeDocker()


@pytest.fixture
def service(tmp_path: Path, factory: FakeWorkspaceFactory, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch) -> SandboxService:
    service = _service(tmp_path, factory)
    monkeypatch.setattr(service, "_run_docker", docker)
    return service


# ---------- prepare_project_dir ----------

def test_prepare_unzips_the_project_into_the_run_folder(service: SandboxService, tmp_path: Path) -> None:
    _store_archive(service, make_zip(SAMPLE_FILES))

    project_dir = service.prepare_project_dir(PROJECT_ID, RUN_ID)

    assert project_dir == (tmp_path / "data").resolve() / "runs" / RUN_ID / "project"
    assert project_dir.is_absolute()
    assert _files(project_dir) == SAMPLE_FILES


def test_prepare_replaces_an_existing_copy(service: SandboxService) -> None:
    _store_archive(service, make_zip(SAMPLE_FILES))
    project_dir = service.prepare_project_dir(PROJECT_ID, RUN_ID)
    (project_dir / "stale.txt").write_text("left over from an earlier attempt")
    (project_dir / "pom.xml").write_text("changed")

    again = service.prepare_project_dir(PROJECT_ID, RUN_ID)

    assert again == project_dir
    assert _files(project_dir) == SAMPLE_FILES


def test_prepare_without_an_archive_is_not_found(service: SandboxService) -> None:
    with pytest.raises(NotFoundError) as error:
        service.prepare_project_dir(PROJECT_ID, RUN_ID)

    assert error.value.status_code == 404
    assert error.value.message == ErrorMessages.PROJECT_NOT_FOUND
    assert not service.runs_root.exists()


@pytest.mark.parametrize("project_id", ["../" + PROJECT_ID, "a/b", "..", ""])
def test_prepare_treats_unsafe_project_ids_as_not_found(service: SandboxService, project_id: str) -> None:
    with pytest.raises(NotFoundError):
        service.prepare_project_dir(project_id, RUN_ID)


@pytest.mark.parametrize("run_id", ["..", "../other", "a/b", ""])
def test_prepare_rejects_unsafe_run_ids(service: SandboxService, run_id: str) -> None:
    _store_archive(service, make_zip(SAMPLE_FILES))

    with pytest.raises(ValueError):
        service.prepare_project_dir(PROJECT_ID, run_id)

    assert not service.runs_root.exists()


def test_prepare_rejects_zip_slip_and_leaves_nothing_behind(service: SandboxService) -> None:
    _store_archive(service, make_zip({"pom.xml": "<project/>", "../evil.txt": "pwned"}))

    with pytest.raises(ValidationError) as error:
        service.prepare_project_dir(PROJECT_ID, RUN_ID)

    assert error.value.error_code == ErrorCode.INVALID_ARCHIVE
    assert not (service.runs_root / RUN_ID / "evil.txt").exists()
    assert not (service.runs_root / RUN_ID / "project").exists()


def test_prepare_removes_a_partial_copy_when_unzipping_fails(service: SandboxService, tmp_path: Path) -> None:
    source = tmp_path / "damaged.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.txt", b"first file")
        archive.writestr("b.txt", b"SECOND-CONTENT")
    _store_archive(service, source.read_bytes().replace(b"SECOND-CONTENT", b"XECOND-CONTENT"))

    with pytest.raises(ValidationError, match="Not a valid zip"):
        service.prepare_project_dir(PROJECT_ID, RUN_ID)

    assert not (service.runs_root / RUN_ID / "project").exists()


# ---------- open_readonly ----------

def test_open_readonly_mounts_the_project_read_only(
    service: SandboxService, factory: FakeWorkspaceFactory, tmp_path: Path
) -> None:
    project_dir = _project_dir(tmp_path)

    with service.open_readonly(project_dir) as session:
        assert isinstance(session, SandboxSession)
        assert session.container_id == CONTAINER_ID
        assert session.workspace is factory.workspace

    assert factory.calls == [{
        "server_image": "test-image:1",
        "working_dir": SANDBOX_PROJECT_DIR,
        "volumes": [f"{project_dir.resolve()}:/workspace/project:ro"],
        "detach_logs": False,
        "health_check_timeout": 30,
    }]
    assert SANDBOX_PROJECT_DIR == "/workspace/project"


def test_limits_are_applied_before_the_session_is_handed_out(
    service: SandboxService, docker: FakeDocker, tmp_path: Path
) -> None:
    with service.open_readonly(_project_dir(tmp_path)):
        assert docker.calls == [
            ["update", "--cpus", "1.5", "--memory", "512m", "--memory-swap", "512m", "--", CONTAINER_ID]
        ]


def test_on_started_gets_the_container_id_after_the_limits(
    service: SandboxService, docker: FakeDocker, tmp_path: Path
) -> None:
    seen: list[tuple[str, int]] = []

    with service.open_readonly(_project_dir(tmp_path), on_started=lambda cid: seen.append((cid, len(docker.calls)))):
        assert seen == [(CONTAINER_ID, 1)]

    assert seen == [(CONTAINER_ID, 1)]


def test_container_is_cleaned_up_after_the_session(
    service: SandboxService, factory: FakeWorkspaceFactory, tmp_path: Path
) -> None:
    with service.open_readonly(_project_dir(tmp_path)):
        assert factory.workspace.cleanup_calls == 0

    assert factory.workspace.cleanup_calls == 1


def test_container_is_cleaned_up_when_the_session_fails(
    service: SandboxService, factory: FakeWorkspaceFactory, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="analysis broke"):
        with service.open_readonly(_project_dir(tmp_path)):
            raise RuntimeError("analysis broke")

    assert factory.workspace.cleanup_calls == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RuntimeError("Container failed to become healthy in time"), "the agent server did not start within 30 seconds"),
        (RuntimeError("Docker is not available. Please install and start Docker Desktop/daemon."), "Docker is not available"),
        (FileNotFoundError(2, "The system cannot find the file specified", "docker"), "Docker is not available"),
        (RuntimeError("Failed to run docker container: Unable to find image 'x:1' locally"), "the sandbox image is not available"),
        (RuntimeError("Container stopped unexpectedly. Logs:\nsecret-log-line"), "the container stopped right after starting"),
        (RuntimeError("Port 30001 is not available"), "no free local port"),
        (RuntimeError("Failed to run docker container: invalid mount C:\\server\\path"), "Docker could not start the container"),
        (ValueError("odd failure at C:\\server\\path"), "unexpected error"),
    ],
)
def test_startup_failure_is_a_short_sandbox_error(tmp_path: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str) -> None:
    service = _service(tmp_path, FakeWorkspaceFactory(error=error))
    monkeypatch.setattr(service, "_run_docker", docker)
    started: list[str] = []

    with pytest.raises(SandboxError) as raised:
        with service.open_readonly(_project_dir(tmp_path), on_started=started.append):
            pytest.fail("the session must not start")

    assert raised.value.error_code == ErrorCode.SANDBOX_FAILED
    assert raised.value.status_code == 503
    assert raised.value.message == ErrorMessages.SANDBOX_FAILED.format(reason=reason)
    assert "secret-log-line" not in raised.value.message
    assert "C:\\server\\path" not in raised.value.message
    assert docker.calls == []
    assert started == []


def test_missing_project_folder_fails_without_starting_docker(service: SandboxService, factory: FakeWorkspaceFactory, tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="the project folder is missing"):
        with service.open_readonly(tmp_path / "gone"):
            pytest.fail("the session must not start")

    assert factory.calls == []


@pytest.mark.parametrize(
    "answer",
    [_completed(1, "Error response from daemon: Cannot update container"), subprocess.TimeoutExpired("docker", 60), OSError("no docker")],
    ids=["docker-error", "timeout", "cannot-run"],
)
def test_container_is_stopped_when_limits_cannot_be_set(
    tmp_path: Path, factory: FakeWorkspaceFactory, monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    service = _service(tmp_path, factory)
    monkeypatch.setattr(service, "_run_docker", FakeDocker(answer))
    started: list[str] = []

    with pytest.raises(SandboxError, match="limits could not be set"):
        with service.open_readonly(_project_dir(tmp_path), on_started=started.append):
            pytest.fail("the session must not start")

    assert factory.workspace.cleanup_calls == 1
    assert started == []


def test_failing_on_started_still_cleans_up(service: SandboxService, factory: FakeWorkspaceFactory, tmp_path: Path) -> None:
    def on_started(container_id: str) -> None:
        raise RuntimeError("could not record the container")

    with pytest.raises(RuntimeError, match="could not record"):
        with service.open_readonly(_project_dir(tmp_path), on_started=on_started):
            pytest.fail("the session must not start")

    assert factory.workspace.cleanup_calls == 1


def test_workspace_without_a_container_id_is_a_sandbox_error(tmp_path: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeWorkspaceFactory(container_id=None)
    service = _service(tmp_path, factory)
    monkeypatch.setattr(service, "_run_docker", docker)

    with pytest.raises(SandboxError, match="container id is unknown"):
        with service.open_readonly(_project_dir(tmp_path)):
            pytest.fail("the session must not start")

    assert factory.workspace.cleanup_calls == 1
    assert docker.calls == []


def test_failed_cleanup_falls_back_to_docker_stop_and_keeps_the_original_error(
    tmp_path: Path, docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = FakeWorkspaceFactory(cleanup_error=RuntimeError("cleanup broke"))
    service = _service(tmp_path, factory)
    monkeypatch.setattr(service, "_run_docker", docker)

    with pytest.raises(ValueError, match="analysis failed"):
        with service.open_readonly(_project_dir(tmp_path)):
            raise ValueError("analysis failed")

    assert factory.workspace.cleanup_calls == 1
    assert docker.calls[-1] == ["stop", "--", CONTAINER_ID]


# ---------- the default workspace class ----------

def test_half_started_container_is_stopped_when_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[str] = []

    def start_then_fail(self: DockerWorkspace, image: str, context: Any) -> None:
        self._container_id = "c0ffee"
        raise RuntimeError("Container failed to become healthy in time")

    def cleanup(self: DockerWorkspace) -> None:
        if self._container_id:
            stopped.append(self._container_id)
            self._container_id = None

    monkeypatch.setattr(DockerWorkspace, "_start_container", start_then_fail)
    monkeypatch.setattr(DockerWorkspace, "cleanup", cleanup)

    with pytest.raises(RuntimeError, match="healthy"):
        _SandboxDockerWorkspace(server_image="test-image:1", working_dir=SANDBOX_PROJECT_DIR)

    assert stopped == ["c0ffee"]


def test_started_container_is_kept_running(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[str] = []

    def start(self: DockerWorkspace, image: str, context: Any) -> None:
        self._container_id = "c0ffee"

    def cleanup(self: DockerWorkspace) -> None:
        if self._container_id:
            stopped.append(self._container_id)
            self._container_id = None

    monkeypatch.setattr(DockerWorkspace, "_start_container", start)
    monkeypatch.setattr(DockerWorkspace, "cleanup", cleanup)

    workspace = _SandboxDockerWorkspace(server_image="test-image:1", working_dir=SANDBOX_PROJECT_DIR)

    assert stopped == []
    assert isinstance(workspace, DockerWorkspace)
    workspace.cleanup()
    assert stopped == ["c0ffee"]


def test_default_workspace_factory_is_the_self_cleaning_docker_workspace(tmp_path: Path) -> None:
    assert SandboxService(data_dir=tmp_path).workspace_factory is _SandboxDockerWorkspace


# ---------- stop_container ----------

def test_stop_container_runs_docker_stop(service: SandboxService, docker: FakeDocker) -> None:
    service.stop_container(CONTAINER_ID)

    assert docker.calls == [["stop", "--", CONTAINER_ID]]


def test_stopping_a_missing_container_is_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(service, "_run_docker", FakeDocker(_completed(1, f"Error response from daemon: No such container: {CONTAINER_ID}\n")))

    service.stop_container(CONTAINER_ID)


@pytest.mark.parametrize(
    "answer",
    [_completed(1, "Error response from daemon: permission denied"), _completed(1, ""), subprocess.TimeoutExpired("docker", 60), OSError("no docker")],
    ids=["docker-error", "no-output", "timeout", "cannot-run"],
)
def test_other_stop_failures_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Any) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(service, "_run_docker", FakeDocker(answer))

    with pytest.raises(SandboxError) as error:
        service.stop_container(CONTAINER_ID)

    assert error.value.error_code == ErrorCode.SANDBOX_FAILED


@pytest.mark.parametrize("container_id", ["", "-t", "--help", "a b", "../x", "x;rm"])
def test_stop_container_rejects_values_that_are_not_container_ids(service: SandboxService, docker: FakeDocker, container_id: str) -> None:
    with pytest.raises(ValueError):
        service.stop_container(container_id)

    assert docker.calls == []


# ---------- remove_run_dir ----------

def test_remove_run_dir_deletes_the_run_folder(service: SandboxService) -> None:
    _store_archive(service, make_zip(SAMPLE_FILES))
    project_dir = service.prepare_project_dir(PROJECT_ID, RUN_ID)

    service.remove_run_dir(RUN_ID)

    assert not project_dir.exists()
    assert not (service.runs_root / RUN_ID).exists()
    assert service.storage.archive_path(PROJECT_ID).is_file()  # the project itself stays
    service.remove_run_dir(RUN_ID)  # already gone: no error


@pytest.mark.parametrize("run_id", ["..", "", ".", "../..", "a/b"])
def test_remove_run_dir_ignores_unexpected_ids(service: SandboxService, run_id: str) -> None:
    _store_archive(service, make_zip(SAMPLE_FILES))
    service.prepare_project_dir(PROJECT_ID, RUN_ID)

    service.remove_run_dir(run_id)

    assert service.storage.archive_path(PROJECT_ID).is_file()
    assert (service.runs_root / RUN_ID / "project" / "pom.xml").is_file()


# ---------- configuration ----------

def test_session_key_is_created_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SESSION_KEY_ENV, raising=False)

    _service(tmp_path)

    assert len(os.environ[SESSION_KEY_ENV]) >= 40


def test_existing_session_key_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SESSION_KEY_ENV, "key-chosen-by-the-operator-0123456789")

    _service(tmp_path)

    assert os.environ[SESSION_KEY_ENV] == "key-chosen-by-the-operator-0123456789"


def test_empty_session_key_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SESSION_KEY_ENV, "")

    _service(tmp_path)

    assert len(os.environ[SESSION_KEY_ENV]) >= 40


def test_sdk_command_log_does_not_show_the_session_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(SESSION_KEY_ENV, "super-secret-session-key-0123456789")
    _service(tmp_path)
    _service(tmp_path)  # the log filter is installed once
    sdk_logger = logging.getLogger("openhands.sdk.utils.command")

    with caplog.at_level(logging.INFO, logger=sdk_logger.name):
        sdk_logger.info("$ %s", f"docker run -d -e {SESSION_KEY_ENV}=super-secret-session-key-0123456789 test-image:1")

    assert "super-secret-session-key" not in caplog.text
    assert f"{SESSION_KEY_ENV}=***" in caplog.text
    assert sum(type(f).__name__ == "_HideSessionKey" for f in sdk_logger.filters) == 1


@pytest.mark.parametrize("field", ["cpus", "memory_mb", "startup_timeout_seconds"])
def test_limits_must_be_positive(tmp_path: Path, field: str) -> None:
    # Docker reads 0 as "no limit".
    with pytest.raises(ValueError):
        SandboxService(data_dir=tmp_path, workspace_factory=FakeWorkspaceFactory(), **{field: 0})


# ---------- real Docker (opt-in) ----------

@pytest.mark.docker
@docker_tests
def test_real_sandbox_reads_but_cannot_write_the_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "hello.txt").write_text("hello from the host")
    service = SandboxService(cpus=1, memory_mb=1024, data_dir=tmp_path / "data")
    started: list[str] = []

    with service.open_readonly(project_dir, on_started=started.append) as session:
        read = session.workspace.execute_command("cat /workspace/project/src/hello.txt")
        write = session.workspace.execute_command("touch /workspace/project/new.txt")
        limits = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}",
             session.container_id],
            capture_output=True, encoding="utf-8", check=True,
        )

    assert started == [session.container_id]
    assert read.exit_code == 0
    assert "hello from the host" in read.stdout
    assert write.exit_code != 0
    assert "read-only" in (write.stdout + write.stderr).lower()
    assert not (project_dir / "new.txt").exists()
    assert limits.stdout.split() == [str(10**9), str(1024 * 1024 * 1024), str(1024 * 1024 * 1024)]
    service.stop_container(session.container_id)  # already stopped and removed: not an error
