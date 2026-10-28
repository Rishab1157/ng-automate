"""ProjectStorageService.ensure_working_copy: the project is unpacked into project.partial and only then renamed to
project, so a run never sees half a project. Windows refuses that rename while any program has a file inside open (an
antivirus scanning the files just unpacked does): the rename is tried again after 2, 3, 5 and 7 seconds."""

import errno
import importlib
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.projectService.ProjectStorageService import ProjectStorageService

storage_module = importlib.import_module("app.services.projectService.ProjectStorageService")
ON_WINDOWS = sys.platform == "win32"
FILES = {"pom.xml": "<project/>", "src/test/java/LoginTest.java": "class LoginTest {}", "lib/big.bin": "x" * 50_000}
PROJECT_ID = "p1"
REAL_RENAME = Path.rename


@pytest.fixture
def storage(tmp_path: Path) -> ProjectStorageService:
    store = ProjectStorageService(root=tmp_path, max_bytes=10_000_000)
    archive = store.archive_path(PROJECT_ID)
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, text in FILES.items():
            zipped.writestr(name, text)
    return store


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between tries, recorded instead of slept. Only the storage module's `time` is replaced: replacing
    time.sleep itself would also stop every other thread's sleep (PyMongo's monitor then spins)."""
    slept: list[float] = []
    monkeypatch.setattr(storage_module, "time", SimpleNamespace(sleep=slept.append))
    return slept


def _refuse(monkeypatch: pytest.MonkeyPatch, times: int, error: OSError) -> None:
    """The next `times` renames fail with `error`, as Windows does while a file inside is in use."""
    left = [times]

    def rename(self: Path, target: Path) -> Path:
        if left[0] > 0:
            left[0] -= 1
            raise error
        return REAL_RENAME(self, target)

    monkeypatch.setattr(Path, "rename", rename)


def _in_use() -> PermissionError:
    return PermissionError(errno.EACCES, "[WinError 5] Access is denied")


def _assert_whole_project(working: Path) -> None:
    assert working.name == "project"
    assert {p.relative_to(working).as_posix() for p in working.rglob("*") if p.is_file()} == set(FILES)
    assert not working.with_name("project.partial").exists()


def test_the_project_is_renamed_at_once_when_nothing_holds_its_files(storage: ProjectStorageService, waits: list[float]) -> None:
    _assert_whole_project(storage.ensure_working_copy(PROJECT_ID))
    assert waits == []


def test_a_refused_rename_is_tried_again_after_2_then_3_seconds(
    storage: ProjectStorageService, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse(monkeypatch, 2, _in_use())

    _assert_whole_project(storage.ensure_working_copy(PROJECT_ID))

    assert waits == [2, 3]


def test_after_2_3_5_and_7_seconds_the_real_error_is_raised_and_nothing_is_left_behind(
    storage: ProjectStorageService, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse(monkeypatch, 5, _in_use())

    with pytest.raises(PermissionError, match="Access is denied"):
        storage.ensure_working_copy(PROJECT_ID)

    assert waits == [2, 3, 5, 7]
    assert not storage.working_dir(PROJECT_ID).exists()
    assert not storage.working_dir(PROJECT_ID).with_name("project.partial").exists()


def test_other_errors_are_not_tried_again(
    storage: ProjectStorageService, waits: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse(monkeypatch, 1, OSError(errno.ENOSPC, "No space left on device"))

    with pytest.raises(OSError, match="No space left"):
        storage.ensure_working_copy(PROJECT_ID)

    assert waits == []


def test_unpacking_that_breaks_halfway_leaves_no_project_and_the_next_try_works(
    storage: ProjectStorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def breaks_halfway(archive: Path, target: Path, max_bytes: int) -> None:
        (target / "src").mkdir(parents=True)
        (target / "pom.xml").write_text("<project/>", encoding="utf-8")
        raise OSError(errno.EIO, "the archive could not be read to the end")

    monkeypatch.setattr(storage_module, "extract_archive", breaks_halfway)
    with pytest.raises(OSError):
        storage.ensure_working_copy(PROJECT_ID)
    assert not storage.working_dir(PROJECT_ID).exists()  # never half a project under the real name

    monkeypatch.undo()
    _assert_whole_project(storage.ensure_working_copy(PROJECT_ID))


def test_a_second_request_for_the_same_project_never_breaks_the_first_ones_unpacking(
    storage: ProjectStorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first request has unpacked everything and is about to rename (on Windows it may wait seconds for that).
    # A second request for the same project starts then: it must not clear the folder and unpack it again under the
    # first one, which would then rename a half-written folder to project.
    real_extract = storage_module.extract_archive
    calls: list[str] = []

    def extract(archive: Path, target: Path, max_bytes: int) -> None:
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            real_extract(archive, target, max_bytes)
            time.sleep(0.6)  # the first request is between unpacking and renaming
        else:
            (target / "pom.xml").parent.mkdir(parents=True, exist_ok=True)
            (target / "pom.xml").write_text("<project/>", encoding="utf-8")  # only its first file so far
            time.sleep(1.0)
            real_extract(archive, target, max_bytes)

    monkeypatch.setattr(storage_module, "extract_archive", extract)
    results: dict[str, Path] = {}
    errors: dict[str, BaseException] = {}

    def unpack() -> None:
        name = threading.current_thread().name
        try:
            results[name] = storage.ensure_working_copy(PROJECT_ID)
        except BaseException as error:  # noqa: BLE001 - the test reports any failure
            errors[name] = error

    first, second = threading.Thread(target=unpack, name="first"), threading.Thread(target=unpack, name="second")
    first.start()
    time.sleep(0.2)
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)

    assert errors == {}
    assert set(results) == {"first", "second"}
    _assert_whole_project(storage.working_dir(PROJECT_ID))
    assert calls == ["first"]  # the second request waited and used the first one's project


@pytest.mark.skipif(not ON_WINDOWS, reason="only Windows refuses to rename a folder with an open file inside")
def test_a_file_held_open_like_an_antivirus_scan_delays_the_rename_but_it_succeeds(
    storage: ProjectStorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_extract = storage_module.extract_archive
    held: list[object] = []

    def extract_then_scan(archive: Path, target: Path, max_bytes: int) -> None:
        real_extract(archive, target, max_bytes)
        held.append(open(target / "lib" / "big.bin", "rb"))  # noqa: SIM115 - like a scanner, closed later
        threading.Timer(1.5, held[0].close).start()

    monkeypatch.setattr(storage_module, "extract_archive", extract_then_scan)
    monkeypatch.setattr(storage_module, "RENAME_RETRY_WAITS", (1, 1, 1, 1))  # same logic, a faster test
    tries: list[int] = []
    monkeypatch.setattr(Path, "rename", lambda self, target: (tries.append(1), REAL_RENAME(self, target))[1])

    _assert_whole_project(storage.ensure_working_copy(PROJECT_ID))
    assert len(tries) >= 2  # the first rename was really refused while the file was open


@pytest.mark.skipif(not ON_WINDOWS, reason="only Windows refuses to rename a folder with an open file inside")
def test_a_file_that_stays_open_gives_the_real_error_and_the_next_request_works_once_it_is_closed(
    storage: ProjectStorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_extract = storage_module.extract_archive
    held: list[object] = []

    def extract_and_hold(archive: Path, target: Path, max_bytes: int) -> None:
        real_extract(archive, target, max_bytes)
        held.append(open(target / "pom.xml", "rb"))  # noqa: SIM115 - closed by the test below

    monkeypatch.setattr(storage_module, "extract_archive", extract_and_hold)
    monkeypatch.setattr(storage_module, "RENAME_RETRY_WAITS", (0.1, 0.1, 0.1, 0.1))
    with pytest.raises(PermissionError):
        storage.ensure_working_copy(PROJECT_ID)
    assert not storage.working_dir(PROJECT_ID).exists()

    held[0].close()
    monkeypatch.undo()
    _assert_whole_project(storage.ensure_working_copy(PROJECT_ID))
