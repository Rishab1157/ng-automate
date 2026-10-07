import stat
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from app.core.exceptions import ErrorCode, ValidationError
from app.utils import ArchiveUtils
from app.utils.ArchiveUtils import extract_archive
from tests.conftest import make_zip

MAX_BYTES = 10 * 1024 * 1024


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "source.zip"
    path.write_bytes(data)
    return path


def _zip_with(tmp_path: Path, entries: list[tuple[zipfile.ZipInfo | str, bytes]]) -> Path:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for entry, content in entries:
            archive.writestr(entry, content)
    return path


def _files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _skip_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach the checks inside the unzip loop, which back up inspect_archive."""
    monkeypatch.setattr(ArchiveUtils, "inspect_archive", lambda path, max_bytes: None)


def _set_central_directory_field(data: bytes, offset: int, value: int) -> bytes:
    # Central directory header: signature(4) made-by(2) needed(2) flags(2 @8) method(2 @10) ...
    field = data.rfind(b"PK\x01\x02") + offset
    return data[:field] + struct.pack("<H", value) + data[field + 2:]


# ---------- normal archives ----------

def test_files_are_unzipped_with_their_folders(tmp_path: Path) -> None:
    archive = _zip_with(tmp_path, [
        ("pom.xml", b"<project/>"),
        ("src/test/java/LoginTest.java", b"class LoginTest {}"),
        (zipfile.ZipInfo("empty-folder/"), b""),
    ])
    destination = tmp_path / "runs" / "r1" / "project"

    extract_archive(archive, destination, MAX_BYTES)

    assert _files(destination) == {"pom.xml": b"<project/>", "src/test/java/LoginTest.java": b"class LoginTest {}"}
    assert not (destination / "empty-folder").exists()  # directory entries are skipped


def test_large_files_are_copied_completely(tmp_path: Path) -> None:
    content = bytes(range(256)) * 10_000  # 2.5 MB: several read chunks
    archive = _write(tmp_path, make_zip({"data/blob.bin": content}))
    destination = tmp_path / "out"

    extract_archive(archive, destination, MAX_BYTES)

    assert (destination / "data" / "blob.bin").read_bytes() == content


def test_backslash_names_become_folders(tmp_path: Path) -> None:
    archive = _zip_with(tmp_path, [(zipfile.ZipInfo("src\\main\\App.java"), b"class App {}")])
    destination = tmp_path / "out"

    extract_archive(archive, destination, MAX_BYTES)

    assert _files(destination) == {"src/main/App.java": b"class App {}"}


# ---------- unsafe archives ----------

@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/evil.txt", "C:/evil.txt"])
def test_zip_slip_is_rejected_before_anything_is_written(tmp_path: Path, name: str) -> None:
    archive = _zip_with(tmp_path, [("good.txt", b"fine"), (zipfile.ZipInfo(name), b"pwned")])
    destination = tmp_path / "out" / "project"

    with pytest.raises(ValidationError) as error:
        extract_archive(archive, destination, MAX_BYTES)

    assert error.value.error_code == ErrorCode.INVALID_ARCHIVE
    assert "Unsafe path" in error.value.message
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out" / "evil.txt").exists()
    assert not destination.exists() or _files(destination) == {}


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/evil.txt"])
def test_each_target_is_checked_again_after_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _skip_inspection(monkeypatch)
    archive = _zip_with(tmp_path, [(zipfile.ZipInfo(name), b"pwned")])
    destination = tmp_path / "out" / "project"

    with pytest.raises(ValidationError, match="Unsafe path"):
        extract_archive(archive, destination, MAX_BYTES)

    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out" / "evil.txt").exists()
    assert not Path("/evil.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters only exist on Windows")
def test_drive_letter_inside_a_name_cannot_escape_on_windows(tmp_path: Path) -> None:
    # "a/b:c" passes the name check, but on Windows "b:c" is a path on drive B:.
    archive = _zip_with(tmp_path, [(zipfile.ZipInfo("a/b:c"), b"pwned")])

    with pytest.raises(ValidationError, match="Unsafe path"):
        extract_archive(archive, tmp_path / "out", MAX_BYTES)


def test_links_are_rejected(tmp_path: Path) -> None:
    link = zipfile.ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _zip_with(tmp_path, [(link, b"/etc/passwd")])
    destination = tmp_path / "out"

    with pytest.raises(ValidationError, match="Links are not allowed"):
        extract_archive(archive, destination, MAX_BYTES)

    assert not (destination / "link").exists()


# ---------- size limits ----------

def test_declared_size_is_checked_before_unzipping(tmp_path: Path) -> None:
    # 6000 bytes unzipped with a 1000-byte limit: over the 5x allowance.
    archive = _write(tmp_path, make_zip({"zeros.bin": b"\0" * 6000}))
    destination = tmp_path / "out"

    with pytest.raises(ValidationError, match="Unzipped size is too large"):
        extract_archive(archive, destination, max_bytes=1000)

    assert not (destination / "zeros.bin").exists()


@pytest.mark.parametrize("sizes", [{"big.bin": 600}, {"a.bin": 300, "b.bin": 300}])
def test_bytes_really_written_are_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sizes: dict[str, int]) -> None:
    # Allowance with max_bytes=100 is 500 bytes, whatever the zip headers claim.
    _skip_inspection(monkeypatch)
    archive = _write(tmp_path, make_zip({name: b"x" * size for name, size in sizes.items()}))

    with pytest.raises(ValidationError, match="Unzipped size is too large"):
        extract_archive(archive, tmp_path / "out", max_bytes=100)


def test_archive_exactly_at_the_allowance_is_accepted(tmp_path: Path) -> None:
    archive = _write(tmp_path, make_zip({"a.bin": b"x" * 300, "b.bin": b"y" * 200}))

    extract_archive(archive, tmp_path / "out", max_bytes=100)

    assert len((tmp_path / "out" / "a.bin").read_bytes()) == 300


# ---------- damaged archives ----------

def test_not_a_zip_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Not a valid zip"):
        extract_archive(_write(tmp_path, b"hello"), tmp_path / "out", MAX_BYTES)


def test_corrupted_entry_is_an_invalid_zip(tmp_path: Path) -> None:
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.txt", b"first file")
        archive.writestr("b.txt", b"SECOND-CONTENT")
    path.write_bytes(path.read_bytes().replace(b"SECOND-CONTENT", b"XECOND-CONTENT"))  # CRC no longer matches

    with pytest.raises(ValidationError) as error:
        extract_archive(path, tmp_path / "out", MAX_BYTES)

    assert error.value.error_code == ErrorCode.INVALID_ARCHIVE
    assert error.value.message == "Not a valid zip file"


@pytest.mark.parametrize(
    ("offset", "value"),
    [(8, 0x0001), (10, 99)],
    ids=["encrypted-entry", "unsupported-compression"],
)
def test_entries_that_cannot_be_read_are_an_invalid_zip(tmp_path: Path, offset: int, value: int) -> None:
    data = _set_central_directory_field(make_zip({"a.txt": "content"}), offset, value)

    with pytest.raises(ValidationError, match="Not a valid zip"):
        extract_archive(_write(tmp_path, data), tmp_path / "out", MAX_BYTES)
