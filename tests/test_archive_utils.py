import stat
import zipfile
from pathlib import Path

import pytest

from app.core.exceptions import ErrorCode, ValidationError
from app.utils.ArchiveUtils import create_archive, inspect_archive
from tests.conftest import make_zip

MAX_BYTES = 10 * 1024 * 1024


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "project.zip"
    path.write_bytes(data)
    return path


def test_valid_zip_is_described(tmp_path: Path) -> None:
    path = _write(tmp_path, make_zip({"pom.xml": "<project/>", "src/A.java": "class A {}"}))

    info = inspect_archive(path, MAX_BYTES)

    assert info.file_count == 2
    assert info.size_bytes == path.stat().st_size
    assert len(info.sha256) == 64


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/etc/passwd", "C:/Windows/evil.txt", "..\\evil.txt"])
def test_paths_outside_the_folder_are_rejected(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, make_zip({name: "x"}))

    with pytest.raises(ValidationError) as error:
        inspect_archive(path, MAX_BYTES)

    assert error.value.error_code == ErrorCode.INVALID_ARCHIVE
    assert "Unsafe path" in error.value.message


def test_links_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "project.zip"
    with zipfile.ZipFile(path, "w") as archive:
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(ValidationError, match="Links are not allowed"):
        inspect_archive(path, MAX_BYTES)


def test_zip_bomb_is_rejected(tmp_path: Path) -> None:
    # 6000 bytes unzipped with a 1000-byte limit: over the 5x allowance.
    path = _write(tmp_path, make_zip({"zeros.bin": b"\0" * 6000}))

    with pytest.raises(ValidationError, match="Unzipped size is too large"):
        inspect_archive(path, max_bytes=1000)


def test_non_zip_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Not a valid zip"):
        inspect_archive(_write(tmp_path, b"hello"), MAX_BYTES)


def test_empty_zip_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no files"):
        inspect_archive(_write(tmp_path, make_zip({})), MAX_BYTES)


def test_create_archive_leaves_out_generated_folders(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    for relative in [".git/config", "node_modules/lib.js", "target/A.class", "src/Main.java"]:
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x")
    destination = tmp_path / "out.zip"

    create_archive(source, destination)

    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["src/Main.java"]
