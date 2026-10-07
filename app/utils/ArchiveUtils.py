"""Project archives (zip files): checking untrusted ones and creating our own."""

import hashlib
import os
import stat
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from app.core.exceptions import ErrorCode, ErrorMessages, ValidationError
from app.models.projectModel import ArchiveModel

MAX_FILES = 100_000
# An archive may unzip to at most this many times the size limit (stops "zip bombs").
MAX_UNZIP_FACTOR = 5
# Generated or tool folders: large, rebuilt in the sandbox, never needed in a snapshot.
SKIPPED_DIRS = frozenset({".git", "node_modules", "target", ".venv", "venv", "__pycache__", ".gradle", ".idea"})
_CHUNK_BYTES = 1024 * 1024


def inspect_archive(path: Path, max_bytes: int) -> ArchiveModel:
    """Validate an untrusted zip and describe it.

    Rejects entries that would land outside the target folder when unzipped (zip slip),
    symbolic links, and archives that unzip far beyond the size limit (zip bombs).
    """
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile:
        raise _invalid(ErrorMessages.INVALID_ZIP) from None

    if len(entries) > MAX_FILES:
        raise _invalid(ErrorMessages.TOO_MANY_FILES.format(limit=MAX_FILES))

    unzipped_bytes = 0
    file_count = 0
    for entry in entries:
        _check_entry_path(entry.filename)
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise _invalid(ErrorMessages.ZIP_LINK_NOT_ALLOWED.format(name=entry.filename))
        unzipped_bytes += entry.file_size
        if unzipped_bytes > max_bytes * MAX_UNZIP_FACTOR:
            raise _invalid(ErrorMessages.ZIP_TOO_LARGE_UNZIPPED)
        if not entry.is_dir():
            file_count += 1

    if file_count == 0:
        raise _invalid(ErrorMessages.EMPTY_ZIP)
    return ArchiveModel(size_bytes=path.stat().st_size, sha256=sha256_file(path), file_count=file_count)


def create_archive(source_dir: Path, destination: Path) -> None:
    """Zip a folder, leaving out SKIPPED_DIRS and symbolic links."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in _iter_files(source_dir):
            archive.write(file, file.relative_to(source_dir).as_posix())


def directory_size(source_dir: Path) -> int:
    """Total size of the files create_archive would include."""
    return sum(file.stat().st_size for file in _iter_files(source_dir))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _check_entry_path(name: str) -> None:
    normalized = name.replace("\\", "/")
    is_absolute = normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":")
    if is_absolute or ".." in PurePosixPath(normalized).parts:
        raise _invalid(ErrorMessages.UNSAFE_ZIP_PATH.format(name=name))


def _iter_files(source_dir: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIPPED_DIRS and not (root_path / d).is_symlink()]
        for name in files:
            file = root_path / name
            if not file.is_symlink():
                yield file


def _invalid(message: str) -> ValidationError:
    return ValidationError(message, error_code=ErrorCode.INVALID_ARCHIVE)
