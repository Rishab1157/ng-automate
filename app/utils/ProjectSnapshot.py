"""A copy of a project's text files, to see exactly what an agent changed.

Taken on the host copy of the project before an agent edits it. Build outputs, dependency folders and binary or
very large files are left out: agents rebuild those, and they are not something to review line by line (only the paths
of binary and very large files are kept, to see which ones an agent added). Nothing is ever put back from a snapshot:
an agent's change is never rolled back (the user's rule).
"""

import difflib
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.models.healerModel import FileChangeModel
from app.utils.ArchiveUtils import SKIPPED_DIRS

# Build and tool outputs on top of the archive's skipped folders.
SNAPSHOT_SKIPPED_DIRS = SKIPPED_DIRS | {"build", "dist", "out", "bin", "obj", "ngauto-results", ".pytest_cache", "allure-results"}
# The temporary file an agent's file editor writes next to a file while saving it (".LoginTest.java.u1d6m2yl.tmp"):
# never a project file, only there for a moment (or left behind by a crash mid-write).
_IN_FLIGHT_WRITE = re.compile(r"^\..+\.[a-z0-9_]{8}\.tmp$")
MAX_FILE_BYTES = 1024 * 1024
MAX_FILES = 20_000
MAX_DIFF_CHARS = 20_000


def is_in_flight_write(name: str) -> bool:
    return _IN_FLIGHT_WRITE.match(name) is not None


@dataclass(frozen=True)
class ProjectSnapshot:
    root: Path
    files: dict[str, str]  # repo-relative posix path -> text
    # The files left out of `files` (binary, or larger than MAX_FILE_BYTES): their paths only.
    other_files: frozenset[str] = frozenset()

    @classmethod
    def capture(cls, project_dir: Path) -> "ProjectSnapshot":
        root = project_dir.resolve()
        files, other_files = _read_files(root)
        return cls(root=root, files=files, other_files=frozenset(other_files))

    def changes(self) -> list[FileChangeModel]:
        """What changed since the snapshot, sorted by path."""
        return _compare(self.files, _read_text_files(self.root))

    def added_other_files(self) -> list[str]:
        """Binary or very large files that are new since the snapshot (a downloaded jar, an archive), sorted: changes()
        leaves them out, so without this an agent could add one unseen."""
        return sorted(set(_read_files(self.root)[1]) - self.other_files - self.files.keys())

    def content(self, path: str) -> str | None:
        return self.files.get(path)


def project_fingerprint(project_dir: Path) -> str:
    """One hash of the project's text files: equal fingerprints mean the same files (build outputs left out)."""
    digest = hashlib.sha256()
    for path, text in sorted(_read_text_files(project_dir.resolve()).items()):
        digest.update(path.encode("utf-8") + b"\0" + hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()


def compare_projects(before_dir: Path, after_dir: Path) -> list[FileChangeModel]:
    """How the text files of `after_dir` differ from `before_dir` (e.g. a run's copy against the project), by path."""
    return _compare(_read_text_files(before_dir.resolve()), _read_text_files(after_dir.resolve()))


def _compare(before_files: dict[str, str], after_files: dict[str, str]) -> list[FileChangeModel]:
    changes: list[FileChangeModel] = []
    for path in sorted(before_files.keys() | after_files.keys()):
        before, after = before_files.get(path), after_files.get(path)
        if before == after:
            continue
        change = "added" if before is None else "deleted" if after is None else "modified"
        changes.append(FileChangeModel(path=path, change=change, diff=_diff(path, before or "", after or "")))
    return changes


def _read_text_files(root: Path) -> dict[str, str]:
    return _read_files(root)[0]


def _read_files(root: Path) -> tuple[dict[str, str], list[str]]:
    """The text files (path -> text), and the paths of the binary or very large ones."""
    files: dict[str, str] = {}
    other_files: list[str] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_SKIPPED_DIRS and not (Path(current) / d).is_symlink()]
        for name in names:
            path = Path(current) / name
            if path.is_symlink() or is_in_flight_write(name):
                continue
            relative = path.relative_to(root).as_posix()
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    other_files.append(relative)
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\0" in data:
                other_files.append(relative)  # binary
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                other_files.append(relative)
                continue
            files[relative] = text
            if len(files) >= MAX_FILES:
                return files, other_files
    return files, other_files


def _diff(path: str, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=f"a/{path}", tofile=f"b/{path}"
    )
    text = "".join(lines)
    return text if len(text) <= MAX_DIFF_CHARS else text[: MAX_DIFF_CHARS - 1] + "…"
