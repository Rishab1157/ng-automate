"""A copy of a project's text files, to see exactly what an agent changed and to undo single files.

Taken on the host copy of the project before an agent edits it. Build outputs, dependency folders and binary or
very large files are left out: agents rebuild those, and they are not something to review or restore.
"""

import difflib
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from app.models.healerModel import FileChangeModel
from app.utils.ArchiveUtils import SKIPPED_DIRS

# Build and tool outputs on top of the archive's skipped folders.
SNAPSHOT_SKIPPED_DIRS = SKIPPED_DIRS | {"build", "dist", "out", "bin", "obj", "ngauto-results", ".pytest_cache", "allure-results"}
MAX_FILE_BYTES = 1024 * 1024
MAX_FILES = 20_000
MAX_DIFF_CHARS = 20_000


@dataclass(frozen=True)
class ProjectSnapshot:
    root: Path
    files: dict[str, str]  # repo-relative posix path -> text

    @classmethod
    def capture(cls, project_dir: Path) -> "ProjectSnapshot":
        root = project_dir.resolve()
        return cls(root=root, files=_read_text_files(root))

    def changes(self) -> list[FileChangeModel]:
        """What changed since the snapshot, sorted by path."""
        current = _read_text_files(self.root)
        changes: list[FileChangeModel] = []
        for path in sorted(self.files.keys() | current.keys()):
            before, after = self.files.get(path), current.get(path)
            if before == after:
                continue
            change = "added" if before is None else "deleted" if after is None else "modified"
            changes.append(FileChangeModel(path=path, change=change, diff=_diff(path, before or "", after or "")))
        return changes

    def content(self, path: str) -> str | None:
        return self.files.get(path)

    def restore(self, path: str) -> None:
        """Put one file back as it was in the snapshot (delete it if it did not exist then)."""
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Not a project path: {path!r}")
        original = self.files.get(path)
        if original is None:
            target.unlink(missing_ok=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original, encoding="utf-8", newline="")


def project_fingerprint(project_dir: Path) -> str:
    """One hash of the project's text files: equal fingerprints mean the same files (build outputs left out)."""
    digest = hashlib.sha256()
    for path, text in sorted(_read_text_files(project_dir.resolve()).items()):
        digest.update(path.encode("utf-8") + b"\0" + hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()


def _read_text_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_SKIPPED_DIRS and not (Path(current) / d).is_symlink()]
        for name in names:
            path = Path(current) / name
            if path.is_symlink():
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\0" in data:
                continue  # binary
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files[path.relative_to(root).as_posix()] = text
            if len(files) >= MAX_FILES:
                return files
    return files


def _diff(path: str, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=f"a/{path}", tofile=f"b/{path}"
    )
    text = "".join(lines)
    return text if len(text) <= MAX_DIFF_CHARS else text[: MAX_DIFF_CHARS - 1] + "…"
