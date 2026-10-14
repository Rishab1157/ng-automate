"""Where a project's files live: <DATA_DIR>/projects/<project_id>/

- source.zip  the upload or the git snapshot, written once and never changed
- project/    the editable copy: what the editor shows and saves, and what every run starts from
- test-data/  the original test-data files, kept apart from the project (never inside it)

Each run works on its own copy in <DATA_DIR>/runs/<run_id>/ (see SandboxService).
Moving to S3 later only changes this class.
"""

import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from bson import ObjectId

from app.config import settings
from app.core.exceptions import ErrorMessages, PayloadTooLargeError
from app.utils.ArchiveUtils import extract_archive

RUNS_DIR = "runs"
MAX_STORED_NAME_CHARS = 150
# Characters Windows and most tools do not accept in a file name.
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ProjectStorageService:
    def __init__(self, root: Path = settings.DATA_DIR, max_bytes: int = settings.MAX_PROJECT_BYTES) -> None:
        self.root = root.resolve()
        self.max_bytes = max_bytes

    def archive_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "source.zip"

    def working_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project"

    def test_data_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "test-data"

    async def save_archive(self, project_id: str, chunks: AsyncIterator[bytes]) -> Path:
        """Stream an upload to disk, stopping as soon as it passes the size limit."""
        path = self.archive_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("wb") as out:
            async for chunk in chunks:
                written += len(chunk)
                if written > self.max_bytes:
                    raise PayloadTooLargeError(ErrorMessages.PROJECT_TOO_LARGE.format(limit_mb=self.max_bytes // (1024 * 1024)))
                out.write(chunk)
        return path

    def ensure_working_copy(self, project_id: str) -> Path:
        """The editable copy, unzipped from the archive the first time (projects made before it existed too)."""
        working = self.working_dir(project_id)
        if working.is_dir():
            return working
        partial = working.with_name(working.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        try:
            extract_archive(self.archive_path(project_id), partial, self.max_bytes)
            partial.rename(working)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return working

    def save_test_data_file(self, project_id: str, filename: str, content: bytes) -> str:
        """Keep an uploaded test-data file as it is; returns its name in the folder (made unique)."""
        folder = self.test_data_dir(project_id)
        folder.mkdir(parents=True, exist_ok=True)
        name = _unique_name(folder, _safe_name(filename))
        (folder / name).write_bytes(content)
        return name

    def delete_test_data_file(self, project_id: str, name: str) -> None:
        (self.test_data_dir(project_id) / _safe_name(name)).unlink(missing_ok=True)

    def apply_run_changes(self, project_id: str, run_project_dir: Path, paths: list[str]) -> int:
        """Bring the files a run's agents changed into the editable copy, as they are at the end of the run.

        A path whose file still exists in the run's copy is copied over; one that no longer exists is deleted. So
        a file changed several times during the run ends up exactly as the run left it.
        """
        working = self.ensure_working_copy(project_id).resolve()
        source_root = run_project_dir.resolve()
        applied = 0
        for path in dict.fromkeys(paths):
            target = (working / path).resolve()
            source = (source_root / path).resolve()
            if not target.is_relative_to(working) or not source.is_relative_to(source_root) or target == working:
                continue  # never outside the project
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif source.exists():
                continue  # a folder where a file was: leave it to the user
            else:
                target.unlink(missing_ok=True)
            applied += 1
        return applied

    def delete(self, project_id: str) -> None:
        shutil.rmtree(self._project_dir(project_id), ignore_errors=True)

    def delete_runs(self, run_ids: list[str]) -> None:
        """Remove the folders runs left behind (a run's own copy of the project; a finished run usually removed it)."""
        for run_id in run_ids:
            if ObjectId.is_valid(run_id):
                shutil.rmtree(self.root / RUNS_DIR / run_id, ignore_errors=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.root / "projects" / project_id


def _safe_name(filename: str) -> str:
    """A plain file name: no folders, no characters file systems reject."""
    name = _UNSAFE_NAME_CHARS.sub("_", Path(filename.replace("\\", "/")).name).strip(" .")
    return (name or "test-data")[-MAX_STORED_NAME_CHARS:]


def _unique_name(folder: Path, name: str) -> str:
    if not (folder / name).exists():
        return name
    stem, dot, suffix = name.rpartition(".")
    stem, suffix = (stem, f".{suffix}") if dot and stem else (name, "")
    number = 2
    while (folder / f"{stem} ({number}){suffix}").exists():
        number += 1
    return f"{stem} ({number}){suffix}"
