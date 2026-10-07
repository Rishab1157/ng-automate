"""Where project archives live: <DATA_DIR>/projects/<project_id>/source.zip.

Archives are written once and never changed. Moving to S3 later only changes this class.
"""

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import settings
from app.core.exceptions import ErrorMessages, PayloadTooLargeError


class ProjectStorageService:
    def __init__(self, root: Path = settings.DATA_DIR, max_bytes: int = settings.MAX_PROJECT_BYTES) -> None:
        self.root = root.resolve()
        self.max_bytes = max_bytes

    def archive_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "source.zip"

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

    def delete(self, project_id: str) -> None:
        shutil.rmtree(self._project_dir(project_id), ignore_errors=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.root / "projects" / project_id
