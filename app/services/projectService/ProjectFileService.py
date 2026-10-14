"""The editor's view of a project: the project's editable copy and, beside it, its test-data files.

Users list, read and save text files of the project; test-data files and binary files are read-only (download).
Every path is checked to stay inside its folder. While a run uses the project, its files cannot be saved: the
agents work on a copy and their changes come back when the run completes.
"""

import asyncio
import os
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import ErrorMessages, NotFoundError, PayloadTooLargeError, ProjectBusyError, ValidationError
from app.models.projectFileModel import FileContentModel, FileNodeModel, FileTreeModel, ProjectFileRoot, RunFilesModel
from app.agents.RunnerAgent.TestCommandResolver import is_test_file, read_small_text
from app.utils.ArchiveUtils import SKIPPED_DIRS
from app.utils.ProjectSnapshot import compare_projects, is_in_flight_write

from .ProjectService import ProjectService

if TYPE_CHECKING:
    # The run service imports this package: it is loaded on first use, not at import time.
    from app.models.runModel import RunModel
    from app.services.runService import RunService

# Larger files are not shown or edited in the browser: they can be downloaded.
MAX_EDITABLE_BYTES = 1024 * 1024
MAX_TREE_ENTRIES = 5000
# Folders users never edit; .git also must never be written.
HIDDEN_DIRS = SKIPPED_DIRS
_DRIVE = re.compile(r"^[A-Za-z]:")


class ProjectFileService:
    def __init__(self, project_service: ProjectService | None = None, run_service: "RunService | None" = None) -> None:
        self.project_service = project_service or ProjectService()
        self._run_service = run_service
        self.storage = self.project_service.storage

    @property
    def run_service(self) -> "RunService":
        if self._run_service is None:
            from app.services.runService import RunService

            self._run_service = RunService()
        return self._run_service

    async def tree(self, project_id: str, org_id: str) -> FileTreeModel:
        await self.project_service.get(project_id, org_id)
        working = await asyncio.to_thread(self.storage.ensure_working_copy, project_id)
        project, truncated = await asyncio.to_thread(_list_tree, working, True)
        test_data, _ = await asyncio.to_thread(_list_tree, self.storage.test_data_dir(project_id))
        busy = await self.run_service.has_active_run(project_id, org_id)
        return FileTreeModel(project=project, test_data=test_data, truncated=truncated, busy=busy)

    async def read(self, project_id: str, org_id: str, root: ProjectFileRoot, path: str) -> FileContentModel:
        target = await self.file_path(project_id, org_id, root, path)
        return await asyncio.to_thread(_read, root, _clean(path), target)

    async def save(self, project_id: str, org_id: str, path: str, content: str) -> FileContentModel:
        """Write a text file of the project (new files and folders are created)."""
        await self.project_service.get(project_id, org_id)
        if await self.run_service.has_active_run(project_id, org_id):
            raise ProjectBusyError(ErrorMessages.PROJECT_BUSY)
        data = content.encode("utf-8")
        if len(data) > MAX_EDITABLE_BYTES:
            raise PayloadTooLargeError(ErrorMessages.FILE_TOO_LARGE_TO_EDIT.format(limit_kb=MAX_EDITABLE_BYTES // 1024))
        working = await asyncio.to_thread(self.storage.ensure_working_copy, project_id)
        target = _inside(working, path)
        if any(part in HIDDEN_DIRS for part in Path(_clean(path)).parts):
            raise ValidationError(ErrorMessages.FILE_PATH_INVALID.format(path=path))
        if target.exists() and not target.is_file():
            raise ValidationError(ErrorMessages.FILE_PATH_INVALID.format(path=path))
        await asyncio.to_thread(_write, target, data)
        return await asyncio.to_thread(_read, ProjectFileRoot.PROJECT, _clean(path), target)

    async def run_files(self, run_id: str, org_id: str) -> RunFilesModel:
        """The run's copy of the project as its agents work on it, and what they changed so far (read-only)."""
        run, folder = await self._run_folder(run_id, org_id)
        if folder is None:
            working = await asyncio.to_thread(self.storage.ensure_working_copy, run.project_id)
            project, truncated = await asyncio.to_thread(_list_tree, working, True)
            return RunFilesModel(live=False, project=project, truncated=truncated)
        working = self.storage.working_dir(run.project_id)
        project, truncated = await asyncio.to_thread(_list_tree, folder, True)
        changes = await asyncio.to_thread(compare_projects, working, folder)
        return RunFilesModel(live=True, project=project, changes=changes, truncated=truncated)

    async def read_run_file(self, run_id: str, org_id: str, path: str) -> FileContentModel:
        """One file of the run's copy (of the project once the copy is gone); never editable here."""
        run, folder = await self._run_folder(run_id, org_id)
        if folder is None:
            folder = await asyncio.to_thread(self.storage.ensure_working_copy, run.project_id)
        target = _inside(folder, path)
        if not target.is_file():
            raise NotFoundError(ErrorMessages.FILE_NOT_FOUND)
        file = await asyncio.to_thread(_read, ProjectFileRoot.PROJECT, _clean(path), target)
        return file.model_copy(update={"editable": False})

    async def _run_folder(self, run_id: str, org_id: str) -> "tuple[RunModel, Path | None]":
        """The run, and its copy of the project while the run has one."""
        run = await self.run_service.get(run_id, org_id)
        folder = Path(run.outputs.project_dir) if run.outputs.project_dir else None
        return run, folder if folder is not None and folder.is_dir() else None

    async def file_path(self, project_id: str, org_id: str, root: ProjectFileRoot, path: str) -> Path:
        """The file on disk (for reading or downloading); NotFoundError when it is not there."""
        await self.project_service.get(project_id, org_id)
        if root == ProjectFileRoot.PROJECT:
            folder = await asyncio.to_thread(self.storage.ensure_working_copy, project_id)
        else:
            folder = self.storage.test_data_dir(project_id)
        target = _inside(folder, path)
        if not target.is_file():
            raise NotFoundError(ErrorMessages.FILE_NOT_FOUND)
        return target


def _clean(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _inside(folder: Path, path: str) -> Path:
    """`path` inside `folder`, or a ValidationError (absolute paths, "..", the folder itself)."""
    raw = path.replace("\\", "/")
    if raw.startswith("/") or _DRIVE.match(raw):
        raise ValidationError(ErrorMessages.FILE_PATH_INVALID.format(path=path))
    relative = raw.strip("/")
    root = folder.resolve()
    target = (root / relative).resolve()
    if not relative or not target.is_relative_to(root) or target == root:
        raise ValidationError(ErrorMessages.FILE_PATH_INVALID.format(path=path))
    return target


def _list_tree(folder: Path, mark_tests: bool = False) -> tuple[list[FileNodeModel], bool]:
    """Folders first, then files, by name; hidden tool folders and links left out. `mark_tests`: flag test files."""
    if not folder.is_dir():
        return [], False
    count = 0
    truncated = False

    def walk(current: Path, prefix: str) -> list[FileNodeModel]:
        nonlocal count, truncated
        try:
            entries = sorted(os.scandir(current), key=lambda entry: (not entry.is_dir(follow_symlinks=False), entry.name.lower()))
        except OSError:
            return []
        nodes: list[FileNodeModel] = []
        for entry in entries:
            if count >= MAX_TREE_ENTRIES:
                truncated = True
                break
            if entry.is_symlink():
                continue
            path = f"{prefix}{entry.name}"
            if entry.is_dir():
                if entry.name in HIDDEN_DIRS:
                    continue
                count += 1
                nodes.append(FileNodeModel(name=entry.name, path=path, type="folder", children=walk(Path(entry.path), f"{path}/")))
            elif entry.is_file() and not is_in_flight_write(entry.name):
                count += 1
                nodes.append(FileNodeModel(
                    name=entry.name, path=path, type="file", size=entry.stat().st_size,
                    is_test=mark_tests and is_test_file(path, partial(read_small_text, Path(entry.path))),
                ))
        return nodes

    return walk(folder, ""), truncated


def _read(root: ProjectFileRoot, path: str, target: Path) -> FileContentModel:
    size = target.stat().st_size
    content: str | None = None
    if size <= MAX_EDITABLE_BYTES:
        data = target.read_bytes()
        if b"\0" not in data:
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                content = None
    binary = content is None
    return FileContentModel(
        root=root,
        path=path,
        size=size,
        binary=binary,
        editable=root == ProjectFileRoot.PROJECT and not binary,
        content=content,
    )


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".ngauto-partial")
    partial.write_bytes(data)
    partial.replace(target)
