"""Creating and reading projects.

Uploaded or fetched from Git, every project ends up the same way: an archive in storage
(never changed afterwards) plus one document in `projects`.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

from bson import ObjectId
from pymongo import DESCENDING

from app.core.exceptions import ErrorMessages, NotFoundError
from app.models.projectModel import GitOriginModel, ProjectMapper, ProjectModel, ProjectSource
from app.projections.projectProjection import PROJECT_DETAIL_PROJECTION
from app.repositories.projectRepository import ProjectRepository
from app.services.gitConnectionService import GitConnectionService, GitFetchService
from app.utils.ArchiveUtils import inspect_archive

from .ProjectStorageService import ProjectStorageService

logger = logging.getLogger(__name__)

DEFAULT_LIST_LIMIT = 100


class ProjectService:
    def __init__(self) -> None:
        self.project_repo = ProjectRepository()
        self.storage = ProjectStorageService()
        self.git_connection_service = GitConnectionService()
        self.git_fetch_service = GitFetchService()

    async def create_from_upload(
        self,
        chunks: AsyncIterator[bytes],
        *,
        filename: str | None,
        name: str | None,
        org_id: str,
        user_id: str,
    ) -> ProjectModel:
        project_id = ObjectId()
        with self._discard_on_error(project_id):
            path = await self.storage.save_archive(str(project_id), chunks)
            archive = await asyncio.to_thread(inspect_archive, path, self.storage.max_bytes)
            db_model = ProjectMapper.to_create_db_model(
                project_id=project_id,
                org_id=org_id,
                user_id=user_id,
                name=name or _name_from_filename(filename),
                source=ProjectSource.UPLOAD,
                archive=archive,
            )
            await self.project_repo.insert_one(db_model.model_dump(by_alias=True))

        logger.info("Project %s created from upload (org %s, %d files)", project_id, org_id, archive.file_count)
        return ProjectMapper.to_model(db_model.model_dump(by_alias=True))

    async def create_from_git(
        self,
        *,
        git_connection_id: str,
        branch: str | None,
        name: str | None,
        org_id: str,
        user_id: str,
    ) -> ProjectModel:
        source = await self.git_connection_service.get_clone_source(git_connection_id, org_id)
        project_id = ObjectId()
        with self._discard_on_error(project_id):
            path = self.storage.archive_path(str(project_id))
            fetched = await asyncio.to_thread(
                self.git_fetch_service.fetch_to_archive, source, branch or source.branch, path, self.storage.max_bytes
            )
            archive = await asyncio.to_thread(inspect_archive, path, self.storage.max_bytes)
            db_model = ProjectMapper.to_create_db_model(
                project_id=project_id,
                org_id=org_id,
                user_id=user_id,
                name=name or source.userdefined_name or _name_from_repo_url(source.repo_url),
                source=ProjectSource.GIT,
                archive=archive,
                git=GitOriginModel(
                    connection_id=source.id,
                    repo_url=source.repo_url,
                    branch=fetched.branch,
                    commit_sha=fetched.commit_sha,
                ),
            )
            await self.project_repo.insert_one(db_model.model_dump(by_alias=True))

        logger.info("Project %s created from git %s@%s (org %s)", project_id, source.repo_url, fetched.commit_sha[:8], org_id)
        return ProjectMapper.to_model(db_model.model_dump(by_alias=True))

    async def get(self, project_id: str, org_id: str) -> ProjectModel:
        if not ObjectId.is_valid(project_id):
            raise NotFoundError(ErrorMessages.PROJECT_NOT_FOUND)
        doc = await self.project_repo.find_one(
            {"_id": ObjectId(project_id), "org_id": ObjectId(org_id)}, PROJECT_DETAIL_PROJECTION
        )
        if doc is None:
            raise NotFoundError(ErrorMessages.PROJECT_NOT_FOUND)
        return ProjectMapper.to_model(doc)

    async def get_all(self, org_id: str, limit: int = DEFAULT_LIST_LIMIT) -> list[ProjectModel]:
        docs = await self.project_repo.find(
            {"org_id": ObjectId(org_id)}, PROJECT_DETAIL_PROJECTION, sort=[("created_at", DESCENDING)], limit=limit
        )
        return [ProjectMapper.to_model(doc) for doc in docs]

    @contextmanager
    def _discard_on_error(self, project_id: ObjectId) -> Iterator[None]:
        """Leave nothing in storage if creation fails or the client goes away halfway."""
        try:
            yield
        except BaseException:
            self.storage.delete(str(project_id))
            raise


def _name_from_filename(filename: str | None) -> str:
    return (filename or "project").removesuffix(".zip")


def _name_from_repo_url(repo_url: str) -> str:
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
