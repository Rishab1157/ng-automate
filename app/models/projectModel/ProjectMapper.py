from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from .ProjectDbModel import ArchiveDbModel, GitOriginDbModel, ProjectCreateDbModel
from .ProjectModel import ArchiveModel, GitOriginModel, ProjectModel, ProjectSource, ProjectStatus


class ProjectMapper:
    @staticmethod
    def to_create_db_model(
        *,
        project_id: ObjectId,
        org_id: str,
        user_id: str,
        name: str,
        source: ProjectSource,
        archive: ArchiveModel,
        git: GitOriginModel | None = None,
    ) -> ProjectCreateDbModel:
        now = datetime.now(UTC)
        return ProjectCreateDbModel(
            id=project_id,
            org_id=ObjectId(org_id),
            created_by=ObjectId(user_id),
            name=name,
            source=source.value,
            status=ProjectStatus.READY.value,
            archive=ArchiveDbModel(**archive.model_dump()),
            git=GitOriginDbModel(
                connection_id=ObjectId(git.connection_id),
                repo_url=git.repo_url,
                branch=git.branch,
                commit_sha=git.commit_sha,
            ) if git else None,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> ProjectModel:
        git = doc.get("git")
        return ProjectModel(
            id=str(doc["_id"]),
            org_id=str(doc["org_id"]),
            created_by=str(doc["created_by"]),
            name=doc["name"],
            source=ProjectSource(doc["source"]),
            status=ProjectStatus(doc["status"]),
            archive=ArchiveModel(**doc["archive"]),
            git=GitOriginModel(
                connection_id=str(git["connection_id"]),
                repo_url=git["repo_url"],
                branch=git["branch"],
                commit_sha=git["commit_sha"],
            ) if git else None,
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )
