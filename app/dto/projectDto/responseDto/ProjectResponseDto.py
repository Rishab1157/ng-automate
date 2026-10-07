from datetime import datetime

from pydantic import BaseModel

from app.models.projectModel import ProjectModel, ProjectSource, ProjectStatus


class ProjectArchiveDTO(BaseModel):
    size_bytes: int
    sha256: str
    file_count: int


class ProjectGitOriginDTO(BaseModel):
    connection_id: str
    repo_url: str
    branch: str
    commit_sha: str


class ProjectResponseDTO(BaseModel):
    id: str
    org_id: str
    created_by: str
    name: str
    source: ProjectSource
    status: ProjectStatus
    archive: ProjectArchiveDTO
    git: ProjectGitOriginDTO | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, project: ProjectModel) -> "ProjectResponseDTO":
        return cls.model_validate(project.model_dump())
