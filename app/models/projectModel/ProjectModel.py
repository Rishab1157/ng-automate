from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ProjectSource(str, Enum):
    UPLOAD = "upload"
    GIT = "git"


class ProjectStatus(str, Enum):
    READY = "ready"


class ArchiveModel(BaseModel):
    """Facts about the project's saved source.zip."""

    size_bytes: int
    sha256: str
    file_count: int


class GitOriginModel(BaseModel):
    """Where a git project came from. The token is never stored with the project."""

    connection_id: str
    repo_url: str
    branch: str
    commit_sha: str


class ProjectModel(BaseModel):
    id: str
    org_id: str
    created_by: str
    name: str
    source: ProjectSource
    status: ProjectStatus
    archive: ArchiveModel
    git: GitOriginModel | None = None
    created_at: datetime
    updated_at: datetime
