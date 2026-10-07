from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class ArchiveDbModel(BaseModel):
    size_bytes: int
    sha256: str
    file_count: int


class GitOriginDbModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection_id: ObjectId
    repo_url: str
    branch: str
    commit_sha: str


class ProjectCreateDbModel(BaseModel):
    """A new document in `projects`. The archive itself lives in storage, keyed by _id."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    org_id: ObjectId
    created_by: ObjectId
    name: str
    source: str
    status: str
    archive: ArchiveDbModel
    git: GitOriginDbModel | None = None
    created_at: datetime
    updated_at: datetime
