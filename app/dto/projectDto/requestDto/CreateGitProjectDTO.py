from pydantic import BaseModel, Field

from app.dto.common import object_id_validator


class CreateGitProjectDTO(BaseModel):
    git_connection_id: str = Field(description="QXcel git connection enabled for NG Automate in your organization")
    branch: str | None = Field(None, min_length=1, max_length=255, description="Defaults to the connection's branch")
    name: str | None = Field(None, min_length=1, max_length=200, description="Defaults to the connection's name")

    _validate_git_connection_id = object_id_validator("git_connection_id")
