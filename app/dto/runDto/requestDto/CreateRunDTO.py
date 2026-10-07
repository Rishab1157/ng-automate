from pydantic import BaseModel, Field

from app.dto.common import object_id_validator


class CreateRunDTO(BaseModel):
    project_id: str = Field(description="The project to analyze")

    _validate_project_id = object_id_validator("project_id")
