from datetime import datetime
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class ProjectProfileCreateDbModel(BaseModel):
    """A new document in `project_profiles`. A project keeps every profile; the newest one is current."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    org_id: ObjectId
    project_id: ObjectId
    run_id: ObjectId
    fact_sheet: dict[str, Any]
    findings: dict[str, Any]
    llm_model: str
    created_at: datetime
