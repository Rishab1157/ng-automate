from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class RunCommandCreateDbModel(BaseModel):
    """A new document in `run_commands`. Workers read the pending ones of their run, oldest first."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    run_id: ObjectId
    org_id: ObjectId
    created_by: ObjectId
    type: str
    text: str | None = None
    status: str
    reason: str | None = None
    created_at: datetime
    applied_at: datetime | None = None
