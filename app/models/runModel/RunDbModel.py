from datetime import datetime
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class RunCreateDbModel(BaseModel):
    """A new document in `runs`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    org_id: ObjectId
    project_id: ObjectId
    created_by: ObjectId
    model_connection_id: ObjectId | None = None
    status: str
    stage: str
    # Same shape as RunOutputsModel; profile_id is stored as an ObjectId once set.
    outputs: dict[str, Any] = Field(default_factory=dict)
    sandbox_container_id: str | None = None
    error: dict[str, str] | None = None
    # Counter for run_events.seq, incremented atomically per event.
    event_seq: int = 0
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunEventCreateDbModel(BaseModel):
    """A new document in `run_events`. (run_id, seq) is unique."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    run_id: ObjectId
    org_id: ObjectId
    seq: int
    type: str
    level: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
