from datetime import datetime
from typing import Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class TestDataCreateDbModel(BaseModel):
    """A new document in `test_data`: one uploaded test-data source of a project."""

    __test__ = False  # not a pytest test class

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    org_id: ObjectId
    project_id: ObjectId
    created_by: ObjectId
    filename: str
    source_format: str
    status: str
    raw_source: str | None = None
    model_connection_id: ObjectId | None = None
    # TestDataSetModel as JSON, once ready.
    data_set: dict[str, Any] | None = None
    case_count: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime
