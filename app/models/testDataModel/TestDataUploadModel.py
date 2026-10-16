from datetime import datetime

from pydantic import BaseModel

from app.models.tokenUsageModel import TokenUsageModel

from .TestDataModel import TestDataFormat, TestDataSetModel, TestDataStatus


class TestDataUploadModel(BaseModel):
    """One uploaded test-data source of a project. The test generator reads its cases once it is ready."""

    __test__ = False  # not a pytest test class

    id: str
    org_id: str
    project_id: str
    created_by: str
    filename: str
    source_format: TestDataFormat
    status: TestDataStatus
    # Free-form JSON or text, kept for the LLM (and a retry). None for our own JSON/CSV format.
    raw_source: str | None = None
    # The model that reads a free-form source; None means the default model from .env.
    model_connection_id: str | None = None
    data_set: TestDataSetModel | None = None  # set once ready
    case_count: int
    error: str | None = None  # why reading the source failed
    # Tokens the LLM used to read a free-form source (the last reading); None for our own format.
    token_usage: TokenUsageModel | None = None
    # Counts the editor's saves: a reading of an older version of the source is not written over a newer one.
    revision: int = 0
    created_at: datetime
    updated_at: datetime
