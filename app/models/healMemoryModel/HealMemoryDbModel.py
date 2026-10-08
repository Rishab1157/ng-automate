from datetime import datetime

from pydantic import BaseModel, Field


class HealMemoryDbModel(BaseModel):
    """The payload of one point in the Qdrant collection (the vector is the embedding of `problem`).

    org_id, run_id and embedding_model are indexed: every search filters on the organization and the model, and
    leaves out the current run (its earlier fixes are already in the healer's prompt).
    """

    org_id: str
    project_id: str
    run_id: str
    attempt: int
    created_at: datetime
    embedding_model: str  # vectors of different models cannot be compared
    stack: str
    test_command: str
    failure_kind: str
    failure_reason: str
    failure_evidence: list[str] = Field(default_factory=list)
    problem: str
    fix_summary: str
    changed_files: list[str] = Field(default_factory=list)
    reverted_files: list[str] = Field(default_factory=list)
    diff: str = ""
    blocker: str | None = None
    result: str
    succeeded: bool
    next_run: str | None = None
