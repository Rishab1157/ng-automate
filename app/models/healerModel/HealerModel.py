from pydantic import BaseModel, Field

from app.models.testRunModel import FailureKind


class GuardViolationModel(BaseModel):
    """An edit the healer is not allowed to make (it would hide a failure instead of fixing it)."""

    file: str  # repo-relative
    rule: str  # e.g. "assertion_removed", "test_skipped"
    line: str  # the offending removed/added line, stripped and truncated


class PlaybookModel(BaseModel):
    """Known fix for a common technical failure, given to the healer only when it matches."""

    key: str
    title: str
    applies_to: list[FailureKind]
    # Case-insensitive regular expressions matched against the run output.
    patterns: list[str]
    guidance: str
    priority: int = Field(default=0, description="Higher wins when several playbooks match")
