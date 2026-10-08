from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class HealMemoryResult(str, Enum):
    """What came of a heal, as the next test run (or the end of the test phase) showed it."""

    FIXED = "fixed"  # the tests passed
    CHANGED = "changed"  # a different failure came next: the problem the healer worked on is gone
    NOT_FIXED = "not_fixed"  # the same failure came back
    BLOCKED = "blocked"  # the healer proved that the environment stops it
    UNFINISHED = "unfinished"  # the healer stopped (out of time, model failure) before the tests ran again


SUCCESSFUL_RESULTS = frozenset({HealMemoryResult.FIXED, HealMemoryResult.CHANGED})


class HealMemoryModel(BaseModel):
    """One earlier heal of an organization: the problem, what the healer tried, and what came of it."""

    id: str  # Qdrant point id
    org_id: str
    project_id: str
    run_id: str
    attempt: int  # the test run whose failure was healed
    created_at: datetime
    stack: str  # e.g. "Java · Maven · TestNG · Selenium"
    test_command: str
    failure_kind: str
    failure_reason: str
    failure_evidence: list[str] = Field(default_factory=list)
    problem: str  # the text the memory was found by (embedded)
    fix_summary: str  # the healer's own summary
    changed_files: list[str] = Field(default_factory=list)  # kept changes
    reverted_files: list[str] = Field(default_factory=list)  # edits rolled back by the guards
    diff: str = ""  # the kept changes, truncated
    blocker: str | None = None
    result: HealMemoryResult
    succeeded: bool
    next_run: str | None = None  # one line about the next test run
    score: float | None = None  # similarity to the current problem, when found by a search
