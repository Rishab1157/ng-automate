from pydantic import BaseModel, Field


class GuardViolationModel(BaseModel):
    """A risky edit by an agent (it may hide a failure instead of fixing it): kept, and reported for the user."""

    file: str  # repo-relative
    rule: str  # e.g. "assertion_removed", "test_skipped"
    line: str  # the offending removed/added line, stripped and truncated


class FileChangeModel(BaseModel):
    """One file the healer (or test generator) changed in the run's project copy."""

    path: str  # repo-relative
    change: str  # "added" | "modified" | "deleted"
    diff: str = ""  # unified diff, truncated


class HealOutcomeModel(BaseModel):
    summary: str  # the agent's own short explanation of what it changed and why
    changes: list[FileChangeModel] = Field(default_factory=list)  # every change it made (all kept)
    violations: list[GuardViolationModel] = Field(default_factory=list)  # risky changes, kept, for the user to review
    # Why the healer stopped early (model unreachable, out of time, ...); its changes are still checked and kept.
    error: str | None = None
    # The agent's blocker (reason and the command output that proves it): nothing it can do gets past it.
    blocker: str | None = None
    # The agent said it is blocked, but the error it quoted is not in the output of any command it ran.
    blocker_rejected: bool = False
    # Time the healer worked, without paused time: counts against the run's healing budget.
    seconds: float = 0.0
