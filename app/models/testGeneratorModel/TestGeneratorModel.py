from pydantic import BaseModel, Field

from app.models.healerModel import FileChangeModel, GuardViolationModel
from app.models.tokenUsageModel import TokenUsageModel


class SkippedCaseModel(BaseModel):
    """A test case the generator left out, and why (it has no steps, or the generator stopped before it)."""

    case_id: str
    reason: str


class UnverifiedLocatorModel(BaseModel):
    """A locator the generator assumed (not in the test data, not in the project): a person should check it."""

    path: str  # repo-relative file
    line: int  # 1-based line of the marker
    code: str  # the code line with the locator


class GenerationOutcomeModel(BaseModel):
    summary: str  # the agent's own short explanation of what it wrote
    files: list[FileChangeModel] = Field(default_factory=list)  # every file it added, changed or deleted (all kept)
    generated_cases: list[str] = Field(default_factory=list)  # ids of the cases given to the agent
    skipped_cases: list[SkippedCaseModel] = Field(default_factory=list)
    violations: list[GuardViolationModel] = Field(default_factory=list)  # risky changes, kept, for the user to review
    # Whether the tests build after the generator's work (its stack's compile/collect step). None: not checked (the
    # stack has no such step, or the project did not build before the generator started).
    builds: bool | None = None
    # Locators the generator assumed and marked in the files it wrote (NGAUTO-UNVERIFIED).
    unverified_locators: list[UnverifiedLocatorModel] = Field(default_factory=list)
    # Why the generator stopped early (model unreachable, out of time, ...); its files are still checked and kept.
    error: str | None = None
    # Tokens the generator's LLM calls used; None when they could not be read.
    token_usage: TokenUsageModel | None = None
