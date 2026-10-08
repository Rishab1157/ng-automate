from pydantic import BaseModel, Field

from app.models.healerModel import FileChangeModel, GuardViolationModel


class SkippedCaseModel(BaseModel):
    """A test case the generator left out, and why (e.g. a step has no locator: it is never guessed)."""

    case_id: str
    reason: str


class GenerationOutcomeModel(BaseModel):
    summary: str  # the agent's own short explanation of what it wrote
    files: list[FileChangeModel] = Field(default_factory=list)  # files written or changed, kept after the guards
    generated_cases: list[str] = Field(default_factory=list)  # ids of the cases given to the agent
    skipped_cases: list[SkippedCaseModel] = Field(default_factory=list)
    # How to run only the generated tests (a test selector), or None when they could not be told apart.
    selector: str | None = None
    violations: list[GuardViolationModel] = Field(default_factory=list)
    reverted_files: list[str] = Field(default_factory=list)
    # Why the generator stopped early (model unreachable, out of time, ...); its files were still checked.
    error: str | None = None
