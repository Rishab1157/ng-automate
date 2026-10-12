from pydantic import BaseModel, Field

from app.models.healerModel import FileChangeModel, GuardViolationModel


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
    files: list[FileChangeModel] = Field(default_factory=list)  # files written or changed, kept after the guards
    generated_cases: list[str] = Field(default_factory=list)  # ids of the cases given to the agent
    skipped_cases: list[SkippedCaseModel] = Field(default_factory=list)
    # How to run only the generated tests (a test selector), or None when they could not be told apart.
    selector: str | None = None
    violations: list[GuardViolationModel] = Field(default_factory=list)
    reverted_files: list[str] = Field(default_factory=list)
    # Locators the generator assumed and marked in the files it wrote (NGAUTO-UNVERIFIED).
    unverified_locators: list[UnverifiedLocatorModel] = Field(default_factory=list)
    # Why the generator stopped early (model unreachable, out of time, ...); its files were still checked.
    error: str | None = None
