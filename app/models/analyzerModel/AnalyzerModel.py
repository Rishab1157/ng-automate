"""What the analyzer learns about a project.

Two kinds of knowledge, kept apart on purpose:
- FactSheetModel: found by plain code (file names, dependencies). Same answer every time.
- AnalyzerFindingsModel: judged by the LLM (purpose, architecture). Must cite evidence.
The LLM receives the fact sheet as given truth and cannot overwrite it.
"""

from enum import Enum

from pydantic import BaseModel, Field


class FactSource(str, Enum):
    CODE = "code"
    LLM = "llm"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FactModel(BaseModel):
    """One answer plus the proof behind it. value is None when nothing was found."""

    value: str | list[str] | None = None
    source: FactSource
    evidence: list[str] = Field(default_factory=list, description="Repo-relative file paths (optionally path:line)")
    confidence: Confidence


class FactSheetModel(BaseModel):
    """Collected by code only (no LLM)."""

    total_files: int
    # Language -> number of source files, e.g. {"Java": 120, "Gherkin": 14}
    language_files: dict[str, int]
    primary_language: FactModel
    build_tool: FactModel
    test_frameworks: FactModel
    automation_tools: FactModel
    bdd_tool: FactModel
    # Repo-relative paths of files that identify the stack (pom.xml, testng.xml, package.json, ...)
    marker_files: list[str]
    # Repo-relative folders that hold tests
    test_dirs: list[str]
    feature_file_count: int
    # First levels of the folder tree, repo-relative, directories end with "/"
    top_level_tree: list[str]


class ImportantPathModel(BaseModel):
    path: str
    role: str = Field(description="e.g. page objects, step definitions, test data, config")


class AnalyzerFindingsModel(BaseModel):
    """Judged by the LLM inside the sandbox. Every answer carries evidence."""

    project_summary: FactModel
    architecture_pattern: FactModel
    test_command: FactModel
    reporting_tools: FactModel
    important_paths: list[ImportantPathModel] = Field(default_factory=list)
    # Things the agent could not decide: shown to the user instead of guessed.
    open_questions: list[str] = Field(default_factory=list)
