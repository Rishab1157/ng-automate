from enum import Enum

from pydantic import BaseModel, Field

from app.models.healerModel import FileChangeModel


class ProjectFileRoot(str, Enum):
    """The two folders the editor shows. They live side by side: test data is never inside the project."""

    PROJECT = "project"  # the project's editable copy (runs start from it)
    TEST_DATA = "test-data"  # the original test-data files (read-only here)


class FileNodeModel(BaseModel):
    name: str
    path: str  # relative to its root, "/"-separated
    type: str  # "file" | "folder"
    size: int | None = None  # bytes, files only
    # A test file the project's test runner picks up (project files only): users can choose to run just these.
    is_test: bool = False
    children: list["FileNodeModel"] = Field(default_factory=list)


class FileTreeModel(BaseModel):
    project: list[FileNodeModel] = Field(default_factory=list)
    test_data: list[FileNodeModel] = Field(default_factory=list)
    # The project has more entries than the editor lists; the rest are left out.
    truncated: bool = False
    # A run is using the project: its files are read-only until the run has finished.
    busy: bool = False


class RunFilesModel(BaseModel):
    """A run's copy of the project while the run works on it: the files and what its agents changed so far."""

    # True while the run's own copy is shown; False once it is gone (its changes are then in the project).
    live: bool
    project: list[FileNodeModel] = Field(default_factory=list)
    changes: list[FileChangeModel] = Field(default_factory=list)
    truncated: bool = False


class FileContentModel(BaseModel):
    root: ProjectFileRoot
    path: str
    size: int
    binary: bool  # not UTF-8 text, or too large to show: download it instead
    editable: bool
    content: str | None = None  # the text, when it can be shown
