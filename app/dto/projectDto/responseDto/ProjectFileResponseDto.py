from pydantic import BaseModel, Field

from app.models.healerModel import FileChangeModel
from app.models.projectFileModel import FileContentModel, FileTreeModel, ProjectFileRoot, RunFilesModel


class FileNodeDTO(BaseModel):
    name: str
    path: str
    type: str
    size: int | None = None
    is_test: bool = False
    children: list["FileNodeDTO"] = Field(default_factory=list)


class ProjectFileTreeResponseDTO(BaseModel):
    project: list[FileNodeDTO]
    test_data: list[FileNodeDTO]
    truncated: bool
    busy: bool

    @classmethod
    def from_model(cls, tree: FileTreeModel) -> "ProjectFileTreeResponseDTO":
        return cls.model_validate(tree.model_dump())


class RunFilesResponseDTO(BaseModel):
    live: bool
    project: list[FileNodeDTO]
    changes: list[FileChangeModel]
    truncated: bool

    @classmethod
    def from_model(cls, files: RunFilesModel) -> "RunFilesResponseDTO":
        return cls.model_validate(files.model_dump())


class ProjectFileContentResponseDTO(BaseModel):
    root: ProjectFileRoot
    path: str
    size: int
    binary: bool
    editable: bool
    content: str | None = None

    @classmethod
    def from_model(cls, file: FileContentModel) -> "ProjectFileContentResponseDTO":
        return cls.model_validate(file.model_dump())

