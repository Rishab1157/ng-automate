from pydantic import BaseModel

from app.models.testDataModel import TestDataContentModel


class TestDataContentResponseDTO(BaseModel):
    """A test-data file's text as it is stored, for the editor. `content` is null (and `editable` false) when the
    file is too large or not text; `reason` says why."""

    __test__ = False  # not a pytest test class

    test_data_id: str
    filename: str
    content: str | None = None
    editable: bool
    size_bytes: int
    reason: str | None = None

    @classmethod
    def from_model(cls, content: TestDataContentModel) -> "TestDataContentResponseDTO":
        return cls(**content.model_dump())
