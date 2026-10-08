from datetime import datetime

from pydantic import BaseModel

from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataStatus, TestDataUploadModel


class TestDataSummaryResponseDTO(BaseModel):
    """A test-data source without its cases (lists)."""

    __test__ = False  # not a pytest test class

    id: str
    project_id: str
    filename: str
    source_format: TestDataFormat
    # processing: an LLM is still reading a free-form source; ready: the cases can be reviewed and used; failed.
    status: TestDataStatus
    case_count: int
    # e.g. "case LOGIN-2 step 3: click needs a locator". Such cases are not generated.
    warnings: list[str]
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, upload: TestDataUploadModel) -> "TestDataSummaryResponseDTO":
        return cls(
            id=upload.id,
            project_id=upload.project_id,
            filename=upload.filename,
            source_format=upload.source_format,
            status=upload.status,
            case_count=upload.case_count,
            warnings=upload.data_set.warnings if upload.data_set else [],
            error=upload.error,
            created_at=upload.created_at,
            updated_at=upload.updated_at,
        )


class TestDataResponseDTO(TestDataSummaryResponseDTO):
    cases: list[TestCaseSpecModel]

    @classmethod
    def from_model(cls, upload: TestDataUploadModel) -> "TestDataResponseDTO":
        summary = TestDataSummaryResponseDTO.from_model(upload)
        return cls(**summary.model_dump(), cases=upload.data_set.cases if upload.data_set else [])
