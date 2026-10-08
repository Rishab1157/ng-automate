from bson import ObjectId
from pydantic import BaseModel, Field

from pydantic import field_validator

from app.dto.common import object_id_validator
from app.models.runModel import RunMode, RunScope


class CreateRunDTO(BaseModel):
    project_id: str = Field(description="The project to analyze")
    mode: RunMode = Field(
        RunMode.ANALYZE,
        description="analyze: build the project profile. test: also run the project's tests and let the healer fix "
        "build, dependency and environment failures. generate: write new tests from uploaded test data, then run and "
        "heal just those. Test and generate runs reuse the project's latest profile when it has one.",
    )
    test_selector: str | None = Field(
        None,
        max_length=200,
        description="Test mode only: which tests to run, e.g. a test class (LoginTest), a pytest path or a tag "
        "expression (@smoke). Leave empty to run all tests.",
    )

    test_data_id: str | None = Field(
        None, description="Generate mode only: the test data to write tests from (POST /project/{id}/test-data)."
    )
    run_scope: RunScope = Field(
        RunScope.GENERATED,
        description="Generate mode only: generated = run only the tests written from the test data; "
        "all = run the project's whole suite.",
    )

    _validate_project_id = object_id_validator("project_id")

    @field_validator("test_data_id")
    @classmethod
    def _validate_test_data_id(cls, value: str | None) -> str | None:
        if value is not None and not ObjectId.is_valid(value):
            raise ValueError("test_data_id is not a valid id")
        return value
