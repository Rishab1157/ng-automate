from bson import ObjectId
from pydantic import BaseModel, Field

from pydantic import field_validator

from app.dto.common import object_id_validator
from app.models.runModel import RunMode


class CreateRunDTO(BaseModel):
    project_id: str = Field(description="The project to analyze")
    mode: RunMode = Field(
        RunMode.ANALYZE,
        description="analyze: build the project profile. test: also run the project's tests and let the healer fix "
        "technical failures (build, dependencies, environment, locators, errors in the test code). generate: only write new tests from uploaded test data (they are "
        "saved to the project for review; run them with a test run and their test_files). Test and generate runs "
        "reuse the project's latest profile when it has one.",
    )
    test_selector: str | None = Field(
        None,
        max_length=200,
        description="Test mode only: which tests to run, e.g. a test class (LoginTest), a pytest path or a tag "
        "expression (@smoke). Leave empty to run all tests.",
    )
    test_files: list[str] | None = Field(
        None,
        max_length=200,
        description="Test mode only, instead of test_selector: the test files to run, as project-relative paths "
        "from the file tree (its is_test files), e.g. src/test/java/com/acme/LoginTest.java or a .feature file.",
    )

    test_data_id: str | None = Field(
        None, description="Generate mode only: the test data to write tests from (POST /project/{id}/test-data)."
    )

    _validate_project_id = object_id_validator("project_id")

    @field_validator("test_data_id")
    @classmethod
    def _validate_test_data_id(cls, value: str | None) -> str | None:
        if value is not None and not ObjectId.is_valid(value):
            raise ValueError("test_data_id is not a valid id")
        return value
