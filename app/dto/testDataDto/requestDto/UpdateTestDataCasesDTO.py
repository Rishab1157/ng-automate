from pydantic import BaseModel, Field

from app.models.testDataModel import TestCaseSpecModel
from app.utils.TestDataParser import MAX_CASES


class UpdateTestDataCasesDTO(BaseModel):
    """The reviewed test cases that replace the ones read from the source."""

    cases: list[TestCaseSpecModel] = Field(min_length=1, max_length=MAX_CASES)
