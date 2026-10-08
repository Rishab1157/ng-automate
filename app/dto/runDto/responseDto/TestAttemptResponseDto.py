from pydantic import BaseModel

from app.models.healerModel import HealOutcomeModel
from app.models.runModel import TestAttemptModel
from app.models.testRunModel import TestRunResultModel


class TestAttemptResponseDTO(BaseModel):
    """One run of the tests (results, report cases, end of the output) and the healer's fix after it, with diffs."""

    __test__ = False  # not a pytest test class

    number: int
    result: TestRunResultModel
    heal: HealOutcomeModel | None = None

    @classmethod
    def from_model(cls, attempt: TestAttemptModel) -> "TestAttemptResponseDTO":
        return cls(number=attempt.number, result=attempt.result, heal=attempt.heal)
