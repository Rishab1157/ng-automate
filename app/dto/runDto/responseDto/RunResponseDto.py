from datetime import datetime

from pydantic import BaseModel

from app.models.runModel import RunMode, RunModel, RunScope, RunStage, RunStatus, TestReportModel
from app.models.testGeneratorModel import GenerationOutcomeModel


class RunErrorDTO(BaseModel):
    code: str
    message: str


class RunResponseDTO(BaseModel):
    """What callers see of a run. The host folder and the sandbox container stay internal."""

    id: str
    org_id: str
    project_id: str
    created_by: str
    model_connection_id: str | None = None
    mode: RunMode
    test_selector: str | None = None
    test_data_id: str | None = None
    run_scope: RunScope = RunScope.GENERATED
    status: RunStatus
    stage: RunStage
    error: RunErrorDTO | None = None
    profile_id: str | None = None
    # Generate mode: the tests the generator wrote (with diffs) and the cases it left out.
    generation: GenerationOutcomeModel | None = None
    # Test and generate modes: the outcome once the tests ran. Details per attempt: GET /run/{id}/tests.
    test_report: TestReportModel | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_model(cls, run: RunModel) -> "RunResponseDTO":
        return cls(
            id=run.id,
            org_id=run.org_id,
            project_id=run.project_id,
            created_by=run.created_by,
            model_connection_id=run.model_connection_id,
            mode=run.mode,
            test_selector=run.test_selector,
            test_data_id=run.test_data_id,
            run_scope=run.run_scope,
            status=run.status,
            stage=run.stage,
            error=RunErrorDTO(code=run.error.code, message=run.error.message) if run.error else None,
            profile_id=run.outputs.profile_id,
            generation=run.outputs.generation,
            test_report=run.outputs.test_report,
            created_at=run.created_at,
            updated_at=run.updated_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
