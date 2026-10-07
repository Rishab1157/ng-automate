from datetime import datetime

from pydantic import BaseModel

from app.models.runModel import RunModel, RunStage, RunStatus


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
    status: RunStatus
    stage: RunStage
    error: RunErrorDTO | None = None
    profile_id: str | None = None
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
            status=run.status,
            stage=run.stage,
            error=RunErrorDTO(code=run.error.code, message=run.error.message) if run.error else None,
            profile_id=run.outputs.profile_id,
            created_at=run.created_at,
            updated_at=run.updated_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
