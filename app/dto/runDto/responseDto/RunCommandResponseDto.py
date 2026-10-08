from datetime import datetime

from pydantic import BaseModel

from app.models.runCommandModel import RunCommandModel, RunCommandStatus, RunCommandType


class RunCommandResponseDTO(BaseModel):
    id: str
    run_id: str
    type: RunCommandType
    text: str | None = None
    status: RunCommandStatus
    reason: str | None = None
    created_at: datetime
    applied_at: datetime | None = None

    @classmethod
    def from_model(cls, command: RunCommandModel) -> "RunCommandResponseDTO":
        return cls(
            id=command.id,
            run_id=command.run_id,
            type=command.type,
            text=command.text,
            status=command.status,
            reason=command.reason,
            created_at=command.created_at,
            applied_at=command.applied_at,
        )
