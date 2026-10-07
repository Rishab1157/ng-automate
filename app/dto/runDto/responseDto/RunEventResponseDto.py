from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.models.runModel import RunEventLevel, RunEventModel, RunEventType


class RunEventResponseDTO(BaseModel):
    seq: int
    type: RunEventType
    level: RunEventLevel
    message: str
    data: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_model(cls, event: RunEventModel) -> "RunEventResponseDTO":
        return cls(
            seq=event.seq,
            type=event.type,
            level=event.level,
            message=event.message,
            data=event.data,
            created_at=event.created_at,
        )
