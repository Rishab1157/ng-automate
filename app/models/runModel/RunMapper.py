from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from .RunDbModel import RunCreateDbModel, RunEventCreateDbModel
from .RunModel import (
    RunErrorModel,
    RunEventLevel,
    RunEventModel,
    RunEventType,
    RunModel,
    RunOutputsModel,
    RunStage,
    RunStatus,
)


class RunMapper:
    @staticmethod
    def to_create_db_model(
        *, org_id: str, project_id: str, user_id: str, model_connection_id: str | None
    ) -> RunCreateDbModel:
        now = datetime.now(UTC)
        return RunCreateDbModel(
            org_id=ObjectId(org_id),
            project_id=ObjectId(project_id),
            created_by=ObjectId(user_id),
            model_connection_id=ObjectId(model_connection_id) if model_connection_id else None,
            status=RunStatus.QUEUED.value,
            stage=RunStage.QUEUED.value,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> RunModel:
        outputs = dict(doc.get("outputs") or {})
        if outputs.get("profile_id") is not None:
            outputs["profile_id"] = str(outputs["profile_id"])
        model_connection_id = doc.get("model_connection_id")
        error = doc.get("error")
        return RunModel(
            id=str(doc["_id"]),
            org_id=str(doc["org_id"]),
            project_id=str(doc["project_id"]),
            created_by=str(doc["created_by"]),
            model_connection_id=str(model_connection_id) if model_connection_id else None,
            status=RunStatus(doc["status"]),
            stage=RunStage(doc["stage"]),
            outputs=RunOutputsModel.model_validate(outputs),
            sandbox_container_id=doc.get("sandbox_container_id"),
            error=RunErrorModel(**error) if error else None,
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            started_at=doc.get("started_at"),
            finished_at=doc.get("finished_at"),
        )

    @staticmethod
    def to_event_create_db_model(
        *,
        run_id: str,
        org_id: str,
        seq: int,
        type: RunEventType,
        level: RunEventLevel,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> RunEventCreateDbModel:
        return RunEventCreateDbModel(
            run_id=ObjectId(run_id),
            org_id=ObjectId(org_id),
            seq=seq,
            type=type.value,
            level=level.value,
            message=message,
            data=data or {},
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def to_event_model(doc: dict[str, Any]) -> RunEventModel:
        return RunEventModel(
            run_id=str(doc["run_id"]),
            seq=doc["seq"],
            type=RunEventType(doc["type"]),
            level=RunEventLevel(doc["level"]),
            message=doc["message"],
            data=doc.get("data") or {},
            created_at=doc["created_at"],
        )
