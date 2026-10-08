from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from .RunCommandDbModel import RunCommandCreateDbModel
from .RunCommandModel import RunCommandModel, RunCommandStatus, RunCommandType


class RunCommandMapper:
    @staticmethod
    def to_create_db_model(
        *, run_id: str, org_id: str, user_id: str, type: RunCommandType, text: str | None
    ) -> RunCommandCreateDbModel:
        return RunCommandCreateDbModel(
            run_id=ObjectId(run_id),
            org_id=ObjectId(org_id),
            created_by=ObjectId(user_id),
            type=RunCommandType(type).value,
            text=text,
            status=RunCommandStatus.PENDING.value,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> RunCommandModel:
        return RunCommandModel(
            id=str(doc["_id"]),
            run_id=str(doc["run_id"]),
            org_id=str(doc["org_id"]),
            created_by=str(doc["created_by"]),
            type=RunCommandType(doc["type"]),
            text=doc.get("text"),
            status=RunCommandStatus(doc["status"]),
            reason=doc.get("reason"),
            created_at=doc["created_at"],
            applied_at=doc.get("applied_at"),
        )
