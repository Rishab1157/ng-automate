import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId

from .LiveViewDbModel import LiveViewTicketCreateDbModel
from .LiveViewModel import LiveViewMode, LiveViewTicketModel


def ticket_hash(ticket: str) -> str:
    return hashlib.sha256(ticket.encode("utf-8")).hexdigest()


class LiveViewMapper:
    @staticmethod
    def to_create_db_model(
        *,
        ticket: str,
        run_id: str,
        org_id: str,
        user_id: str,
        container_id: str,
        host_port: int,
        vnc_token: str,
        mode: LiveViewMode,
        lifetime_seconds: int,
    ) -> LiveViewTicketCreateDbModel:
        now = datetime.now(UTC)
        return LiveViewTicketCreateDbModel(
            _id=ticket_hash(ticket),
            run_id=ObjectId(run_id),
            org_id=ObjectId(org_id),
            created_by=ObjectId(user_id),
            container_id=container_id,
            host_port=host_port,
            vnc_token=vnc_token,
            mode=LiveViewMode(mode).value,
            created_at=now,
            expires_at=now + timedelta(seconds=lifetime_seconds),
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> LiveViewTicketModel:
        expires_at = doc["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)  # MongoDB returns naive UTC datetimes
        return LiveViewTicketModel(
            run_id=str(doc["run_id"]),
            org_id=str(doc["org_id"]),
            container_id=doc["container_id"],
            host_port=doc["host_port"],
            vnc_token=doc["vnc_token"],
            mode=LiveViewMode(doc["mode"]),
            expires_at=expires_at,
        )
