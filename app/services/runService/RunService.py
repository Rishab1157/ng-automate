"""Runs of the master agent: their state, stage outputs and event timeline.

The run document is the only record of how far a run got. Every stage saves its output on the run, so a run
interrupted by a restart resumes at the first stage that has no output yet.
"""

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING

from app.core.exceptions import ErrorMessages, NotFoundError
from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.runModel import RunEventLevel, RunEventModel, RunEventType, RunMapper, RunModel, RunStage, RunStatus
from app.projections.runProjection import RUN_DETAIL_PROJECTION, RUN_EVENT_PROJECTION
from app.repositories.runEventRepository import RunEventRepository
from app.repositories.runRepository import RunRepository
from app.services.modelConnectionService import ModelConnectionService
from app.services.projectService import ProjectService

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_LIMIT = 200
MAX_EVENTS_LIMIT = 1000
MAX_EVENT_MESSAGE_CHARS = 1000
MAX_EVENT_DATA_STRING_CHARS = 4000
# Event data nested deeper than this is stored as text.
MAX_EVENT_DATA_DEPTH = 8
MAX_ERROR_MESSAGE_CHARS = 1000
TRUNCATED_MARK = "… [truncated]"
# Fullwidth dollar sign: replaces a leading "$" in event data keys.
EVENT_DATA_DOLLAR_ESCAPE = "＄"

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1

# Values the QXcel UI sends when no model connection is selected.
_NO_CONNECTION = {"", "undefined", "null", "none"}

_UNFINISHED_STATUSES = [RunStatus.QUEUED.value, RunStatus.RUNNING.value]


def _dump_profile_id(value: Any) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise ValueError("profile_id is not a valid id")
    return ObjectId(value)


# How each stage output is stored under `outputs.<key>`. Keys match RunOutputsModel.
_OUTPUT_DUMPERS: dict[str, Callable[[Any], Any]] = {
    "project_dir": str,
    "fact_sheet": lambda value: FactSheetModel.model_validate(value).model_dump(mode="json"),
    "findings": lambda value: AnalyzerFindingsModel.model_validate(value).model_dump(mode="json"),
    "llm_model": str,
    "profile_id": _dump_profile_id,
}


class RunService:
    def __init__(self) -> None:
        self.run_repo = RunRepository()
        self.run_event_repo = RunEventRepository()
        self.project_service = ProjectService()
        self.model_connection_service = ModelConnectionService()

    async def create_run(
        self, *, project_id: str, org_id: str, user_id: str, model_connection_id: str | None
    ) -> RunModel:
        """Insert a queued run. The caller starts it (the run executor)."""
        project = await self.project_service.get(project_id, org_id)
        connection_id = _connection_id_or_none(model_connection_id)
        if connection_id is not None:
            # Fail now (400/404) rather than minutes later inside the run. Only the id is stored: the key stays in QXcel.
            await self.model_connection_service.get_llm_config(connection_id, org_id)

        db_model = RunMapper.to_create_db_model(
            org_id=org_id, project_id=project.id, user_id=user_id, model_connection_id=connection_id
        )
        await self.run_repo.insert_one(db_model.model_dump(by_alias=True))
        logger.info("Run %s queued for project %s (org %s)", db_model.id, project.id, org_id)
        return RunMapper.to_model(db_model.model_dump(by_alias=True))

    async def get(self, run_id: str, org_id: str) -> RunModel:
        return RunMapper.to_model(await self._find_run(run_id, {"org_id": ObjectId(org_id)}, RUN_DETAIL_PROJECTION))

    async def get_by_id(self, run_id: str) -> RunModel:
        """Internal (run executor): no organization check."""
        return RunMapper.to_model(await self._find_run(run_id, {}, RUN_DETAIL_PROJECTION))

    async def get_events(
        self, run_id: str, org_id: str, after_seq: int = 0, limit: int = DEFAULT_EVENTS_LIMIT
    ) -> list[RunEventModel]:
        """Events after `after_seq`, oldest first: a client polls with the last seq it has."""
        await self._find_run(run_id, {"org_id": ObjectId(org_id)}, {"_id": 1})
        docs = await self.run_event_repo.find(
            {"run_id": ObjectId(run_id), "org_id": ObjectId(org_id), "seq": {"$gt": after_seq}},
            RUN_EVENT_PROJECTION,
            sort=[("seq", ASCENDING)],
            # A limit of 0 would mean "no limit" to MongoDB.
            limit=min(max(limit, 1), MAX_EVENTS_LIMIT),
        )
        return [RunMapper.to_event_model(doc) for doc in docs]

    async def find_unfinished(self) -> list[RunModel]:
        """Queued or running runs, oldest first: what a restarted server must resume."""
        docs = await self.run_repo.find(
            {"status": {"$in": _UNFINISHED_STATUSES}},
            RUN_DETAIL_PROJECTION,
            sort=[("created_at", ASCENDING), ("_id", ASCENDING)],
        )
        return [RunMapper.to_model(doc) for doc in docs]

    async def mark_running(self, run_id: str) -> None:
        now = datetime.now(UTC)
        # A resumed run keeps its first start time.
        await self.run_repo.update_one({"_id": ObjectId(run_id), "started_at": None}, {"$set": {"started_at": now}})
        await self._update_run(run_id, {"status": RunStatus.RUNNING.value}, now)

    async def set_stage(self, run_id: str, stage: RunStage) -> None:
        await self._update_run(run_id, {"stage": RunStage(stage).value})

    async def save_outputs(self, run_id: str, **outputs: Any) -> None:
        """Save stage outputs (keys of RunOutputsModel). None clears one, so its stage runs again on resume."""
        unknown = sorted(set(outputs) - _OUTPUT_DUMPERS.keys())
        if unknown:
            raise ValueError(f"Unknown run outputs: {', '.join(unknown)}")
        if not outputs:
            return
        fields = {
            f"outputs.{key}": None if value is None else _OUTPUT_DUMPERS[key](value) for key, value in outputs.items()
        }
        await self._update_run(run_id, fields)

    async def set_sandbox_container(self, run_id: str, container_id: str | None) -> None:
        await self._update_run(run_id, {"sandbox_container_id": container_id})

    async def mark_completed(self, run_id: str) -> None:
        now = datetime.now(UTC)
        await self._update_run(
            run_id, {"status": RunStatus.COMPLETED.value, "stage": RunStage.COMPLETED.value, "finished_at": now}, now
        )
        logger.info("Run %s completed", run_id)

    async def mark_failed(self, run_id: str, code: str, message: str) -> None:
        now = datetime.now(UTC)
        error = {"code": _enum_value(code), "message": _truncate(message, MAX_ERROR_MESSAGE_CHARS)}
        await self._update_run(
            run_id,
            {"status": RunStatus.FAILED.value, "stage": RunStage.FAILED.value, "error": error, "finished_at": now},
            now,
        )
        logger.warning("Run %s failed: %s", run_id, error["code"])

    async def append_event(
        self,
        run_id: str,
        org_id: str,
        type: RunEventType,
        message: str,
        level: RunEventLevel = RunEventLevel.INFO,
        data: dict[str, Any] | None = None,
    ) -> RunEventModel:
        """Add one event to the run's timeline. Safe to call concurrently: seq comes from an atomic counter."""
        counter = await self.run_repo.find_one_and_update(
            {"_id": ObjectId(run_id), "org_id": ObjectId(org_id)},
            {"$inc": {"event_seq": 1}, "$set": {"updated_at": datetime.now(UTC)}},
            projection={"event_seq": 1},
        )
        if counter is None:
            raise NotFoundError(ErrorMessages.RUN_NOT_FOUND)

        db_model = RunMapper.to_event_create_db_model(
            run_id=run_id,
            org_id=org_id,
            seq=counter["event_seq"],
            type=RunEventType(type),
            level=RunEventLevel(level),
            message=_truncate(message, MAX_EVENT_MESSAGE_CHARS),
            data=_clean_event_data(data or {}),
        )
        await self.run_event_repo.insert_one(db_model.model_dump(by_alias=True))
        return RunMapper.to_event_model(db_model.model_dump(by_alias=True))

    async def _find_run(self, run_id: str, scope: dict[str, Any], projection: dict[str, Any]) -> dict[str, Any]:
        if not ObjectId.is_valid(run_id):
            raise NotFoundError(ErrorMessages.RUN_NOT_FOUND)
        doc = await self.run_repo.find_one({"_id": ObjectId(run_id), **scope}, projection)
        if doc is None:
            raise NotFoundError(ErrorMessages.RUN_NOT_FOUND)
        return doc

    async def _update_run(self, run_id: str, fields: dict[str, Any], now: datetime | None = None) -> None:
        await self.run_repo.update_one(
            {"_id": ObjectId(run_id)}, {"$set": {**fields, "updated_at": now or datetime.now(UTC)}}
        )


def _connection_id_or_none(model_connection_id: str | None) -> str | None:
    if model_connection_id is None or model_connection_id.strip().lower() in _NO_CONNECTION:
        return None
    return model_connection_id.strip()


def _enum_value(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - len(TRUNCATED_MARK)] + TRUNCATED_MARK


def _clean_event_data(value: Any, depth: int = 0) -> Any:
    """A JSON- and BSON-safe copy with long strings cut. Event data comes from agents: saving it must never fail."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and _INT64_MIN <= value <= _INT64_MAX:
        return value
    # NaN and infinity are not valid JSON: the events endpoint could not return them.
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str):
        return _truncate(value, MAX_EVENT_DATA_STRING_CHARS)
    if depth < MAX_EVENT_DATA_DEPTH:
        if isinstance(value, dict):
            return {_clean_event_key(key): _clean_event_data(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_clean_event_data(item, depth + 1) for item in value]
    return _truncate(str(value), MAX_EVENT_DATA_STRING_CHARS)


def _clean_event_key(key: Any) -> str:
    """MongoDB rejects a NUL byte in a key. A key starting with "$" is escaped with a look-alike: PyMongo reads
    {"$ref", "$id"} back as a DBRef, which the events endpoint could never return as JSON."""
    text = str(key).replace("\x00", "")
    return EVENT_DATA_DOLLAR_ESCAPE + text[1:] if text.startswith("$") else text
