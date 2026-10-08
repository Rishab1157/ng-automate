"""Commands users send to a running run: a message for the working agent, pause, resume, stop.

The API stores them; the worker executing the run polls its pending commands and applies them (RunControl), so the
API and the worker can be different processes. Every command shows in the run's timeline, and so does its outcome.
"""

import logging
from datetime import UTC, datetime

from bson import ObjectId
from pymongo import ASCENDING

from app.core.exceptions import ErrorMessages, RunNotActiveError, ValidationError
from app.models.runCommandModel import RunCommandMapper, RunCommandModel, RunCommandStatus, RunCommandType
from app.models.runModel import RunEventType, RunStatus
from app.projections.runCommandProjection import RUN_COMMAND_PROJECTION
from app.repositories.runCommandRepository import RunCommandRepository
from app.services.runService import RunService

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
MAX_PENDING_READ = 50
_ACTIVE = (RunStatus.QUEUED, RunStatus.RUNNING)
_REQUESTED = {
    RunCommandType.PAUSE: "You asked to pause the agent",
    RunCommandType.RESUME: "You asked to resume the agent",
    RunCommandType.STOP: "You asked to stop the run",
}


class RunCommandService:
    def __init__(self, run_service: RunService | None = None) -> None:
        self.command_repo = RunCommandRepository()
        self.run_service = run_service or RunService()

    async def create(
        self, *, run_id: str, org_id: str, user_id: str, type: RunCommandType, text: str | None = None
    ) -> RunCommandModel:
        run = await self.run_service.get(run_id, org_id)
        if run.status not in _ACTIVE:
            raise RunNotActiveError(ErrorMessages.RUN_NOT_ACTIVE.format(status=run.status.value))
        text = (text or "").strip() or None
        if type == RunCommandType.MESSAGE:
            if text is None:
                raise ValidationError(ErrorMessages.COMMAND_TEXT_REQUIRED)
            if len(text) > MAX_MESSAGE_CHARS:
                raise ValidationError(ErrorMessages.COMMAND_TEXT_TOO_LONG.format(limit=MAX_MESSAGE_CHARS))
        else:
            text = None
        db_model = RunCommandMapper.to_create_db_model(run_id=run_id, org_id=org_id, user_id=user_id, type=type, text=text)
        await self.command_repo.insert_one(db_model.model_dump(by_alias=True))
        command = RunCommandMapper.to_model(db_model.model_dump(by_alias=True))
        if type == RunCommandType.MESSAGE:
            await self.run_service.append_event(
                run_id, org_id, RunEventType.USER_MESSAGE, f"You: {text}", data={"command_id": command.id}
            )
        else:
            await self.run_service.append_event(
                run_id, org_id, RunEventType.COMMAND, _REQUESTED[type], data={"command_id": command.id, "type": type.value}
            )
        logger.info("Command %s (%s) for run %s", command.id, type.value, run_id)
        return command

    async def get_pending(self, run_id: str) -> list[RunCommandModel]:
        """Internal (worker): the run's pending commands, oldest first."""
        docs = await self.command_repo.find(
            {"run_id": ObjectId(run_id), "status": RunCommandStatus.PENDING.value},
            RUN_COMMAND_PROJECTION,
            sort=[("_id", ASCENDING)],
            limit=MAX_PENDING_READ,
        )
        return [RunCommandMapper.to_model(doc) for doc in docs]

    async def mark(self, command: RunCommandModel, status: RunCommandStatus, message: str | None = None) -> None:
        """Record the outcome; `message` (what happened) goes to the timeline, and to `reason` when rejected."""
        await self.command_repo.update_one(
            {"_id": ObjectId(command.id), "status": RunCommandStatus.PENDING.value},
            {"$set": {
                "status": status.value,
                "reason": message if status == RunCommandStatus.REJECTED else None,
                "applied_at": datetime.now(UTC),
            }},
        )
        if message:
            await self.run_service.append_event(
                command.run_id, command.org_id, RunEventType.COMMAND, message,
                data={"command_id": command.id, "type": command.type.value, "status": status.value},
            )
