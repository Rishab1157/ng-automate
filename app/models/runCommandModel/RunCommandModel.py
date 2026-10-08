from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class RunCommandType(str, Enum):
    MESSAGE = "message"  # say something to the agent that is working (delivered at its next step)
    PAUSE = "pause"  # stop the working agent right away; it waits for resume or stop
    RESUME = "resume"
    STOP = "stop"  # end the whole run (status cancelled)


class RunCommandStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


class RunCommandModel(BaseModel):
    """A user's instruction to a running run, applied by the worker that executes it."""

    id: str
    run_id: str
    org_id: str
    created_by: str
    type: RunCommandType
    text: str | None = None
    status: RunCommandStatus
    reason: str | None = None  # why it was rejected
    created_at: datetime
    applied_at: datetime | None = None
