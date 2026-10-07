from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStage(str, Enum):
    """Where the master agent is. Saved after every step so an interrupted run resumes."""

    QUEUED = "queued"
    PREPARING_WORKSPACE = "preparing_workspace"
    COLLECTING_FACTS = "collecting_facts"
    ANALYZING = "analyzing"
    SAVING_PROFILE = "saving_profile"
    COMPLETED = "completed"
    FAILED = "failed"


class RunEventType(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    AGENT_ACTION = "agent_action"
    AGENT_OBSERVATION = "agent_observation"
    AGENT_MESSAGE = "agent_message"
    AGENT_ERROR = "agent_error"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class RunEventLevel(str, Enum):
    """info: the short timeline a user reads. detail: raw tool calls and output, shown on expand."""

    INFO = "info"
    DETAIL = "detail"


class RunErrorModel(BaseModel):
    code: str
    message: str


class RunOutputsModel(BaseModel):
    """What each finished stage produced. A stage whose output is set is skipped on resume."""

    # Host folder the project was extracted to for this run. Internal: never returned by the API.
    project_dir: str | None = None
    fact_sheet: FactSheetModel | None = None
    findings: AnalyzerFindingsModel | None = None
    # LiteLLM name of the model that produced `findings`, recorded on the profile.
    llm_model: str | None = None
    profile_id: str | None = None


class RunModel(BaseModel):
    id: str
    org_id: str
    project_id: str
    created_by: str
    model_connection_id: str | None = None
    status: RunStatus
    stage: RunStage
    outputs: RunOutputsModel = Field(default_factory=RunOutputsModel)
    # Container of the stage running right now, so a crashed server can stop it on restart.
    sandbox_container_id: str | None = None
    error: RunErrorModel | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunEventModel(BaseModel):
    run_id: str
    seq: int
    type: RunEventType
    level: RunEventLevel
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
