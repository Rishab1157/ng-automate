from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.healerModel import HealOutcomeModel
from app.models.testGeneratorModel import GenerationOutcomeModel
from app.models.testRunModel import FailureKind, TestCommandModel, TestRunResultModel


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"  # stopped by a user


class RunMode(str, Enum):
    """What a run does after the project profile is ready."""

    ANALYZE = "analyze"  # profile only
    TEST = "test"  # also run the project's tests, and let the healer fix build/dependency/environment failures
    GENERATE = "generate"  # write new tests from uploaded test data, then run and heal just those


class RunScope(str, Enum):
    """Generate mode: which tests run after the generator wrote its tests."""

    GENERATED = "generated"  # only the tests written from the test data (a framework may hold thousands of others)
    ALL = "all"  # the project's whole suite


class RunStage(str, Enum):
    """Where the master agent is. Saved after every step so an interrupted run resumes."""

    QUEUED = "queued"
    PREPARING_WORKSPACE = "preparing_workspace"
    COLLECTING_FACTS = "collecting_facts"
    ANALYZING = "analyzing"
    SAVING_PROFILE = "saving_profile"
    GENERATING_TESTS = "generating_tests"
    RUNNING_TESTS = "running_tests"
    HEALING = "healing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunEventType(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    AGENT_ACTION = "agent_action"
    AGENT_OBSERVATION = "agent_observation"
    AGENT_MESSAGE = "agent_message"
    AGENT_ERROR = "agent_error"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    USER_MESSAGE = "user_message"  # a message a user sent to the working agent
    COMMAND = "command"  # pause / resume / stop requests and what came of them


class RunEventLevel(str, Enum):
    """info: the short timeline a user reads. detail: raw tool calls and output, shown on expand."""

    INFO = "info"
    DETAIL = "detail"


class RunErrorModel(BaseModel):
    code: str
    message: str


class TestStopReason(str, Enum):
    """Why the test phase stopped."""

    __test__ = False  # not a pytest test class

    PASSED = "passed"
    NOT_HEALABLE = "not_healable"  # failing assertions, locators, no tests: for the user, not the healer
    MAX_ATTEMPTS = "max_attempts"  # older runs only: there is no fix limit any more
    # The same failure keeps coming back, or the healer's edits brought back files that already failed this way.
    NO_PROGRESS = "no_progress"
    BLOCKED = "blocked"  # the healer proved that the environment stops it (no network, no permission, ...)
    TIMED_OUT = "timed_out"  # the run's healing time budget is used up
    HEALER_FAILED = "healer_failed"  # the healer itself stopped (model unreachable, ...)


class TestAttemptModel(BaseModel):
    """One run of the tests, and the healer's fix after it (if it ran)."""

    __test__ = False

    number: int  # 1-based
    result: TestRunResultModel
    heal: HealOutcomeModel | None = None
    # Fingerprint of the project's files when the tests ran: tells when the healer brings back files already tested.
    project_state: str | None = None


class TestReportModel(BaseModel):
    """The outcome of the test phase."""

    __test__ = False

    outcome: FailureKind  # classification of the last run
    stop_reason: TestStopReason
    detail: str | None = None  # e.g. why the healer stopped
    runs: int
    heals: int
    # Counts of the last run.
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    # Every file the healer changed and kept, over all its attempts.
    changed_files: list[str] = Field(default_factory=list)


class RunOutputsModel(BaseModel):
    """What each finished stage produced. A stage whose output is set is skipped on resume."""

    # Host folder the project was extracted to for this run. Internal: never returned by the API.
    project_dir: str | None = None
    fact_sheet: FactSheetModel | None = None
    findings: AnalyzerFindingsModel | None = None
    # LiteLLM name of the model that produced `findings`, recorded on the profile.
    llm_model: str | None = None
    profile_id: str | None = None
    # Generate mode only: the tests the generator wrote.
    generation: GenerationOutcomeModel | None = None
    # Test and generate modes.
    test_command: TestCommandModel | None = None
    test_attempts: list[TestAttemptModel] | None = None
    test_report: TestReportModel | None = None


class RunModel(BaseModel):
    id: str
    org_id: str
    project_id: str
    created_by: str
    model_connection_id: str | None = None
    mode: RunMode = RunMode.ANALYZE
    # Test mode: which tests to run (a test class/method, or tags for Cucumber/pytest/Playwright).
    test_selector: str | None = None
    # Generate mode: the uploaded test data to write tests from, and which tests run afterwards.
    test_data_id: str | None = None
    run_scope: RunScope = RunScope.GENERATED
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
