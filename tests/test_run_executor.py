import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from langgraph.graph.state import CompiledStateGraph
from pymongo import MongoClient

import app.main as main_module
from app.agents.MasterAgent.MasterGraph import NODE_NAMES, build_master_graph
from app.agents.MasterAgent.MasterNodes import MasterNodes
from app.agents.MasterAgent.RunExecutor import RUN_COMPLETED_MESSAGE, RUN_TIMEOUT_ERROR_CODE, RunExecutor
from app.config import settings
from app.core.exceptions import AnalysisError, ErrorMessages, NotFoundError, SandboxError
from app.models.analyzerModel import FactSheetModel
from app.models.runModel import (
    RunErrorModel,
    RunEventLevel,
    RunEventType,
    RunMapper,
    RunModel,
    RunOutputsModel,
    RunStage,
    RunStatus,
)
from app.services.projectProfileService import ProjectProfileService
from app.services.runService import RunService
from tests.test_master_graph import (
    _write_cited_files,
    CONTAINER_ID,
    FACT_SHEET,
    FINDINGS,
    LLM_MODEL,
    FakeAnalyzer,
    FakeModelConnections,
    FakeSandbox,
)

GraphBehavior = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def make_run(
    status: RunStatus = RunStatus.QUEUED,
    outputs: RunOutputsModel | None = None,
    container_id: str | None = None,
) -> RunModel:
    now = datetime.now(UTC)
    return RunModel(
        id=str(ObjectId()),
        org_id=str(ObjectId()),
        project_id=str(ObjectId()),
        created_by=str(ObjectId()),
        status=status,
        stage=RunStage.QUEUED if status == RunStatus.QUEUED else RunStage.ANALYZING,
        outputs=outputs or RunOutputsModel(),
        sandbox_container_id=container_id,
        created_at=now,
        updated_at=now,
    )


class FakeRunService:
    """In-memory runs with the RunService methods the executor uses."""

    def __init__(self, *runs: RunModel) -> None:
        self.runs = {run.id: run for run in runs}
        self.events: list[tuple[str, RunEventType, str, dict[str, Any]]] = []

    def _update(self, run_id: str, **changes: Any) -> None:
        self.runs[run_id] = self.runs[run_id].model_copy(update=changes)

    async def get_by_id(self, run_id: str) -> RunModel:
        if run_id not in self.runs:
            raise NotFoundError(ErrorMessages.RUN_NOT_FOUND)
        return self.runs[run_id]

    async def find_unfinished(self) -> list[RunModel]:
        return [run for run in self.runs.values() if run.status in (RunStatus.QUEUED, RunStatus.RUNNING)]

    async def mark_running(self, run_id: str) -> None:
        self._update(run_id, status=RunStatus.RUNNING)

    async def mark_completed(self, run_id: str) -> None:
        self._update(run_id, status=RunStatus.COMPLETED, stage=RunStage.COMPLETED)

    async def mark_cancelled(self, run_id: str, message: str) -> None:
        self._update(
            run_id, status=RunStatus.CANCELLED, stage=RunStage.CANCELLED,
            error=RunErrorModel(code="RUN_CANCELLED", message=message),
        )

    async def mark_failed(self, run_id: str, code: str, message: str) -> None:
        self._update(run_id, status=RunStatus.FAILED, stage=RunStage.FAILED, error=RunErrorModel(code=code, message=message))

    async def set_sandbox_container(self, run_id: str, container_id: str | None) -> None:
        self._update(run_id, sandbox_container_id=container_id)

    async def append_event(
        self,
        run_id: str,
        org_id: str,
        type: RunEventType,
        message: str,
        level: RunEventLevel = RunEventLevel.INFO,
        data: dict[str, Any] | None = None,
    ) -> None:
        assert org_id == self.runs[run_id].org_id
        self.events.append((run_id, type, message, data or {}))


class FakeSandboxService:
    def __init__(self, stop_error: Exception | None = None) -> None:
        self.stop_error = stop_error
        self.stopped: list[str] = []
        self.removed: list[str] = []

    def stop_container(self, container_id: str) -> None:
        self.stopped.append(container_id)
        if self.stop_error is not None:
            raise self.stop_error

    def remove_run_dir(self, run_id: str) -> None:
        self.removed.append(run_id)


class FakeGraph:
    """Stands in for the compiled master graph; tracks how many runs execute at once."""

    def __init__(self, behavior: GraphBehavior | None = None) -> None:
        self.behavior = behavior
        self.states: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.states.append(dict(state))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.behavior is not None:
                return await self.behavior(state)
            await asyncio.sleep(0.01)
            return {**state, "profile_id": "profile-1"}
        finally:
            self.active -= 1


class FakeCommands:
    """No user commands unless a test queues some; records what was marked."""

    def __init__(self) -> None:
        self.pending: list[Any] = []
        self.marked: list[tuple[Any, Any, str | None]] = []

    async def get_pending(self, run_id: str) -> list[Any]:
        pending, self.pending = self.pending, []
        return pending

    async def mark(self, command: Any, status: Any, message: str | None = None) -> None:
        self.marked.append((command, status, message))


def make_executor(
    runs: FakeRunService, graph: FakeGraph, sandbox: FakeSandboxService | None = None, **options: Any
) -> tuple[RunExecutor, FakeSandboxService]:
    sandbox = sandbox or FakeSandboxService()
    options.setdefault("command_service", FakeCommands())
    executor = RunExecutor(
        run_service=runs,  # type: ignore[arg-type]
        sandbox_service=sandbox,  # type: ignore[arg-type]
        graph_factory=lambda: graph,  # type: ignore[arg-type,return-value]
        **options,
    )
    return executor, sandbox


async def settle(executor: RunExecutor) -> None:
    """Wait until every run task of the executor has finished (cleanup included)."""
    async with asyncio.timeout(10):
        while executor._tasks:
            await asyncio.gather(*executor._tasks.values(), return_exceptions=True)
            await asyncio.sleep(0)


async def wait_until(condition: Callable[[], bool], timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- outcomes


@pytest.mark.anyio
async def test_successful_run_is_completed_and_cleaned_up() -> None:
    run = make_run()
    runs = FakeRunService(run)
    graph = FakeGraph()
    executor, sandbox = make_executor(runs, graph)

    await executor.submit(run.id)
    await settle(executor)

    done = runs.runs[run.id]
    assert done.status == RunStatus.COMPLETED
    assert done.error is None
    assert graph.states == [
        {"run_id": run.id, "org_id": run.org_id, "project_id": run.project_id, "model_connection_id": None,
         "mode": "analyze", "test_selector": None, "test_data_id": None,
         "run_scope": "generated"}
    ]
    assert runs.events == [(run.id, RunEventType.RUN_COMPLETED, RUN_COMPLETED_MESSAGE, {"profile_id": "profile-1"})]
    assert sandbox.stopped == []
    assert sandbox.removed == [run.id]


@pytest.mark.anyio
async def test_test_run_reports_the_test_outcome_when_it_completes() -> None:
    run = make_run()
    runs = FakeRunService(run)
    report = {"outcome": "passed", "stop_reason": "passed", "detail": None, "runs": 2, "heals": 1, "total": 3,
              "passed": 3, "failed": 0, "errors": 0, "skipped": 0, "changed_files": ["pom.xml"]}

    async def tests_passed(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "profile_id": "profile-1", "test_report": report}

    executor, _ = make_executor(runs, FakeGraph(tests_passed))

    await executor.submit(run.id)
    await settle(executor)

    [(_, event_type, message, data)] = runs.events
    assert event_type == RunEventType.RUN_COMPLETED
    assert message == "Run completed: All 3 tests passed after 1 fix by the healer"
    assert data == {"profile_id": "profile-1", "test_report": report}


@pytest.mark.anyio
async def test_resumed_run_starts_from_its_saved_outputs() -> None:
    outputs = RunOutputsModel(project_dir="/data/runs/x/project", fact_sheet=FACT_SHEET)
    run = make_run(status=RunStatus.RUNNING, outputs=outputs)
    runs = FakeRunService(run)
    graph = FakeGraph()
    executor, _ = make_executor(runs, graph)

    await executor.submit(run.id)
    await settle(executor)

    state = graph.states[0]
    assert state["project_dir"] == "/data/runs/x/project"
    assert state["fact_sheet"] == FACT_SHEET.model_dump(mode="json")
    assert "findings" not in state and "profile_id" not in state
    assert runs.runs[run.id].status == RunStatus.COMPLETED


@pytest.mark.anyio
async def test_known_error_fails_the_run_with_its_code_and_message() -> None:
    message = ErrorMessages.SANDBOX_FAILED.format(reason="Docker is not running")

    async def sandbox_fails(state: dict[str, Any]) -> dict[str, Any]:
        raise SandboxError(message)

    run = make_run()
    runs = FakeRunService(run)
    executor, sandbox = make_executor(runs, FakeGraph(sandbox_fails))

    await executor.submit(run.id)
    await settle(executor)

    failed = runs.runs[run.id]
    assert failed.status == RunStatus.FAILED
    assert failed.stage == RunStage.FAILED
    assert failed.error == RunErrorModel(code="SANDBOX_FAILED", message=message)
    assert runs.events == [(run.id, RunEventType.RUN_FAILED, message, {"code": "SANDBOX_FAILED"})]
    assert sandbox.removed == [run.id]


@pytest.mark.anyio
async def test_unexpected_error_is_an_internal_error_and_its_text_stays_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def crash(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("cannot connect to mongodb://admin:hunter2@db")

    run = make_run()
    runs = FakeRunService(run)
    executor, sandbox = make_executor(runs, FakeGraph(crash))

    await executor.submit(run.id)
    await settle(executor)

    failed = runs.runs[run.id]
    assert failed.error == RunErrorModel(code="INTERNAL_SERVER_ERROR", message=ErrorMessages.INTERNAL_ERROR)
    assert runs.events == [
        (run.id, RunEventType.RUN_FAILED, ErrorMessages.INTERNAL_ERROR, {"code": "INTERNAL_SERVER_ERROR"})
    ]
    assert "hunter2" not in repr(failed) + repr(runs.events)
    logged = [record for record in caplog.records if "unexpected error" in record.getMessage()]
    assert logged and logged[0].exc_info is not None
    assert sandbox.removed == [run.id]


@pytest.mark.anyio
async def test_slow_run_times_out() -> None:
    async def slow(state: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(30)
        return state

    run = make_run()
    runs = FakeRunService(run)
    executor, sandbox = make_executor(runs, FakeGraph(slow), timeout_seconds=0.1)

    await executor.submit(run.id)
    await settle(executor)

    failed = runs.runs[run.id]
    assert failed.status == RunStatus.FAILED
    assert failed.error == RunErrorModel(
        code=RUN_TIMEOUT_ERROR_CODE, message="The run took longer than 0.1 seconds and was stopped"
    )
    assert runs.events[-1][1] == RunEventType.RUN_FAILED
    assert sandbox.removed == [run.id]


@pytest.mark.anyio
async def test_timeout_error_from_inside_the_run_is_not_reported_as_a_run_timeout() -> None:
    async def socket_timeout(state: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("read timed out")

    run = make_run()
    runs = FakeRunService(run)
    executor, _ = make_executor(runs, FakeGraph(socket_timeout))

    await executor.submit(run.id)
    await settle(executor)

    assert runs.runs[run.id].error == RunErrorModel(code="INTERNAL_SERVER_ERROR", message=ErrorMessages.INTERNAL_ERROR)


@pytest.mark.anyio
async def test_container_still_recorded_after_a_failure_is_stopped_and_cleared() -> None:
    run = make_run()
    runs = FakeRunService(run)

    async def fails_in_the_sandbox(state: dict[str, Any]) -> dict[str, Any]:
        await runs.set_sandbox_container(state["run_id"], "c-1")
        raise AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason="no JSON block"))

    executor, sandbox = make_executor(runs, FakeGraph(fails_in_the_sandbox))

    await executor.submit(run.id)
    await settle(executor)

    assert sandbox.stopped == ["c-1"]
    assert runs.runs[run.id].sandbox_container_id is None
    assert runs.runs[run.id].error is not None and runs.runs[run.id].error.code == "ANALYSIS_FAILED"


@pytest.mark.anyio
async def test_timeout_stops_the_container_of_the_running_analysis() -> None:
    run = make_run()
    runs = FakeRunService(run)

    async def stuck_analysis(state: dict[str, Any]) -> dict[str, Any]:
        await runs.set_sandbox_container(state["run_id"], "c-1")
        await asyncio.sleep(30)
        return state

    executor, sandbox = make_executor(runs, FakeGraph(stuck_analysis), timeout_seconds=0.1)

    await executor.submit(run.id)
    await settle(executor)

    assert sandbox.stopped == ["c-1"]
    assert runs.runs[run.id].sandbox_container_id is None
    assert runs.runs[run.id].error is not None and runs.runs[run.id].error.code == RUN_TIMEOUT_ERROR_CODE


@pytest.mark.anyio
@pytest.mark.parametrize("status", [RunStatus.COMPLETED, RunStatus.FAILED])
async def test_finished_run_is_not_run_again(status: RunStatus) -> None:
    run = make_run(status=status)
    runs = FakeRunService(run)
    graph = FakeGraph()
    executor, sandbox = make_executor(runs, graph)

    await executor.submit(run.id)
    await settle(executor)

    assert graph.states == []
    assert runs.events == []
    assert runs.runs[run.id] == run
    assert sandbox.removed == []


@pytest.mark.anyio
async def test_unknown_run_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    graph = FakeGraph()
    executor, _ = make_executor(FakeRunService(), graph)

    await executor.submit(str(ObjectId()))
    await settle(executor)

    assert graph.states == []
    assert "no longer exists" in caplog.text


# --------------------------------------------------------------------------- scheduling


@pytest.mark.anyio
async def test_duplicate_submit_is_ignored() -> None:
    release = asyncio.Event()

    async def wait_for_release(state: dict[str, Any]) -> dict[str, Any]:
        await release.wait()
        return state

    run = make_run()
    runs = FakeRunService(run)
    graph = FakeGraph(wait_for_release)
    executor, _ = make_executor(runs, graph)

    await executor.submit(run.id)
    await executor.submit(run.id)
    await wait_until(lambda: len(graph.states) == 1)
    await executor.submit(run.id)
    release.set()
    await settle(executor)

    assert len(graph.states) == 1
    assert runs.runs[run.id].status == RunStatus.COMPLETED


@pytest.mark.anyio
async def test_only_max_concurrent_runs_execute_at_once() -> None:
    async def busy(state: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return state

    runs = FakeRunService(*(make_run() for _ in range(5)))
    graph = FakeGraph(busy)
    executor, _ = make_executor(runs, graph, max_concurrent=2)

    for run_id in list(runs.runs):
        await executor.submit(run_id)
    await settle(executor)

    assert graph.max_active == 2
    assert len(graph.states) == 5
    assert {run.status for run in runs.runs.values()} == {RunStatus.COMPLETED}


@pytest.mark.anyio
async def test_graph_is_built_once_on_first_use() -> None:
    runs = FakeRunService(make_run(), make_run())
    builds: list[FakeGraph] = []

    def build() -> FakeGraph:
        builds.append(FakeGraph())
        return builds[-1]

    executor = RunExecutor(
        run_service=runs,  # type: ignore[arg-type]
        sandbox_service=FakeSandboxService(),  # type: ignore[arg-type]
        graph_factory=build,  # type: ignore[arg-type]
        command_service=FakeCommands(),  # type: ignore[arg-type]
    )
    assert builds == []

    for run_id in list(runs.runs):
        await executor.submit(run_id)
    await settle(executor)

    assert len(builds) == 1 and len(builds[0].states) == 2


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        RunExecutor(max_concurrent=0)
    with pytest.raises(ValueError):
        RunExecutor(timeout_seconds=0)


def test_production_dependencies_are_created_on_first_use() -> None:
    executor = RunExecutor()
    assert executor._run_service is None and executor._sandbox_service is None and executor._graph is None

    graph = executor._get_graph()

    assert isinstance(graph, CompiledStateGraph)
    assert set(NODE_NAMES) <= set(graph.nodes)
    assert isinstance(executor.run_service, RunService)
    assert executor._sandbox_service is not None


# --------------------------------------------------------------------------- resume and shutdown


@pytest.mark.anyio
async def test_resume_interrupted_stops_left_containers_and_resubmits_unfinished_runs() -> None:
    crashed = make_run(status=RunStatus.RUNNING, container_id="c-old")
    queued = make_run()
    finished = make_run(status=RunStatus.COMPLETED)
    runs = FakeRunService(crashed, queued, finished)
    container_at_start: dict[str, str | None] = {}

    async def record(state: dict[str, Any]) -> dict[str, Any]:
        container_at_start[state["run_id"]] = runs.runs[state["run_id"]].sandbox_container_id
        return {**state, "profile_id": "profile-1"}

    executor, sandbox = make_executor(runs, FakeGraph(record))

    assert await executor.resume_interrupted() == 2
    await settle(executor)

    assert sandbox.stopped == ["c-old"]
    assert container_at_start == {crashed.id: None, queued.id: None}
    assert runs.runs[crashed.id].status == RunStatus.COMPLETED
    assert runs.runs[queued.id].status == RunStatus.COMPLETED
    assert runs.runs[finished.id] == finished


@pytest.mark.anyio
async def test_resume_leaves_runs_already_executing_here_alone() -> None:
    release = asyncio.Event()
    run = make_run()
    runs = FakeRunService(run)

    async def busy(state: dict[str, Any]) -> dict[str, Any]:
        await runs.set_sandbox_container(state["run_id"], "c-live")
        await release.wait()
        await runs.set_sandbox_container(state["run_id"], None)
        return state

    executor, sandbox = make_executor(runs, FakeGraph(busy))
    await executor.submit(run.id)
    await wait_until(lambda: runs.runs[run.id].sandbox_container_id == "c-live")

    assert await executor.resume_interrupted() == 0
    assert sandbox.stopped == []

    release.set()
    await settle(executor)
    assert runs.runs[run.id].status == RunStatus.COMPLETED


@pytest.mark.anyio
async def test_container_that_cannot_be_stopped_does_not_block_the_run(caplog: pytest.LogCaptureFixture) -> None:
    crashed = make_run(status=RunStatus.RUNNING, container_id="c-old")
    runs = FakeRunService(crashed)
    sandbox = FakeSandboxService(stop_error=SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="Docker is down")))
    executor, _ = make_executor(runs, FakeGraph(), sandbox)

    assert await executor.resume_interrupted() == 1
    await settle(executor)

    assert sandbox.stopped == ["c-old"]
    assert runs.runs[crashed.id].sandbox_container_id is None
    assert runs.runs[crashed.id].status == RunStatus.COMPLETED
    assert "Could not stop container c-old" in caplog.text


@pytest.mark.anyio
async def test_shutdown_leaves_runs_unfinished_so_they_resume_on_the_next_start() -> None:
    started = asyncio.Event()
    cancelled: list[str] = []
    running, waiting = make_run(), make_run()
    runs = FakeRunService(running, waiting)

    async def long_analysis(state: dict[str, Any]) -> dict[str, Any]:
        await runs.set_sandbox_container(state["run_id"], "c-1")
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(state["run_id"])
            raise
        return state

    graph = FakeGraph(long_analysis)
    executor, sandbox = make_executor(runs, graph, max_concurrent=1)
    await executor.submit(running.id)
    await executor.submit(waiting.id)
    async with asyncio.timeout(5):
        await started.wait()

    await executor.shutdown()

    interrupted = runs.runs[running.id]
    assert cancelled == [running.id]
    assert interrupted.status == RunStatus.RUNNING
    assert interrupted.error is None
    assert interrupted.sandbox_container_id is None
    assert sandbox.stopped == ["c-1"]
    assert sandbox.removed == []
    assert runs.events == []
    # The run waiting for a free slot never started and is still queued.
    assert runs.runs[waiting.id].status == RunStatus.QUEUED
    assert len(graph.states) == 1
    assert executor._tasks == {}


# --------------------------------------------------------------------------- real database (RunService, profiles)


def _insert_run(mongo: MongoClient, **fields: Any) -> dict[str, Any]:
    doc = RunMapper.to_create_db_model(
        org_id=str(ObjectId()), project_id=str(ObjectId()), user_id=str(ObjectId()), model_connection_id=None
    ).model_dump(by_alias=True)
    doc.update(fields)
    mongo[settings.MONGO_DATABASE].runs.insert_one(doc)
    return doc


def _real_executor(sandbox: FakeSandbox, analyzer: FakeAnalyzer, fact_collector: Callable[[Path], FactSheetModel]) -> RunExecutor:
    run_service = RunService()

    def graph_factory() -> CompiledStateGraph:
        nodes = MasterNodes(
            run_service=run_service,
            sandbox_service=sandbox,  # type: ignore[arg-type]
            model_connection_service=FakeModelConnections(),  # type: ignore[arg-type]
            profile_service=ProjectProfileService(),
            fact_collector=fact_collector,
            analyzer_factory=analyzer.factory,  # type: ignore[arg-type]
        )
        return build_master_graph(nodes)

    return RunExecutor(run_service=run_service, sandbox_service=sandbox, graph_factory=graph_factory)  # type: ignore[arg-type]


def _event_types(mongo: MongoClient, run_id: ObjectId) -> list[str]:
    events = list(mongo[settings.MONGO_DATABASE].run_events.find({"run_id": run_id}).sort("seq", 1))
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    return [event["type"] for event in events]


@pytest.mark.anyio
async def test_run_end_to_end_with_the_real_run_and_profile_services(app_db: MongoClient, tmp_path: Path) -> None:
    db = app_db[settings.MONGO_DATABASE]
    run_doc = _insert_run(app_db)
    container_during_analysis: list[str | None] = []

    class CheckingAnalyzer(FakeAnalyzer):
        def analyze(self, workspace: Any, fact_sheet: FactSheetModel, on_event: Any = None, control: Any = None) -> Any:
            container_during_analysis.append(db.runs.find_one({"_id": run_doc["_id"]})["sandbox_container_id"])
            return super().analyze(workspace, fact_sheet, on_event)

    sandbox = FakeSandbox(tmp_path)
    executor = _real_executor(sandbox, CheckingAnalyzer(events=3), lambda project_dir: FACT_SHEET)

    await executor.submit(str(run_doc["_id"]))
    await settle(executor)

    doc = db.runs.find_one({"_id": run_doc["_id"]})
    profile = db.project_profiles.find_one({"run_id": run_doc["_id"]})
    assert doc["status"] == "completed" and doc["stage"] == "completed" and doc["error"] is None
    assert doc["started_at"] is not None and doc["finished_at"] is not None
    assert doc["sandbox_container_id"] is None
    assert container_during_analysis == [CONTAINER_ID]
    assert profile is not None and profile["llm_model"] == LLM_MODEL
    assert doc["outputs"]["profile_id"] == profile["_id"]
    assert doc["outputs"]["fact_sheet"] == FACT_SHEET.model_dump(mode="json")
    assert doc["outputs"]["findings"] == FINDINGS.model_dump(mode="json")
    assert doc["outputs"]["llm_model"] == LLM_MODEL
    assert _event_types(app_db, run_doc["_id"]) == [
        "stage_started", "stage_completed",
        "stage_started", "stage_completed",
        "stage_started", "agent_action", "agent_action", "agent_action", "stage_completed",
        "stage_started", "stage_completed",
        "run_completed",
    ]
    assert sandbox.removed == [str(run_doc["_id"])]


@pytest.mark.anyio
async def test_interrupted_run_resumes_after_its_last_saved_stage(app_db: MongoClient, tmp_path: Path) -> None:
    project_dir = tmp_path / "runs" / "previous" / "project"
    _write_cited_files(project_dir)
    run_doc = _insert_run(
        app_db,
        status="running",
        stage="analyzing",
        outputs={"project_dir": str(project_dir), "fact_sheet": FACT_SHEET.model_dump(mode="json")},
        sandbox_container_id="c-crashed",
    )

    def facts_are_already_known(project_dir: Path) -> FactSheetModel:
        raise AssertionError("a finished stage ran again")

    sandbox = FakeSandbox(tmp_path)
    executor = _real_executor(sandbox, FakeAnalyzer(events=1), facts_are_already_known)

    assert await executor.resume_interrupted() == 1
    await settle(executor)

    doc = app_db[settings.MONGO_DATABASE].runs.find_one({"_id": run_doc["_id"]})
    assert sandbox.stopped == ["c-crashed"]
    assert sandbox.opened_with == project_dir
    assert doc["status"] == "completed"
    assert doc["outputs"]["project_dir"] == str(project_dir)
    assert _event_types(app_db, run_doc["_id"]) == [
        "stage_started", "agent_action", "stage_completed",
        "stage_started", "stage_completed",
        "run_completed",
    ]


# --------------------------------------------------------------------------- app lifespan


class RecordingExecutor:
    def __init__(self, resume_error: Exception | None = None) -> None:
        self.resume_error = resume_error
        self.calls: list[str] = []

    async def resume_interrupted(self) -> int:
        self.calls.append("resume_interrupted")
        if self.resume_error is not None:
            raise self.resume_error
        return 2

    async def shutdown(self) -> None:
        self.calls.append("shutdown")


def test_app_resumes_runs_on_startup_and_stops_them_on_shutdown(
    mongo: MongoClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = RecordingExecutor()
    monkeypatch.setattr(main_module, "run_executor", executor)

    with TestClient(main_module.app):
        assert executor.calls == ["resume_interrupted"]

    assert executor.calls == ["resume_interrupted", "shutdown"]


def test_app_starts_even_when_resuming_fails(
    mongo: MongoClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    executor = RecordingExecutor(resume_error=RuntimeError("database hiccup"))
    monkeypatch.setattr(main_module, "run_executor", executor)

    with TestClient(main_module.app) as client:
        assert client.get("/health").status_code == 200

    assert executor.calls == ["resume_interrupted", "shutdown"]
    assert "Could not resume interrupted runs" in caplog.text
