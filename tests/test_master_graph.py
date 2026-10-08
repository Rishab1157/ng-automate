import asyncio
import random
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from bson import ObjectId
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from pydantic import SecretStr

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.MasterAgent.MasterGraph import (
    ANALYZE,
    COLLECT_FACTS,
    GENERATE_TESTS,
    MAX_GRAPH_STEPS,
    PREPARE_WORKSPACE,
    SAVE_PROFILE,
    TEST_AND_HEAL,
    build_master_graph,
    route_next,
)
from app.agents.MasterAgent.MasterNodes import NOTHING_FOUND_MESSAGE, STAGE_STARTED_MESSAGES, MasterNodes
from app.agents.MasterAgent.MasterState import MasterState, initial_state
from app.core.exceptions import AnalysisError, ErrorMessages, NotFoundError, SandboxError
from app.models.analyzerModel import (
    AnalyzerFindingsModel,
    Confidence,
    FactModel,
    FactSheetModel,
    FactSource,
    ImportantPathModel,
)
from app.models.llmModel import LlmConfigModel
from app.models.projectProfileModel import ProjectProfileModel
from app.models.runModel import (
    RunEventLevel,
    RunEventType,
    RunModel,
    RunOutputsModel,
    RunStage,
    RunStatus,
)

CONTAINER_ID = "c0ffee"
LLM_MODEL = "openai/gpt-test"
LLM_KEY = "sk-very-secret"
AGENT_EVENT_TYPES = {
    RunEventType.AGENT_ACTION,
    RunEventType.AGENT_OBSERVATION,
    RunEventType.AGENT_MESSAGE,
    RunEventType.AGENT_ERROR,
}


def _code_fact(value: str | list[str] | None) -> FactModel:
    return FactModel(
        value=value,
        source=FactSource.CODE,
        evidence=["pom.xml"] if value else [],
        confidence=Confidence.HIGH if value else Confidence.LOW,
    )


def _llm_fact(value: str | list[str]) -> FactModel:
    return FactModel(value=value, source=FactSource.LLM, evidence=["pom.xml"], confidence=Confidence.MEDIUM)


FACT_SHEET = FactSheetModel(
    total_files=12,
    language_files={"Java": 8, "Gherkin": 2},
    primary_language=_code_fact("Java"),
    build_tool=_code_fact("Maven"),
    test_frameworks=_code_fact(["TestNG"]),
    automation_tools=_code_fact(["Selenium"]),
    bdd_tool=_code_fact("Cucumber"),
    marker_files=["pom.xml", "testng.xml"],
    test_dirs=["src/test"],
    feature_file_count=2,
    top_level_tree=["pom.xml", "src/", "src/test/"],
)

EMPTY_FACT_SHEET = FactSheetModel(
    total_files=0,
    language_files={},
    primary_language=_code_fact(None),
    build_tool=_code_fact(None),
    test_frameworks=_code_fact(None),
    automation_tools=_code_fact(None),
    bdd_tool=_code_fact(None),
    marker_files=[],
    test_dirs=[],
    feature_file_count=0,
    top_level_tree=[],
)

FINDINGS = AnalyzerFindingsModel(
    project_summary=_llm_fact("UI tests for a web shop"),
    architecture_pattern=_llm_fact("Page Object Model"),
    test_command=_llm_fact("mvn test"),
    reporting_tools=_llm_fact(["Allure"]),
    important_paths=[ImportantPathModel(path="src/test/java/pages", role="page objects")],
    open_questions=["Which environment do the tests target?", "Is the grid URL configurable?"],
)


def _ids() -> MasterState:
    return MasterState(
        run_id=str(ObjectId()), org_id=str(ObjectId()), project_id=str(ObjectId()), model_connection_id=None
    )


async def _wait_until(condition: Callable[[], bool], timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- graph routing (fake nodes)


class FakeNodes:
    """Records the order the nodes run in; each returns the output its stage would save."""

    def __init__(self, workdir: Path, fail_in: str | None = None) -> None:
        self.workdir = workdir
        self.fail_in = fail_in
        self.calls: list[str] = []

    async def prepare_workspace(self, state: MasterState) -> MasterState:
        self._record(PREPARE_WORKSPACE)
        folder = self.workdir / "project"
        folder.mkdir(exist_ok=True)
        return {"project_dir": str(folder)}

    async def collect_facts(self, state: MasterState) -> MasterState:
        self._record(COLLECT_FACTS)
        return {"fact_sheet": FACT_SHEET.model_dump(mode="json")}

    async def analyze(self, state: MasterState) -> MasterState:
        self._record(ANALYZE)
        return {"findings": FINDINGS.model_dump(mode="json"), "llm_model": LLM_MODEL}

    async def save_profile(self, state: MasterState) -> MasterState:
        self._record(SAVE_PROFILE)
        return {"profile_id": "profile-1"}

    async def generate_tests(self, state: MasterState) -> MasterState:
        self._record(GENERATE_TESTS)
        return {"generation": {"selector": "@ngauto"}}

    async def test_and_heal(self, state: MasterState) -> MasterState:
        self._record(TEST_AND_HEAL)
        return {"test_report": {"stop_reason": "passed"}}

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name == self.fail_in:
            raise AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason="no JSON block"))


def _existing_folder(tmp_path: Path) -> str:
    folder = tmp_path / "project"
    _write_cited_files(folder)
    return str(folder)


def _write_cited_files(folder: Path) -> None:
    """The files FINDINGS cites, so the evidence check keeps them."""
    (folder / "src" / "test" / "java" / "pages").mkdir(parents=True, exist_ok=True)
    (folder / "pom.xml").write_text("<project/>", encoding="utf-8")


@pytest.mark.anyio
async def test_new_run_goes_through_every_stage_in_order(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)
    ids = _ids()

    final = await build_master_graph(nodes).ainvoke(ids)

    assert nodes.calls == [PREPARE_WORKSPACE, COLLECT_FACTS, ANALYZE, SAVE_PROFILE]
    assert final == {
        **ids,
        "project_dir": str(tmp_path / "project"),
        "fact_sheet": FACT_SHEET.model_dump(mode="json"),
        "findings": FINDINGS.model_dump(mode="json"),
        "llm_model": LLM_MODEL,
        "profile_id": "profile-1",
    }


@pytest.mark.anyio
async def test_resumed_run_skips_finished_stages(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)
    state = {**_ids(), "project_dir": _existing_folder(tmp_path), "fact_sheet": FACT_SHEET.model_dump(mode="json")}

    await build_master_graph(nodes).ainvoke(state)

    assert nodes.calls == [ANALYZE, SAVE_PROFILE]


@pytest.mark.anyio
async def test_missing_project_folder_is_prepared_again(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)
    state = {**_ids(), "project_dir": str(tmp_path / "deleted"), "fact_sheet": FACT_SHEET.model_dump(mode="json")}

    final = await build_master_graph(nodes).ainvoke(state)

    assert nodes.calls == [PREPARE_WORKSPACE, ANALYZE, SAVE_PROFILE]
    assert final["project_dir"] == str(tmp_path / "project")


@pytest.mark.anyio
async def test_saving_the_profile_does_not_need_the_project_folder(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)
    state = {
        **_ids(),
        "project_dir": str(tmp_path / "deleted"),
        "fact_sheet": FACT_SHEET.model_dump(mode="json"),
        "findings": FINDINGS.model_dump(mode="json"),
        "llm_model": LLM_MODEL,
    }

    await build_master_graph(nodes).ainvoke(state)

    assert nodes.calls == [SAVE_PROFILE]


@pytest.mark.anyio
async def test_finished_run_runs_nothing(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)
    state = {
        **_ids(),
        "project_dir": str(tmp_path / "deleted"),
        "fact_sheet": FACT_SHEET.model_dump(mode="json"),
        "findings": FINDINGS.model_dump(mode="json"),
        "llm_model": LLM_MODEL,
        "profile_id": "profile-1",
    }

    final = await build_master_graph(nodes).ainvoke(state)

    assert nodes.calls == []
    assert final == state


@pytest.mark.anyio
async def test_node_error_propagates_and_stops_the_graph(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path, fail_in=ANALYZE)

    with pytest.raises(AnalysisError) as error:
        await build_master_graph(nodes).ainvoke(_ids())

    assert error.value.error_code.value == "ANALYSIS_FAILED"
    assert nodes.calls == [PREPARE_WORKSPACE, COLLECT_FACTS, ANALYZE]


@pytest.mark.anyio
async def test_routing_loop_fails_fast(tmp_path: Path) -> None:
    class FolderNeverAppears(FakeNodes):
        async def prepare_workspace(self, state: MasterState) -> MasterState:
            self.calls.append(PREPARE_WORKSPACE)
            return {"project_dir": str(self.workdir / "never-created")}

    nodes = FolderNeverAppears(tmp_path)

    with pytest.raises(GraphRecursionError):
        await build_master_graph(nodes).ainvoke(_ids())

    assert nodes.calls == [PREPARE_WORKSPACE] * MAX_GRAPH_STEPS


@pytest.mark.anyio
async def test_cancelling_the_graph_cancels_the_running_node(tmp_path: Path) -> None:
    class SlowAnalysis(FakeNodes):
        cancelled = False

        async def analyze(self, state: MasterState) -> MasterState:
            self.calls.append(ANALYZE)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return {}

    nodes = SlowAnalysis(tmp_path)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.5):
            await build_master_graph(nodes).ainvoke(_ids())

    assert nodes.calls == [PREPARE_WORKSPACE, COLLECT_FACTS, ANALYZE]
    assert nodes.cancelled


@pytest.mark.parametrize(
    ("outputs", "folder", "expected"),
    [
        ({}, None, PREPARE_WORKSPACE),
        ({}, "existing", COLLECT_FACTS),
        ({"fact_sheet"}, "existing", ANALYZE),
        # Findings without the model that produced them cannot be recorded on a profile.
        ({"fact_sheet", "findings"}, "existing", ANALYZE),
        ({"fact_sheet", "findings", "llm_model"}, "existing", SAVE_PROFILE),
        ({"fact_sheet"}, "deleted", PREPARE_WORKSPACE),
        ({"fact_sheet"}, "", PREPARE_WORKSPACE),
        ({"fact_sheet", "findings", "llm_model"}, "deleted", SAVE_PROFILE),
        ({"fact_sheet", "findings", "llm_model", "profile_id"}, "existing", END),
        ({"fact_sheet", "findings", "llm_model", "profile_id"}, None, END),
    ],
)
def test_route_next(tmp_path: Path, outputs: set[str], folder: str | None, expected: str) -> None:
    values: dict[str, Any] = {
        "fact_sheet": FACT_SHEET.model_dump(mode="json"),
        "findings": FINDINGS.model_dump(mode="json"),
        "llm_model": LLM_MODEL,
        "profile_id": "profile-1",
    }
    state: dict[str, Any] = {**_ids(), **{key: values[key] for key in outputs}}
    if folder is not None:
        state["project_dir"] = {"existing": _existing_folder(tmp_path), "deleted": str(tmp_path / "gone"), "": ""}[folder]

    assert route_next(state) == expected  # type: ignore[arg-type]


def test_initial_state_holds_the_ids_and_every_saved_output() -> None:
    now = datetime.now(UTC)
    run = RunModel(
        id=str(ObjectId()),
        org_id=str(ObjectId()),
        project_id=str(ObjectId()),
        created_by=str(ObjectId()),
        model_connection_id=str(ObjectId()),
        status=RunStatus.RUNNING,
        stage=RunStage.ANALYZING,
        outputs=RunOutputsModel(project_dir="/data/runs/x/project", fact_sheet=EMPTY_FACT_SHEET),
        sandbox_container_id="old-container",
        created_at=now,
        updated_at=now,
    )

    state = initial_state(run)

    assert state == {
        "run_id": run.id,
        "org_id": run.org_id,
        "project_id": run.project_id,
        "model_connection_id": run.model_connection_id,
        "mode": "analyze",
        "test_selector": None,
        "test_data_id": None,
        "run_scope": "generated",
        "project_dir": "/data/runs/x/project",
        "fact_sheet": EMPTY_FACT_SHEET.model_dump(mode="json"),
    }
    # Unset facts keep their explicit None, so the dict validates back to the same model.
    assert state["fact_sheet"]["bdd_tool"]["value"] is None
    assert FactSheetModel.model_validate(state["fact_sheet"]) == EMPTY_FACT_SHEET


# --------------------------------------------------------------------------- real nodes (fake services)


@dataclass
class SavedEvent:
    type: RunEventType
    message: str
    level: RunEventLevel
    data: dict[str, Any]


class FakeRunService:
    def __init__(self, fail_agent_events: bool = False, max_event_delay: float = 0) -> None:
        self.fail_agent_events = fail_agent_events
        self.max_event_delay = max_event_delay
        self.random = random.Random(7)
        self.stages: list[RunStage] = []
        self.events: list[SavedEvent] = []
        self.outputs: dict[str, Any] = {}
        self.containers: list[str | None] = []

    async def set_stage(self, run_id: str, stage: RunStage) -> None:
        self.stages.append(stage)

    async def append_event(
        self,
        run_id: str,
        org_id: str,
        type: RunEventType,
        message: str,
        level: RunEventLevel = RunEventLevel.INFO,
        data: dict[str, Any] | None = None,
    ) -> None:
        if type in AGENT_EVENT_TYPES:
            if self.fail_agent_events:
                raise RuntimeError("database is down")
            if self.max_event_delay:
                await asyncio.sleep(self.random.uniform(0, self.max_event_delay))
        self.events.append(SavedEvent(type, message, level, data or {}))

    async def save_outputs(self, run_id: str, **outputs: Any) -> None:
        self.outputs.update(outputs)

    async def set_sandbox_container(self, run_id: str, container_id: str | None) -> None:
        self.containers.append(container_id)

    def event_types(self) -> list[RunEventType]:
        return [event.type for event in self.events]

    def completed_event(self, stage: RunStage) -> SavedEvent:
        return next(e for e in self.events if e.type == RunEventType.STAGE_COMPLETED and e.data["stage"] == stage.value)


@dataclass
class FakeSession:
    workspace: Any
    container_id: str


class FakeSandbox:
    """Stands in for SandboxService: no Docker, the project folder lives under tmp_path."""

    def __init__(self, workdir: Path, start_error: Exception | None = None, hold_start: bool = False) -> None:
        self.workdir = workdir
        self.start_error = start_error
        self.workspace = object()
        self.opened_with: Path | None = None
        self.opening = threading.Event()
        # Set by the test to let a held container start.
        self.start_gate = threading.Event()
        if not hold_start:
            self.start_gate.set()
        self.on_started_error: BaseException | None = None
        self.cleaned_up = threading.Event()
        self.stopped: list[str] = []
        self.removed: list[str] = []
        self.writable_org: str | None = None

    def prepare_project_dir(self, project_id: str, run_id: str) -> Path:
        folder = self.workdir / "runs" / run_id / "project"
        _write_cited_files(folder)
        return folder

    @contextmanager
    def open_writable(
        self, project_dir: Path, org_id: str, on_started: Callable[[str], None] | None = None
    ) -> Iterator[FakeSession]:
        self.writable_org = org_id
        with self.open_readonly(project_dir, on_started) as session:
            yield session

    @contextmanager
    def open_readonly(self, project_dir: Path, on_started: Callable[[str], None] | None = None) -> Iterator[FakeSession]:
        self.opened_with = project_dir
        self.opening.set()
        self.start_gate.wait(5)
        if self.start_error is not None:
            raise self.start_error
        try:
            if on_started is not None:
                try:
                    on_started(CONTAINER_ID)
                except BaseException as error:
                    self.on_started_error = error
                    raise
            yield FakeSession(workspace=self.workspace, container_id=CONTAINER_ID)
        finally:
            self.cleaned_up.set()

    def stop_container(self, container_id: str) -> None:
        self.stopped.append(container_id)

    def remove_run_dir(self, run_id: str) -> None:
        self.removed.append(run_id)


@dataclass
class FakeAnalyzer:
    """Emits agent events from its worker thread like the real analyzer, then returns FINDINGS (or raises)."""

    events: int = 3
    error: Exception | None = None
    hold: bool = False
    events_after_hold: int = 0
    llm_config: LlmConfigModel | None = None
    received: tuple[Any, FactSheetModel] | None = None
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    created: int = 0

    def factory(self, llm_config: LlmConfigModel) -> "FakeAnalyzer":
        self.llm_config = llm_config
        self.created += 1
        return self

    def analyze(
        self, workspace: Any, fact_sheet: FactSheetModel, on_event: Callable[[AgentEvent], None] | None = None,
        control: Any = None,
    ) -> AnalyzerFindingsModel:
        assert on_event is not None
        try:
            self.received = (workspace, fact_sheet)
            for number in range(self.events):
                on_event(_agent_event(f"glob step {number}"))
            self.started.set()
            if self.hold:
                self.release.wait(5)
                for number in range(self.events_after_hold):
                    on_event(_agent_event(f"after release {number}"))
            if self.error is not None:
                raise self.error
            return FINDINGS
        finally:
            self.finished.set()


def _agent_event(message: str) -> AgentEvent:
    return AgentEvent(type=RunEventType.AGENT_ACTION, level=RunEventLevel.INFO, message=message, data={"tool": "glob"})


class FakeModelConnections:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str | None, str]] = []

    async def get_llm_config(self, model_connection_id: str | None, org_id: str) -> LlmConfigModel:
        self.calls.append((model_connection_id, org_id))
        if self.error is not None:
            raise self.error
        return LlmConfigModel(model=LLM_MODEL, base_url="https://gw.example/v1", api_key=SecretStr(LLM_KEY))


class FakeProfiles:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, **fields: Any) -> ProjectProfileModel:
        self.created.append(fields)
        return ProjectProfileModel(id=str(ObjectId()), created_at=datetime.now(UTC), **fields)


@dataclass
class NodeSetup:
    nodes: MasterNodes
    runs: FakeRunService
    sandbox: FakeSandbox
    analyzer: FakeAnalyzer
    connections: FakeModelConnections
    profiles: FakeProfiles
    collector_threads: list[int]


def _setup(
    tmp_path: Path,
    *,
    runs: FakeRunService | None = None,
    sandbox: FakeSandbox | None = None,
    analyzer: FakeAnalyzer | None = None,
    connections: FakeModelConnections | None = None,
    fact_sheet: FactSheetModel = FACT_SHEET,
) -> NodeSetup:
    runs = runs or FakeRunService()
    sandbox = sandbox or FakeSandbox(tmp_path)
    analyzer = analyzer or FakeAnalyzer()
    connections = connections or FakeModelConnections()
    profiles = FakeProfiles()
    collector_threads: list[int] = []

    def collect(project_dir: Path) -> FactSheetModel:
        assert (project_dir / "pom.xml").is_file()
        collector_threads.append(threading.get_ident())
        return fact_sheet

    nodes = MasterNodes(
        run_service=runs,  # type: ignore[arg-type]
        sandbox_service=sandbox,  # type: ignore[arg-type]
        model_connection_service=connections,  # type: ignore[arg-type]
        profile_service=profiles,  # type: ignore[arg-type]
        fact_collector=collect,
        analyzer_factory=analyzer.factory,  # type: ignore[arg-type]
    )
    return NodeSetup(nodes, runs, sandbox, analyzer, connections, profiles, collector_threads)


def _analysis_state(tmp_path: Path) -> MasterState:
    return MasterState(
        **_ids(), project_dir=_existing_folder(tmp_path), fact_sheet=FACT_SHEET.model_dump(mode="json")
    )


@pytest.mark.anyio
async def test_full_run_through_the_real_nodes(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    ids = _ids()

    final = await build_master_graph(setup.nodes).ainvoke(ids)

    runs = setup.runs
    assert runs.stages == [
        RunStage.PREPARING_WORKSPACE,
        RunStage.COLLECTING_FACTS,
        RunStage.ANALYZING,
        RunStage.SAVING_PROFILE,
    ]
    S, C, A = RunEventType.STAGE_STARTED, RunEventType.STAGE_COMPLETED, RunEventType.AGENT_ACTION
    assert runs.event_types() == [S, C, S, C, S, A, A, A, C, S, C]
    assert all(event.level == RunEventLevel.INFO for event in runs.events)
    assert [e.message for e in runs.events if e.type == S] == [STAGE_STARTED_MESSAGES[stage] for stage in runs.stages]

    project_dir = tmp_path / "runs" / ids["run_id"] / "project"
    profile_id = runs.outputs["profile_id"]
    assert runs.outputs == {
        "project_dir": str(project_dir),
        "generation": None,
        "test_attempts": None,
        "fact_sheet": FACT_SHEET,
        "findings": FINDINGS,
        "llm_model": LLM_MODEL,
        "profile_id": profile_id,
    }
    assert final == {
        **ids,
        "project_dir": str(project_dir),
        "generation": None,
        "test_attempts": None,
        "fact_sheet": FACT_SHEET.model_dump(mode="json"),
        "findings": FINDINGS.model_dump(mode="json"),
        "llm_model": LLM_MODEL,
        "profile_id": profile_id,
    }

    # The container is recorded while the analyzer works, and cleared afterwards.
    assert runs.containers == [CONTAINER_ID, None]
    assert setup.sandbox.opened_with == project_dir
    assert setup.sandbox.cleaned_up.is_set()
    assert setup.connections.calls == [(None, ids["org_id"])]
    assert setup.analyzer.llm_config is not None and setup.analyzer.llm_config.model == LLM_MODEL
    assert setup.analyzer.received == (setup.sandbox.workspace, FACT_SHEET)
    # Blocking work ran in a worker thread, not on the event loop.
    assert setup.collector_threads and setup.collector_threads[0] != threading.get_ident()

    assert setup.profiles.created == [{
        "org_id": ids["org_id"],
        "project_id": ids["project_id"],
        "run_id": ids["run_id"],
        "fact_sheet": FACT_SHEET,
        "findings": FINDINGS,
        "llm_model": LLM_MODEL,
    }]
    assert runs.completed_event(RunStage.SAVING_PROFILE).data["profile_id"] == profile_id

    # Events are shown to API callers: no key, no server path (repr escapes backslashes, so match a path part).
    timeline = repr([(event.message, event.data) for event in runs.events])
    assert LLM_KEY not in timeline
    assert tmp_path.name not in timeline and ids["run_id"] not in timeline


@pytest.mark.anyio
async def test_stage_summaries_are_short_plain_english(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    await build_master_graph(setup.nodes).ainvoke(_ids())

    facts = setup.runs.completed_event(RunStage.COLLECTING_FACTS)
    assert facts.message == "Found Java · Maven · TestNG, Selenium, Cucumber"
    assert facts.data == {
        "stage": "collecting_facts",
        "total_files": 12,
        "primary_language": "Java",
        "build_tool": "Maven",
        "test_frameworks": ["TestNG"],
        "automation_tools": ["Selenium"],
        "bdd_tool": "Cucumber",
    }
    analysis = setup.runs.completed_event(RunStage.ANALYZING)
    assert analysis.message == "Analysis finished · Page Object Model · 1 important path · 2 open questions"
    assert analysis.data == {"stage": "analyzing", "llm_model": LLM_MODEL, "important_paths": 1, "open_questions": 2}
    assert setup.runs.completed_event(RunStage.PREPARING_WORKSPACE).message == "Project files are ready"


@pytest.mark.anyio
async def test_project_without_known_stack_says_so(tmp_path: Path) -> None:
    setup = _setup(tmp_path, fact_sheet=EMPTY_FACT_SHEET)
    state = MasterState(**_ids(), project_dir=_existing_folder(tmp_path))
    (tmp_path / "project" / "pom.xml").write_text("", encoding="utf-8")

    result = await setup.nodes.collect_facts(state)

    assert result == {"fact_sheet": EMPTY_FACT_SHEET.model_dump(mode="json")}
    assert setup.runs.completed_event(RunStage.COLLECTING_FACTS).message == NOTHING_FOUND_MESSAGE


@pytest.mark.anyio
async def test_agent_events_are_saved_in_order_before_the_stage_summary(tmp_path: Path) -> None:
    # Random delays per write: without ordering, later events would overtake earlier ones.
    setup = _setup(tmp_path, runs=FakeRunService(max_event_delay=0.005), analyzer=FakeAnalyzer(events=40))

    await setup.nodes.analyze(_analysis_state(tmp_path))

    agent_messages = [e.message for e in setup.runs.events if e.type in AGENT_EVENT_TYPES]
    assert agent_messages == [f"glob step {number}" for number in range(40)]
    assert setup.runs.event_types()[-1] == RunEventType.STAGE_COMPLETED
    assert setup.runs.events[1].data == {"tool": "glob"}


@pytest.mark.anyio
async def test_analysis_error_clears_the_container_and_propagates(tmp_path: Path) -> None:
    error = AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason="no JSON block"))
    setup = _setup(tmp_path, analyzer=FakeAnalyzer(events=2, error=error))

    with pytest.raises(AnalysisError):
        await setup.nodes.analyze(_analysis_state(tmp_path))

    assert setup.runs.containers == [CONTAINER_ID, None]
    assert setup.sandbox.cleaned_up.is_set()
    assert "findings" not in setup.runs.outputs
    # The agent's steps up to the failure stay on the timeline; no summary is reported.
    assert setup.runs.event_types() == [RunEventType.STAGE_STARTED, RunEventType.AGENT_ACTION, RunEventType.AGENT_ACTION]


@pytest.mark.anyio
async def test_sandbox_start_failure_propagates(tmp_path: Path) -> None:
    error = SandboxError(ErrorMessages.SANDBOX_FAILED.format(reason="Docker is not running"))
    setup = _setup(tmp_path, sandbox=FakeSandbox(tmp_path, start_error=error))

    with pytest.raises(SandboxError):
        await setup.nodes.analyze(_analysis_state(tmp_path))

    assert setup.analyzer.created == 0
    assert setup.runs.containers == [None]
    assert "findings" not in setup.runs.outputs


@pytest.mark.anyio
async def test_unusable_model_connection_stops_before_the_sandbox(tmp_path: Path) -> None:
    connections = FakeModelConnections(error=NotFoundError(ErrorMessages.MODEL_CONNECTION_NOT_FOUND))
    setup = _setup(tmp_path, connections=connections)
    state = _analysis_state(tmp_path)
    state["model_connection_id"] = str(ObjectId())

    with pytest.raises(NotFoundError):
        await setup.nodes.analyze(state)

    assert connections.calls == [(state["model_connection_id"], state["org_id"])]
    assert setup.sandbox.opened_with is None
    assert setup.runs.containers == []


@pytest.mark.anyio
async def test_failing_event_writes_do_not_break_the_analysis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    setup = _setup(tmp_path, runs=FakeRunService(fail_agent_events=True))

    result = await setup.nodes.analyze(_analysis_state(tmp_path))

    assert result == {"findings": FINDINGS.model_dump(mode="json"), "llm_model": LLM_MODEL}
    assert setup.runs.outputs == {"findings": FINDINGS, "llm_model": LLM_MODEL}
    assert setup.runs.containers == [CONTAINER_ID, None]
    assert caplog.text.count("Could not save an agent event") == 3


@pytest.mark.anyio
async def test_cancelled_analysis_keeps_the_container_for_the_executor(tmp_path: Path) -> None:
    analyzer = FakeAnalyzer(events=1, hold=True, events_after_hold=3)
    setup = _setup(tmp_path, analyzer=analyzer)
    task = asyncio.create_task(setup.nodes.analyze(_analysis_state(tmp_path)))
    await _wait_until(analyzer.started.is_set)
    await _wait_until(lambda: RunEventType.AGENT_ACTION in setup.runs.event_types())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Timeout or shutdown: the thread still runs, so the container must stay recorded for the executor to stop.
    assert setup.runs.containers == [CONTAINER_ID]
    analyzer.release.set()
    assert await asyncio.to_thread(setup.sandbox.cleaned_up.wait, 5)
    await asyncio.sleep(0.05)
    # Events of the abandoned analysis are dropped; nothing is saved for it.
    assert [e.message for e in setup.runs.events if e.type in AGENT_EVENT_TYPES] == ["glob step 0"]
    assert "findings" not in setup.runs.outputs
    assert setup.runs.containers == [CONTAINER_ID]


@pytest.mark.anyio
async def test_container_that_starts_after_the_run_stopped_is_refused(tmp_path: Path) -> None:
    sandbox = FakeSandbox(tmp_path, hold_start=True)
    setup = _setup(tmp_path, sandbox=sandbox)
    task = asyncio.create_task(setup.nodes.analyze(_analysis_state(tmp_path)))
    await _wait_until(sandbox.opening.is_set)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.start_gate.set()

    assert await asyncio.to_thread(sandbox.cleaned_up.wait, 5)
    # The late container is not recorded on the stopped run, and no analysis starts in it.
    assert sandbox.on_started_error is not None
    assert setup.runs.containers == []
    assert setup.analyzer.created == 0


@pytest.mark.anyio
async def test_save_profile_from_a_resumed_state(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    ids = _ids()
    state = MasterState(
        **ids,
        fact_sheet=FACT_SHEET.model_dump(mode="json"),
        findings=FINDINGS.model_dump(mode="json"),
        llm_model=LLM_MODEL,
    )

    result = await setup.nodes.save_profile(state)

    created = setup.profiles.created[0]
    assert created["fact_sheet"] == FACT_SHEET and created["findings"] == FINDINGS
    assert result == {"profile_id": setup.runs.outputs["profile_id"]}
    assert setup.runs.stages == [RunStage.SAVING_PROFILE]
    assert setup.runs.event_types() == [RunEventType.STAGE_STARTED, RunEventType.STAGE_COMPLETED]


@pytest.mark.anyio
async def test_evidence_that_does_not_exist_is_removed_and_reported(tmp_path: Path) -> None:
    from app.agents.MasterAgent import MasterNodes as master_nodes_module
    from app.agents.AnalyzerAgent.EvidenceVerifier import verify_evidence

    folder = tmp_path / "project"
    _write_cited_files(folder)
    (folder / "src" / "test" / "java" / "pages").rmdir()  # the cited important path is now missing

    findings, removed = verify_evidence(FINDINGS, folder)

    assert removed == ["src/test/java/pages"]
    assert findings.important_paths == []
    assert master_nodes_module.EVIDENCE_REMOVED_MESSAGE.format(count=1).startswith("Removed 1 cited file path")
