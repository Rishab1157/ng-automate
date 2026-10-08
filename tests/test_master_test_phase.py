"""The master's test phase: routing in test mode, resume, and the test-and-heal loop with fake runner and healer.

No Docker, no LLM. The fakes (run service, sandbox, model connections) are the ones the master graph tests use.
"""

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from bson import ObjectId
from langgraph.graph import END

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.MasterAgent.MasterGraph import (
    COLLECT_FACTS,
    GENERATE_TESTS,
    PREPARE_WORKSPACE,
    TEST_AND_HEAL,
    build_master_graph,
    route_next,
)
from app.agents.MasterAgent.MasterNodes import SELECTOR_NOT_FOUND_MESSAGE, MasterNodes
from app.agents.MasterAgent.MasterState import MasterState, initial_state
from app.agents.RunnerAgent.TestCommandResolver import resolve_test_command
from app.core.exceptions import ErrorMessages, GenerationError, ValidationError
from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import FileChangeModel, HealOutcomeModel
from app.models.healMemoryModel import HealMemoryDbModel, HealMemoryModel, HealMemoryResult, memory_point_id
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestDataStatus, TestStepModel
from app.models.testGeneratorModel import GenerationOutcomeModel, SkippedCaseModel
from app.models.runModel import (
    RunEventLevel,
    RunEventType,
    RunMode,
    RunModel,
    RunOutputsModel,
    RunStage,
    RunStatus,
    TestAttemptModel,
    TestReportModel,
    TestStopReason,
)
from app.models.testRunModel import (
    HEALABLE_KINDS,
    FailureClassificationModel,
    FailureKind,
    TestCommandModel,
    TestRunResultModel,
)
from tests.test_master_graph import (
    CONTAINER_ID,
    FACT_SHEET,
    FakeModelConnections,
    FakeNodes,
    FakeProfiles,
    FakeRunService,
    FakeSandbox,
    _code_fact,
    _existing_folder,
    _ids,
)

DEPENDENCY_EVIDENCE = ["Could not find artifact org.testng:testng:jar:99.0.0"]


def _result(kind: FailureKind, evidence: list[str] | None = None, passed: int = 0, failed: int = 0) -> TestRunResultModel:
    return TestRunResultModel(
        command="mvn -B -ntp test", exit_code=0 if kind == FailureKind.PASSED else 1, duration_seconds=5.0,
        total=passed + failed, passed=passed, failed=failed, errors=0, skipped=0,
        output_tail="[ERROR] Could not resolve dependencies for project demo" if kind != FailureKind.PASSED else "",
        classification=FailureClassificationModel(
            kind=kind, healable=kind in HEALABLE_KINDS, reason=f"{kind.value} reason", evidence=evidence or []
        ),
    )


PASSED = _result(FailureKind.PASSED, passed=2)
DEPENDENCY = _result(FailureKind.DEPENDENCY_FAILURE, DEPENDENCY_EVIDENCE)
FIX = HealOutcomeModel(summary="Set TestNG to 7.10.2", changes=[FileChangeModel(path="pom.xml", change="modified", diff="-99\n+7")])


@dataclass
class FakeRunner:
    results: list[TestRunResultModel]
    calls: list[tuple[Any, Path, TestCommandModel]] = field(default_factory=list)
    threads: list[int] = field(default_factory=list)

    def run(self, workspace: Any, project_dir: Path, command: TestCommandModel) -> TestRunResultModel:
        self.calls.append((workspace, project_dir, command))
        self.threads.append(threading.get_ident())
        return self.results.pop(0)


@dataclass
class FakeHealer:
    outcomes: list[HealOutcomeModel]
    calls: list[dict[str, Any]] = field(default_factory=list)
    llm_models: list[str] = field(default_factory=list)

    def factory(self, llm_config: Any) -> "FakeHealer":
        self.llm_models.append(llm_config.model)
        return self

    def heal(self, workspace: Any, project_dir: Path, result: TestRunResultModel, command: TestCommandModel,
             fact_sheet: FactSheetModel, on_event: Any = None, previous: Any = (), control: Any = None,
             time_limit: float | None = None, memories: Any = ()) -> HealOutcomeModel:
        self.calls.append({"workspace": workspace, "result": result, "previous": list(previous),
                           "time_limit": time_limit, "memories": list(memories)})
        on_event(AgentEvent(RunEventType.AGENT_ACTION, RunEventLevel.INFO, "str_replace pom.xml", {"tool": "file_editor"}))
        return self.outcomes.pop(0)


@dataclass
class FakeMemory:
    """The healer's memory: hands out the given memories and records what is remembered."""

    memories: list[HealMemoryModel] = field(default_factory=list)
    recalls: list[dict[str, Any]] = field(default_factory=list)
    remembered: list[tuple[str, HealMemoryDbModel]] = field(default_factory=list)

    async def recall(self, *, org_id: str, run_id: str, result: TestRunResultModel) -> list[HealMemoryModel]:
        self.recalls.append({"org_id": org_id, "run_id": run_id, "result": result})
        return list(self.memories)

    async def remember(self, point_id: str, memory: HealMemoryDbModel) -> bool:
        self.remembered.append((point_id, memory))
        return True


@dataclass
class Setup:
    nodes: MasterNodes
    runs: FakeRunService
    sandbox: FakeSandbox
    runner: FakeRunner
    healer: FakeHealer
    memory: FakeMemory


def _setup(tmp_path: Path, results: list[TestRunResultModel], outcomes: list[HealOutcomeModel] | None = None,
           heal_budget_seconds: float = 3600, memories: list[HealMemoryModel] | None = None) -> Setup:
    runs, sandbox = FakeRunService(), FakeSandbox(tmp_path)
    runner, healer = FakeRunner(list(results)), FakeHealer(list(outcomes or []))
    memory = FakeMemory(list(memories or []))
    nodes = MasterNodes(
        run_service=runs,  # type: ignore[arg-type]
        sandbox_service=sandbox,  # type: ignore[arg-type]
        model_connection_service=FakeModelConnections(),  # type: ignore[arg-type]
        profile_service=FakeProfiles(),  # type: ignore[arg-type]
        runner_factory=lambda: runner,  # type: ignore[arg-type,return-value]
        healer_factory=healer.factory,  # type: ignore[arg-type]
        heal_budget_seconds=heal_budget_seconds,
        heal_memory_service=memory,  # type: ignore[arg-type]
    )
    return Setup(nodes, runs, sandbox, runner, healer, memory)


def _test_state(tmp_path: Path, **extra: Any) -> MasterState:
    return MasterState(
        **_ids(), mode=RunMode.TEST.value, test_selector=None, project_dir=_existing_folder(tmp_path),
        fact_sheet=FACT_SHEET.model_dump(mode="json"), profile_id="profile-1", **extra,
    )


# --------------------------------------------------------------------------- routing


def test_test_mode_runs_the_tests_after_the_profile(tmp_path: Path) -> None:
    assert route_next(_test_state(tmp_path)) == TEST_AND_HEAL


def test_test_mode_ends_once_the_report_is_saved(tmp_path: Path) -> None:
    assert route_next(_test_state(tmp_path, test_report={"stop_reason": "passed"})) == END


def test_analyze_mode_ends_at_the_profile(tmp_path: Path) -> None:
    assert route_next({**_test_state(tmp_path), "mode": RunMode.ANALYZE.value}) == END


def test_test_mode_prepares_a_missing_project_folder_first(tmp_path: Path) -> None:
    assert route_next({**_test_state(tmp_path), "project_dir": str(tmp_path / "gone")}) == PREPARE_WORKSPACE


@pytest.mark.anyio
async def test_reused_profile_skips_the_analysis(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)

    await build_master_graph(nodes).ainvoke({**_ids(), "mode": RunMode.TEST.value, "profile_id": "profile-0"})

    assert nodes.calls == [PREPARE_WORKSPACE, COLLECT_FACTS, TEST_AND_HEAL]


def _run(outputs: RunOutputsModel, mode: RunMode = RunMode.TEST) -> RunModel:
    now = datetime.now(UTC)
    return RunModel(
        id=str(ObjectId()), org_id=str(ObjectId()), project_id=str(ObjectId()), created_by=str(ObjectId()),
        mode=mode, test_selector="@smoke", status=RunStatus.RUNNING, stage=RunStage.HEALING, outputs=outputs,
        created_at=now, updated_at=now,
    )


def test_interrupted_test_phase_starts_over_on_a_fresh_copy() -> None:
    run = _run(RunOutputsModel(project_dir="/data/runs/x/project", profile_id=str(ObjectId()),
                               test_attempts=[TestAttemptModel(number=1, result=DEPENDENCY)]))

    state = initial_state(run)

    assert "test_attempts" not in state
    assert "project_dir" not in state
    assert state["mode"] == "test" and state["test_selector"] == "@smoke"


def test_finished_test_phase_is_kept_on_resume() -> None:
    report = TestReportModel(outcome=FailureKind.PASSED, stop_reason=TestStopReason.PASSED, runs=1, heals=0,
                             total=2, passed=2, failed=0, errors=0, skipped=0)
    run = _run(RunOutputsModel(project_dir="/data/runs/x/project", test_report=report,
                               test_attempts=[TestAttemptModel(number=1, result=PASSED)]))

    state = initial_state(run)

    assert state["project_dir"] == "/data/runs/x/project"
    assert len(state["test_attempts"]) == 1


# --------------------------------------------------------------------------- test_and_heal()


@pytest.mark.anyio
async def test_tests_that_pass_need_no_healer(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [PASSED])
    state = _test_state(tmp_path)

    update = await setup.nodes.test_and_heal(state)

    runs = setup.runs
    assert setup.healer.calls == []
    assert runs.stages == [RunStage.RUNNING_TESTS]
    assert runs.containers == [CONTAINER_ID, None]
    assert setup.sandbox.writable_org == state["org_id"]
    assert setup.sandbox.cleaned_up.is_set()
    assert runs.outputs["test_command"] == resolve_test_command(FACT_SHEET)
    assert [a.number for a in runs.outputs["test_attempts"]] == [1]
    report = runs.outputs["test_report"]
    assert (report.stop_reason, report.runs, report.heals, report.passed) == (TestStopReason.PASSED, 1, 0, 2)
    assert runs.events[-1].message == "All 2 tests passed"
    assert update["test_report"]["stop_reason"] == "passed"
    assert runs.completed_event(RunStage.RUNNING_TESTS).message == "Run 1: 2 passed · passed reason"


@pytest.mark.anyio
async def test_dependency_failure_is_healed_then_the_tests_run_again_in_the_same_sandbox(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, PASSED], [FIX])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    runs = setup.runs
    assert runs.stages == [RunStage.RUNNING_TESTS, RunStage.HEALING, RunStage.RUNNING_TESTS]
    # One sandbox for the whole loop: what the healer installed is still there for the rerun.
    [first, second] = setup.runner.calls
    assert first[0] is second[0] is setup.healer.calls[0]["workspace"] is setup.sandbox.workspace
    assert runs.containers == [CONTAINER_ID, None]
    [call] = setup.healer.calls
    assert call["result"] == DEPENDENCY and call["previous"] == [] and call["time_limit"] == 3600
    attempts = runs.outputs["test_attempts"]
    assert [(a.number, a.heal) for a in attempts] == [(1, FIX), (2, None)]
    assert all(a.project_state for a in attempts)  # a fingerprint of the files each run tested
    report = runs.outputs["test_report"]
    assert (report.stop_reason, report.heals, report.changed_files) == (TestStopReason.PASSED, 1, ["pom.xml"])
    assert runs.events[-1].message == "All 2 tests passed after 1 fix by the healer"
    # The healer's own steps come before its summary.
    messages = [event.message for event in runs.events]
    assert messages.index("str_replace pom.xml") < messages.index("The healer changed 1 file: pom.xml")


@pytest.mark.anyio
async def test_next_healer_hears_about_the_earlier_fix(tmp_path: Path) -> None:
    rolled_back = HealOutcomeModel(summary="broke the pom", reverted_files=["pom.xml"])
    setup = _setup(tmp_path, [DEPENDENCY, DEPENDENCY, PASSED], [rolled_back, FIX])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert [call["previous"] for call in setup.healer.calls] == [[], [rolled_back]]
    assert setup.runs.outputs["test_report"].stop_reason == TestStopReason.PASSED


@pytest.mark.anyio
async def test_failing_assertions_are_reported_not_healed(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [_result(FailureKind.TEST_FAILURE, passed=3, failed=1)])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert setup.healer.calls == []
    report = setup.runs.outputs["test_report"]
    assert (report.stop_reason, report.outcome, report.failed) == (TestStopReason.NOT_HEALABLE, FailureKind.TEST_FAILURE, 1)


@pytest.mark.anyio
async def test_the_same_failure_again_and_again_stops_the_loop(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, DEPENDENCY, DEPENDENCY], [FIX, FIX])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    report = setup.runs.outputs["test_report"]
    assert (report.stop_reason, report.runs, report.heals) == (TestStopReason.NO_PROGRESS, 3, 2)


@pytest.mark.anyio
async def test_the_healing_budget_is_shared_by_all_heals_of_the_run(tmp_path: Path) -> None:
    build = _result(FailureKind.BUILD_FAILURE, ["cannot find symbol"])
    environment = _result(FailureKind.ENVIRONMENT_FAILURE, ["no display"])
    slow = FIX.model_copy(update={"seconds": 2000.0})
    setup = _setup(tmp_path, [DEPENDENCY, build, environment], [slow, slow.model_copy(update={"seconds": 1600.0})])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert [call["time_limit"] for call in setup.healer.calls] == [3600, 1600]
    report = setup.runs.outputs["test_report"]
    assert (report.stop_reason, report.runs, report.heals) == (TestStopReason.TIMED_OUT, 3, 2)
    assert setup.runs.events[-1].message == "Stopped: the healer used all of its 60 minutes of healing time"


@pytest.mark.anyio
async def test_a_heal_that_runs_out_of_time_stops_at_once(tmp_path: Path) -> None:
    out_of_time = HealOutcomeModel(summary="", error="The healer stopped: the agent did not finish in time",
                                   seconds=600.0)
    setup = _setup(tmp_path, [DEPENDENCY], [out_of_time], heal_budget_seconds=600)

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    report = setup.runs.outputs["test_report"]
    assert (report.stop_reason, report.runs, report.detail) == (
        TestStopReason.TIMED_OUT, 1, "Stopped: the healer used all of its 10 minutes of healing time"
    )


@pytest.mark.anyio
async def test_a_proven_blocker_stops_at_once(tmp_path: Path) -> None:
    blocked = HealOutcomeModel(
        summary="BLOCKED: no network",
        blocker="the sandbox cannot reach the internet\n`curl: (6) Could not resolve host: storage.googleapis.com`",
    )
    setup = _setup(tmp_path, [DEPENDENCY], [blocked])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    runs = setup.runs
    report = runs.outputs["test_report"]
    assert (report.stop_reason, report.runs, report.detail) == (
        TestStopReason.BLOCKED, 1, "the sandbox cannot reach the internet"
    )
    healing = runs.completed_event(RunStage.HEALING)
    assert healing.message == "The healer is blocked: the sandbox cannot reach the internet"
    assert healing.data["blocked"] is True
    assert runs.events[-1].message == "Stopped: the healer is blocked: the sandbox cannot reach the internet"


@pytest.mark.anyio
async def test_healer_failure_ends_the_loop_with_a_report(tmp_path: Path) -> None:
    reason = ErrorMessages.HEAL_FAILED.format(reason="the model could not be reached, try again later")
    stopped = HealOutcomeModel(summary="", error=reason, changes=FIX.changes)
    setup = _setup(tmp_path, [DEPENDENCY], [stopped])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    runs = setup.runs
    report = runs.outputs["test_report"]
    assert (report.stop_reason, report.detail) == (TestStopReason.HEALER_FAILED, reason)
    assert report.changed_files == ["pom.xml"]  # what it changed before stopping is still recorded
    healing = runs.completed_event(RunStage.HEALING)
    assert healing.data["stopped"] is True
    assert healing.message == f"{reason} (kept 1 changed file)"
    assert runs.outputs["test_attempts"][0].heal == stopped
    assert runs.containers == [CONTAINER_ID, None]
    assert setup.sandbox.cleaned_up.is_set()


@pytest.mark.anyio
async def test_test_selector_reaches_the_command(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [PASSED])

    await setup.nodes.test_and_heal({**_test_state(tmp_path), "test_selector": "@smoke"})

    command = setup.runner.calls[0][2]
    assert command == resolve_test_command(FACT_SHEET, "@smoke")
    assert "@smoke" in command.command


@pytest.mark.anyio
async def test_project_that_cannot_run_in_the_sandbox_fails_before_starting_it(tmp_path: Path) -> None:
    uft = FACT_SHEET.model_copy(update={"automation_tools": _code_fact(["UFT"]), "build_tool": _code_fact(None),
                                        "test_frameworks": _code_fact(None), "bdd_tool": _code_fact(None)})
    setup = _setup(tmp_path, [])

    with pytest.raises(ValidationError):
        await setup.nodes.test_and_heal({**_test_state(tmp_path), "fact_sheet": uft.model_dump(mode="json")})

    assert setup.sandbox.opened_with is None
    assert setup.runner.calls == []


# --------------------------------------------------------------------------- generate mode

GENERATED = GenerationOutcomeModel(
    summary="Wrote login.feature",
    files=[FileChangeModel(path="src/test/resources/features/ngauto/login.feature", change="added", diff="+@ngauto")],
    generated_cases=["LOGIN-1"],
    skipped_cases=[SkippedCaseModel(case_id="LOGOUT-1", reason="step 1 (click) has no locator in the test data")],
    selector="@ngauto",
)
TEST_DATA = TestDataSetModel(
    source_format=TestDataFormat.JSON,
    cases=[TestCaseSpecModel(id="LOGIN-1", title="Valid login", steps=[TestStepModel(action="open", target="https://x")])],
)


@dataclass
class FakeGenerator:
    outcome: GenerationOutcomeModel
    calls: list[dict[str, Any]] = field(default_factory=list)

    def factory(self, llm_config: Any) -> "FakeGenerator":
        return self

    def generate(self, workspace: Any, project_dir: Path, data_set: TestDataSetModel, fact_sheet: FactSheetModel,
                 findings: Any, command: TestCommandModel, on_event: Any = None, control: Any = None) -> GenerationOutcomeModel:
        self.calls.append({"workspace": workspace, "data_set": data_set, "findings": findings, "command": command})
        on_event(AgentEvent(RunEventType.AGENT_ACTION, RunEventLevel.INFO, "create login.feature", {"tool": "file_editor"}))
        return self.outcome


class FakeTestData:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def get(self, test_data_id: str, project_id: str, org_id: str) -> Any:
        self.calls.append((test_data_id, project_id, org_id))
        return SimpleNamespace(status=TestDataStatus.READY, data_set=TEST_DATA)


class FakeLatestProfile:
    async def get_latest(self, project_id: str, org_id: str) -> Any:
        return SimpleNamespace(findings="the findings")


def _generate_setup(tmp_path: Path, outcome: GenerationOutcomeModel = GENERATED,
                    results: list[TestRunResultModel] | None = None) -> tuple[Setup, FakeGenerator, FakeTestData]:
    setup = _setup(tmp_path, results or [PASSED])
    generator, test_data = FakeGenerator(outcome), FakeTestData()
    setup.nodes.generator_factory = generator.factory  # type: ignore[assignment]
    setup.nodes.test_data_service = test_data  # type: ignore[assignment]
    setup.nodes.profile_service = FakeLatestProfile()  # type: ignore[assignment]
    return setup, generator, test_data


def _generate_state(tmp_path: Path, **extra: Any) -> MasterState:
    return MasterState(**{**_test_state(tmp_path), "mode": RunMode.GENERATE.value, "test_data_id": "data-1", **extra})


def test_generate_mode_writes_tests_before_running_them(tmp_path: Path) -> None:
    assert route_next(_generate_state(tmp_path)) == GENERATE_TESTS
    assert route_next(_generate_state(tmp_path, generation=GENERATED.model_dump(mode="json"))) == TEST_AND_HEAL


@pytest.mark.anyio
async def test_generate_run_with_a_reused_profile_skips_the_analysis(tmp_path: Path) -> None:
    nodes = FakeNodes(tmp_path)

    await build_master_graph(nodes).ainvoke({**_ids(), "mode": RunMode.GENERATE.value, "profile_id": "profile-0"})

    assert nodes.calls == [PREPARE_WORKSPACE, COLLECT_FACTS, GENERATE_TESTS, TEST_AND_HEAL]


def test_interrupted_generation_starts_over_on_a_fresh_copy() -> None:
    run = _run(RunOutputsModel(project_dir="/data/runs/x/project", profile_id=str(ObjectId())), mode=RunMode.GENERATE)

    assert "project_dir" not in initial_state(run)


@pytest.mark.anyio
async def test_generate_tests_saves_the_outcome_and_reports_it(tmp_path: Path) -> None:
    setup, generator, test_data = _generate_setup(tmp_path)
    state = _generate_state(tmp_path)

    update = await setup.nodes.generate_tests(state)

    runs = setup.runs
    assert test_data.calls == [("data-1", state["project_id"], state["org_id"])]
    [call] = generator.calls
    assert call["data_set"] == TEST_DATA and call["findings"] == "the findings"
    assert call["workspace"] is setup.sandbox.workspace
    assert setup.sandbox.writable_org == state["org_id"]
    assert runs.containers == [CONTAINER_ID, None]
    assert runs.outputs["generation"] == GENERATED
    assert update == {"generation": GENERATED.model_dump(mode="json")}
    completed = runs.completed_event(RunStage.GENERATING_TESTS)
    assert completed.message == "Wrote tests for 1 case in 1 file · 1 case left out (no locator)"
    assert completed.data["selector"] == "@ngauto"
    messages = [event.message for event in runs.events]
    assert messages.index("create login.feature") < messages.index(completed.message)


@pytest.mark.anyio
async def test_generation_without_files_fails_the_run(tmp_path: Path) -> None:
    empty = GENERATED.model_copy(update={"files": [], "selector": None})
    setup, _, _ = _generate_setup(tmp_path, empty)

    with pytest.raises(GenerationError) as caught:
        await setup.nodes.generate_tests(_generate_state(tmp_path))

    assert caught.value.message == "The test generator stopped: the generator wrote no test files"
    assert setup.runs.outputs["generation"] == empty


@pytest.mark.anyio
async def test_unknown_selector_is_announced(tmp_path: Path) -> None:
    setup, _, _ = _generate_setup(tmp_path, GENERATED.model_copy(update={"selector": None}))

    await setup.nodes.generate_tests(_generate_state(tmp_path))

    assert setup.runs.events[-1].message == SELECTOR_NOT_FOUND_MESSAGE


@pytest.mark.anyio
async def test_generated_tests_are_the_ones_that_run(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [PASSED])
    state = _generate_state(tmp_path, generation=GENERATED.model_dump(mode="json"), test_selector="ignored")

    await setup.nodes.test_and_heal(state)

    assert setup.runner.calls[0][2] == resolve_test_command(FACT_SHEET, "@ngauto")


# --------------------------------------------------------------------------- the healer's memory


EARLIER_HEAL = HealMemoryModel(
    id="memory-1", org_id="org", project_id="project", run_id="an-earlier-run", attempt=1,
    created_at=datetime(2026, 10, 1, tzinfo=UTC), stack="Java · Maven · TestNG", test_command="mvn -B -ntp test",
    failure_kind="dependency_failure", failure_reason="A dependency could not be resolved", problem="dependency_failure: ...",
    fix_summary="Set TestNG to 7.10.2", changed_files=["pom.xml"], result=HealMemoryResult.FIXED, succeeded=True,
    next_run="the tests passed (2/2)", score=0.81,
)


@pytest.mark.anyio
async def test_the_healer_gets_similar_earlier_heals_and_the_heal_is_remembered(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, PASSED], [FIX], memories=[EARLIER_HEAL])
    state = _test_state(tmp_path)

    await setup.nodes.test_and_heal(state)

    [recall] = setup.memory.recalls
    assert (recall["org_id"], recall["run_id"], recall["result"]) == (state["org_id"], state["run_id"], DEPENDENCY)
    assert setup.healer.calls[0]["memories"] == [EARLIER_HEAL]
    event = next(e for e in setup.runs.events if e.message.startswith("The healer got"))
    assert event.message == "The healer got 1 similar earlier heal as context (1 worked, 0 did not)"
    assert event.data["memories"][0]["id"] == "memory-1"
    # Once the next run showed what the fix did, it is remembered.
    [(point_id, memory)] = setup.memory.remembered
    assert point_id == memory_point_id(state["run_id"], 1)
    assert (memory.org_id, memory.project_id, memory.run_id, memory.attempt) == (
        state["org_id"], state["project_id"], state["run_id"], 1
    )
    assert (memory.result, memory.succeeded, memory.next_run) == ("fixed", True, "the tests passed (2/2)")
    assert memory.fix_summary == FIX.summary and memory.changed_files == ["pom.xml"]


@pytest.mark.anyio
async def test_a_heal_that_did_not_help_is_remembered_too(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, DEPENDENCY, PASSED], [FIX, FIX])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert [(m.attempt, m.result, m.succeeded) for _, m in setup.memory.remembered] == [
        (1, "not_fixed", False), (2, "fixed", True)
    ]


@pytest.mark.anyio
async def test_a_heal_that_ends_the_loop_is_remembered_as_it_ended(tmp_path: Path) -> None:
    blocked = HealOutcomeModel(summary="BLOCKED: no network", blocker="the sandbox cannot reach the internet")
    setup = _setup(tmp_path, [DEPENDENCY], [blocked])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    [(_, memory)] = setup.memory.remembered
    assert (memory.result, memory.succeeded, memory.next_run, memory.blocker) == (
        "blocked", False, None, "the sandbox cannot reach the internet"
    )


@pytest.mark.anyio
async def test_a_heal_that_did_nothing_is_not_remembered(tmp_path: Path) -> None:
    from app.agents.HealerAgent.HealerAgent import NO_SUMMARY

    unreachable = HealOutcomeModel(summary=NO_SUMMARY, error="The healer stopped: the model could not be reached")
    setup = _setup(tmp_path, [DEPENDENCY], [unreachable])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert setup.memory.remembered == []
    assert setup.runs.outputs["test_report"].stop_reason == TestStopReason.HEALER_FAILED


@pytest.mark.anyio
async def test_no_memories_no_event(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, PASSED], [FIX])

    await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert setup.healer.calls[0]["memories"] == []
    assert not any(e.message.startswith("The healer got") for e in setup.runs.events)
