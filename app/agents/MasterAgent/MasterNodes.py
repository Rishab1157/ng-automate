"""The master agent's stages. Each one marks its stage, does the work, saves its output and reports a summary.

A stage saves its output on the run before it returns, so a run interrupted later resumes after it.
Blocking work (unzipping, scanning files, the agents, the tests) runs in worker threads; the event loop stays free.
"""

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Any, TypeVar

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.AnalyzerAgent.AnalyzerAgent import AnalyzerAgent
from app.agents.AnalyzerAgent.EvidenceVerifier import verify_evidence
from app.agents.AnalyzerAgent.FactSheetCollector import collect_fact_sheet
from app.agents.HealerAgent.HealerAgent import NO_SUMMARY, HealerAgent
from app.agents.HealerAgent.HealerPrompts import describe_stack
from app.agents.HealerAgent.HealLoopPolicy import build_test_report, describe_test_report, judge_heal, stop_reason
from app.agents.RunnerAgent.TestCommandResolver import resolve_test_command
from app.agents.RunnerAgent.TestRunner import TestRunner
from app.agents.TestGeneratorAgent.TestGeneratorAgent import TestGeneratorAgent
from app.config import settings
from app.core.exceptions import ErrorMessages, GenerationError, ValidationError
from app.models.analyzerModel import AnalyzerFindingsModel, FactModel, FactSheetModel
from app.models.healerModel import HealOutcomeModel
from app.models.healMemoryModel import SUCCESSFUL_RESULTS, HealMemoryMapper, HealMemoryModel, memory_point_id
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType, RunScope, RunStage, TestAttemptModel, TestStopReason
from app.models.testDataModel import TestDataStatus
from app.models.testGeneratorModel import GenerationOutcomeModel
from app.models.testRunModel import TestCommandModel, TestRunResultModel
from app.services.healMemoryService import HealMemoryService
from app.services.modelConnectionService import ModelConnectionService
from app.services.projectProfileService import ProjectProfileService
from app.services.runService import RunService
from app.services.sandboxService import SandboxService, SandboxSession
from app.services.testDataService import TestDataService
from app.utils.ProjectSnapshot import project_fingerprint

from .MasterState import MasterState
from .RunControl import agent_control

logger = logging.getLogger(__name__)

# How long the analyzer thread waits for the run to record its container before it gives up on the run.
CONTAINER_RECORD_TIMEOUT_SECONDS = 30
# How long a finished analysis waits for its last agent events to be saved before reporting its summary.
EVENT_DRAIN_TIMEOUT_SECONDS = 15
MAX_SUMMARY_VALUE_CHARS = 80

STAGE_STARTED_MESSAGES: dict[RunStage, str] = {
    RunStage.PREPARING_WORKSPACE: "Preparing a private copy of the project",
    RunStage.COLLECTING_FACTS: "Reading the project's files, build setup and dependencies",
    RunStage.ANALYZING: "The analyzer agent is studying the project in a read-only sandbox",
    RunStage.SAVING_PROFILE: "Saving the project profile",
    RunStage.GENERATING_TESTS: "The test generator is writing tests from the test data",
    RunStage.RUNNING_TESTS: "Running the project's tests in a sandbox",
    RunStage.HEALING: "The healer agent is fixing the problem that stopped the tests",
}
WORKSPACE_READY_MESSAGE = "Project files are ready"
NOTHING_FOUND_MESSAGE = "No known language, build tool or test framework found"
ANALYSIS_FINISHED_MESSAGE = "Analysis finished"
PROFILE_SAVED_MESSAGE = "Project profile saved"
EVIDENCE_REMOVED_MESSAGE = "Removed {count} cited file path(s) that do not exist in the project"
MAX_REPORTED_REMOVED_PATHS = 50
HEALER_NO_CHANGES_MESSAGE = "The healer finished without changing any project file"
HEALER_TIMED_OUT_MESSAGE = "Stopped: the healer used all of its {minutes:g} minutes of healing time"
# A heal that ends this close to the healing budget ran out of time.
BUDGET_SLACK_SECONDS = 1.0
MAX_DETAIL_CHARS = 300
MEMORIES_MESSAGE = "The healer got {count} similar earlier {heals} as context ({worked} worked, {failed} did not)"
NO_GENERATED_FILES = "the generator wrote no test files"
SELECTOR_NOT_FOUND_MESSAGE = "Could not tell the generated tests apart: all tests of the project will run"
MAX_REPORTED_FILES = 20

FactCollector = Callable[[Path], FactSheetModel]
AnalyzerFactory = Callable[[LlmConfigModel], AnalyzerAgent]
RunnerFactory = Callable[[], TestRunner]
HealerFactory = Callable[[LlmConfigModel], HealerAgent]
GeneratorFactory = Callable[[LlmConfigModel], TestGeneratorAgent]
T = TypeVar("T")


def _default_runner() -> TestRunner:
    return TestRunner(timeout_seconds=settings.TEST_RUN_TIMEOUT_SECONDS)


class MasterNodes:
    def __init__(
        self,
        run_service: RunService | None = None,
        sandbox_service: SandboxService | None = None,
        model_connection_service: ModelConnectionService | None = None,
        profile_service: ProjectProfileService | None = None,
        test_data_service: TestDataService | None = None,
        fact_collector: FactCollector = collect_fact_sheet,
        analyzer_factory: AnalyzerFactory = AnalyzerAgent,
        runner_factory: RunnerFactory = _default_runner,
        healer_factory: HealerFactory = HealerAgent,
        generator_factory: GeneratorFactory = TestGeneratorAgent,
        heal_budget_seconds: float = settings.HEALER_TIME_BUDGET_SECONDS,
        heal_memory_service: HealMemoryService | None = None,
    ) -> None:
        self.run_service = run_service or RunService()
        self.sandbox_service = sandbox_service or SandboxService()
        self.model_connection_service = model_connection_service or ModelConnectionService()
        self.profile_service = profile_service or ProjectProfileService()
        self.test_data_service = test_data_service or TestDataService()
        self.fact_collector = fact_collector
        self.analyzer_factory = analyzer_factory
        self.runner_factory = runner_factory
        self.healer_factory = healer_factory
        self.generator_factory = generator_factory
        self.heal_budget_seconds = heal_budget_seconds
        self.heal_memory_service = heal_memory_service or HealMemoryService()

    async def prepare_workspace(self, state: MasterState) -> MasterState:
        stage = RunStage.PREPARING_WORKSPACE
        await self._start_stage(state, stage)
        project_dir = str(
            await asyncio.to_thread(self.sandbox_service.prepare_project_dir, state["project_id"], state["run_id"])
        )
        # A fresh copy has none of the agents' edits: generated tests and test attempts of an older copy no longer apply.
        await self.run_service.save_outputs(state["run_id"], project_dir=project_dir, generation=None, test_attempts=None)
        await self._finish_stage(state, stage, WORKSPACE_READY_MESSAGE)
        return {"project_dir": project_dir, "generation": None, "test_attempts": None}

    async def collect_facts(self, state: MasterState) -> MasterState:
        stage = RunStage.COLLECTING_FACTS
        await self._start_stage(state, stage)
        fact_sheet = await asyncio.to_thread(self.fact_collector, Path(state["project_dir"]))
        await self.run_service.save_outputs(state["run_id"], fact_sheet=fact_sheet)
        await self._finish_stage(state, stage, _describe_fact_sheet(fact_sheet), **_fact_sheet_summary(fact_sheet))
        return {"fact_sheet": fact_sheet.model_dump(mode="json")}

    async def analyze(self, state: MasterState) -> MasterState:
        stage = RunStage.ANALYZING
        run_id = state["run_id"]
        await self._start_stage(state, stage)
        llm_config = await self.model_connection_service.get_llm_config(
            state.get("model_connection_id"), state["org_id"]
        )
        fact_sheet = FactSheetModel.model_validate(state["fact_sheet"])
        project_dir = Path(state["project_dir"])
        bridge = _AgentBridge(asyncio.get_running_loop(), self.run_service, run_id, state["org_id"])
        findings = await self._in_sandbox(
            run_id,
            bridge,
            lambda: self.sandbox_service.open_readonly(project_dir, bridge.on_started),
            lambda session: self.analyzer_factory(llm_config).analyze(
                session.workspace, fact_sheet, bridge.on_event, control=agent_control(run_id)
            ),
        )

        findings, removed = await asyncio.to_thread(verify_evidence, findings, project_dir)
        if removed:
            await self.run_service.append_event(
                run_id,
                state["org_id"],
                RunEventType.AGENT_ERROR,
                EVIDENCE_REMOVED_MESSAGE.format(count=len(removed)),
                data={"removed": removed[:MAX_REPORTED_REMOVED_PATHS]},
            )
        await self.run_service.save_outputs(run_id, findings=findings, llm_model=llm_config.model)
        await self._finish_stage(
            state,
            stage,
            _describe_findings(findings),
            llm_model=llm_config.model,
            important_paths=len(findings.important_paths),
            open_questions=len(findings.open_questions),
        )
        return {"findings": findings.model_dump(mode="json"), "llm_model": llm_config.model}

    async def save_profile(self, state: MasterState) -> MasterState:
        stage = RunStage.SAVING_PROFILE
        await self._start_stage(state, stage)
        profile = await self.profile_service.create(
            org_id=state["org_id"],
            project_id=state["project_id"],
            run_id=state["run_id"],
            fact_sheet=FactSheetModel.model_validate(state["fact_sheet"]),
            findings=AnalyzerFindingsModel.model_validate(state["findings"]),
            llm_model=state["llm_model"],
        )
        await self.run_service.save_outputs(state["run_id"], profile_id=profile.id)
        await self._finish_stage(state, stage, PROFILE_SAVED_MESSAGE, profile_id=profile.id)
        return {"profile_id": profile.id}

    async def generate_tests(self, state: MasterState) -> MasterState:
        """Write tests for the run's test data, in the project's own style, in a writable sandbox."""
        stage = RunStage.GENERATING_TESTS
        run_id, org_id = state["run_id"], state["org_id"]
        await self._start_stage(state, stage)
        upload = await self.test_data_service.get(state["test_data_id"], state["project_id"], org_id)
        if upload.status != TestDataStatus.READY or upload.data_set is None:
            raise ValidationError(ErrorMessages.TEST_DATA_NOT_READY.format(status=upload.status.value))
        data_set = upload.data_set
        profile = await self.profile_service.get_latest(state["project_id"], org_id)
        fact_sheet = FactSheetModel.model_validate(state["fact_sheet"])
        # Also stops a stack the sandbox cannot run (UFT) before the generator spends any time on it.
        command = resolve_test_command(fact_sheet)
        llm_config = await self.model_connection_service.get_llm_config(state.get("model_connection_id"), org_id)
        project_dir = Path(state["project_dir"])
        bridge = _AgentBridge(asyncio.get_running_loop(), self.run_service, run_id, org_id)
        outcome = await self._in_sandbox(
            run_id,
            bridge,
            lambda: self.sandbox_service.open_writable(project_dir, org_id, bridge.on_started),
            lambda session: self.generator_factory(llm_config).generate(
                session.workspace, project_dir, data_set, fact_sheet, profile.findings, command, bridge.on_event,
                control=agent_control(run_id),
            ),
        )
        await self.run_service.save_outputs(run_id, generation=outcome)
        if not outcome.files:
            raise GenerationError(ErrorMessages.GENERATION_FAILED.format(reason=outcome.error or NO_GENERATED_FILES))
        await self._finish_stage(state, stage, _describe_generation(outcome), **_generation_summary(outcome))
        if outcome.selector is None:
            await self.run_service.append_event(run_id, org_id, RunEventType.AGENT_ERROR, SELECTOR_NOT_FOUND_MESSAGE)
        return {"generation": outcome.model_dump(mode="json")}

    async def test_and_heal(self, state: MasterState) -> MasterState:
        """Run the tests; while the failure is the healer's job, let the healer fix it and run them again.

        The loop stops when the tests pass, when the failure is for the user, when there is no progress
        (HealLoopPolicy), when the healer proves a blocker, or when the run's healing time budget is used up.
        One writable sandbox serves the whole loop, so what the healer installs is still there for the next run.
        Progress is saved after every run and every fix, for the UI; an interrupted loop starts over (initial_state).
        Before each heal the healer gets the organization's most similar earlier heals (memory); once the next run
        (or the end of the loop) shows what a heal did, it is remembered, successful or not.
        """
        run_id, org_id = state["run_id"], state["org_id"]
        await self._start_stage(state, RunStage.RUNNING_TESTS, attempt=1)
        fact_sheet = FactSheetModel.model_validate(state["fact_sheet"])
        stack = describe_stack(fact_sheet)
        command = resolve_test_command(fact_sheet, _test_selector(state))
        await self.run_service.save_outputs(run_id, test_command=command)
        llm_config = await self.model_connection_service.get_llm_config(state.get("model_connection_id"), org_id)
        project_dir = Path(state["project_dir"])
        runner = self.runner_factory()
        bridge = _AgentBridge(asyncio.get_running_loop(), self.run_service, run_id, org_id)
        attempts: list[TestAttemptModel] = []
        reason: TestStopReason | None = None
        detail: str | None = None
        budget_left = self.heal_budget_seconds
        sandbox = ExitStack()
        cancelled = False
        try:
            session = await asyncio.to_thread(
                sandbox.enter_context, self.sandbox_service.open_writable(project_dir, org_id, bridge.on_started)
            )
            while True:
                number = len(attempts) + 1
                if number > 1:
                    await self._start_stage(state, RunStage.RUNNING_TESTS, attempt=number)
                state_before = await asyncio.to_thread(project_fingerprint, project_dir)
                result = await asyncio.to_thread(runner.run, session.workspace, project_dir, command)
                attempts.append(TestAttemptModel(number=number, result=result, project_state=state_before))
                await self.run_service.save_outputs(run_id, test_attempts=attempts)
                await self._finish_stage(
                    state, RunStage.RUNNING_TESTS, _describe_test_run(number, result), **_test_run_summary(number, result)
                )
                if number > 1:
                    await self._remember_heal(state, attempts[-2], attempts[-1], command, stack)
                reason = stop_reason(attempts)
                if reason is None and budget_left <= BUDGET_SLACK_SECONDS:
                    reason, detail = TestStopReason.TIMED_OUT, self._timed_out_message()
                if reason is not None:
                    break

                await self._start_stage(state, RunStage.HEALING, attempt=number)
                memories = await self.heal_memory_service.recall(org_id=org_id, run_id=run_id, result=result)
                if memories:
                    await self.run_service.append_event(
                        run_id, org_id, RunEventType.AGENT_MESSAGE, _describe_memories(memories),
                        data={"memories": [_memory_summary(memory) for memory in memories]},
                    )
                healer = self.healer_factory(llm_config)
                previous = [attempt.heal for attempt in attempts if attempt.heal is not None]
                time_limit = budget_left
                outcome = await asyncio.to_thread(
                    lambda: healer.heal(
                        session.workspace, project_dir, result, command, fact_sheet, bridge.on_event,
                        previous, control=agent_control(run_id), time_limit=time_limit, memories=memories,
                    )
                )
                budget_left -= outcome.seconds
                await bridge.drain(EVENT_DRAIN_TIMEOUT_SECONDS)
                attempts[-1] = attempts[-1].model_copy(update={"heal": outcome})
                await self.run_service.save_outputs(run_id, test_attempts=attempts)
                await self._finish_stage(state, RunStage.HEALING, _describe_heal(outcome), **_heal_summary(number, outcome))
                if outcome.blocker:
                    reason, detail = TestStopReason.BLOCKED, _first_line(outcome.blocker)
                    break
                if outcome.error and budget_left <= BUDGET_SLACK_SECONDS:
                    reason, detail = TestStopReason.TIMED_OUT, self._timed_out_message()
                    break
                if outcome.error:
                    reason, detail = TestStopReason.HEALER_FAILED, outcome.error
                    break
        except asyncio.CancelledError:
            # Timeout or shutdown: as in analyze(), the run executor stops the recorded container.
            cancelled = True
            raise
        finally:
            bridge.close()
            if not cancelled:
                await bridge.drain(EVENT_DRAIN_TIMEOUT_SECONDS)
                await asyncio.to_thread(sandbox.close)
                await self.run_service.set_sandbox_container(run_id, None)

        if attempts and attempts[-1].heal is not None:
            # The loop stopped right after a heal (blocked, out of time, healer failed): remember it as it ended.
            await self._remember_heal(state, attempts[-1], None, command, stack)
        report = build_test_report(attempts, reason, detail)
        await self.run_service.save_outputs(run_id, test_report=report)
        await self.run_service.append_event(
            run_id,
            org_id,
            RunEventType.STAGE_COMPLETED,
            describe_test_report(report),
            data={"stage": "tests", **report.model_dump(mode="json")},
        )
        return {
            "test_command": command.model_dump(mode="json"),
            "test_attempts": [attempt.model_dump(mode="json") for attempt in attempts],
            "test_report": report.model_dump(mode="json"),
        }

    async def _remember_heal(
        self,
        state: MasterState,
        attempt: TestAttemptModel,
        next_attempt: TestAttemptModel | None,
        command: TestCommandModel,
        stack: str,
    ) -> None:
        """Keep what `attempt`'s heal tried and what came of it (`next_attempt`: the run after it, if any)."""
        heal = attempt.heal
        if heal is None or not _worth_remembering(heal):
            return
        memory = HealMemoryMapper.to_db_model(
            org_id=state["org_id"],
            project_id=state["project_id"],
            run_id=state["run_id"],
            attempt=attempt,
            next_attempt=next_attempt,
            result=judge_heal(attempt, next_attempt),
            stack=stack,
            test_command=command.command,
            embedding_model=settings.EMBEDDING_MODEL,
        )
        await self.heal_memory_service.remember(memory_point_id(state["run_id"], attempt.number), memory)

    def _timed_out_message(self) -> str:
        return HEALER_TIMED_OUT_MESSAGE.format(minutes=round(self.heal_budget_seconds / 60, 1))

    async def _in_sandbox(
        self,
        run_id: str,
        bridge: "_AgentBridge",
        open_session: Callable[[], AbstractContextManager[SandboxSession]],
        work: Callable[[SandboxSession], T],
    ) -> T:
        """Run blocking agent work in a worker thread, inside a sandbox that is always stopped afterwards."""

        def run() -> T:
            with open_session() as session:
                return work(session)

        cancelled = False
        try:
            return await asyncio.to_thread(run)
        except asyncio.CancelledError:
            # Timeout or shutdown. A thread cannot be interrupted, so the container stays recorded: the run executor
            # stops it, which also ends the thread.
            cancelled = True
            raise
        finally:
            bridge.close()
            if not cancelled:
                await bridge.drain(EVENT_DRAIN_TIMEOUT_SECONDS)
                await self.run_service.set_sandbox_container(run_id, None)

    async def _start_stage(self, state: MasterState, stage: RunStage, **data: Any) -> None:
        await self.run_service.set_stage(state["run_id"], stage)
        await self.run_service.append_event(
            state["run_id"],
            state["org_id"],
            RunEventType.STAGE_STARTED,
            STAGE_STARTED_MESSAGES[stage],
            data={"stage": stage.value, **data},
        )

    async def _finish_stage(self, state: MasterState, stage: RunStage, message: str, **summary: Any) -> None:
        await self.run_service.append_event(
            state["run_id"],
            state["org_id"],
            RunEventType.STAGE_COMPLETED,
            message,
            data={"stage": stage.value, **summary},
        )


class _RunStopped(Exception):
    """The run stopped (timeout or shutdown) while its sandbox was starting: abandon the stage."""


class _AgentBridge:
    """Connects an agent's threads to the event loop, which owns the database client.

    Agent events are saved one at a time, in the order they happen. Once closed, events of an abandoned stage are
    dropped, and a sandbox that comes up after the run stopped is refused, so the sandbox cleans it up at once.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, run_service: RunService, run_id: str, org_id: str) -> None:
        self._loop = loop
        self._run_service = run_service
        self._run_id = run_id
        self._org_id = org_id
        self._in_order = asyncio.Lock()
        self._closed = threading.Event()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._pending_lock = threading.Lock()

    def on_started(self, container_id: str) -> None:
        """Agent thread: record the container before the agent starts, so a restarted server can stop it."""
        if self._closed.is_set():
            raise _RunStopped()
        future = self._schedule(self._run_service.set_sandbox_container(self._run_id, container_id))
        if future is None:
            raise _RunStopped()
        try:
            future.result(timeout=CONTAINER_RECORD_TIMEOUT_SECONDS)
        except BaseException:
            future.cancel()
            raise
        if self._closed.is_set():
            raise _RunStopped()

    def on_event(self, event: AgentEvent) -> None:
        """Agent threads: hand the event to the event loop and return at once (fire-and-forget)."""
        if self._closed.is_set():
            return
        future = self._schedule(self._append(event))
        if future is None:
            return
        with self._pending_lock:
            self._pending.add(future)
        future.add_done_callback(self._forget)

    def close(self) -> None:
        self._closed.set()

    async def drain(self, timeout: float) -> None:
        """Wait (bounded) for the events already handed over, so they come before the stage's summary."""
        with self._pending_lock:
            pending = [asyncio.wrap_future(future) for future in self._pending]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def _append(self, event: AgentEvent) -> None:
        # Never raises: a lost timeline entry must not break the agent, and nobody awaits the result.
        try:
            async with self._in_order:
                await self._run_service.append_event(
                    self._run_id, self._org_id, event.type, event.message, level=event.level, data=event.data
                )
        except Exception as error:
            logger.warning("Could not save an agent event of run %s: %s", self._run_id, error)

    def _schedule(self, coroutine: Coroutine[Any, Any, Any]) -> concurrent.futures.Future[Any] | None:
        try:
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except RuntimeError:
            # The event loop is closed: the server has stopped.
            coroutine.close()
            return None

    def _forget(self, future: concurrent.futures.Future[Any]) -> None:
        with self._pending_lock:
            self._pending.discard(future)


def _describe_fact_sheet(fact_sheet: FactSheetModel) -> str:
    """One timeline line, e.g. "Found Java · Maven · TestNG, Selenium, Cucumber"."""
    tools = [
        *_values(fact_sheet.test_frameworks),
        *_values(fact_sheet.automation_tools),
        *_values(fact_sheet.bdd_tool),
    ]
    parts = [", ".join(_values(fact_sheet.primary_language)), ", ".join(_values(fact_sheet.build_tool)), ", ".join(tools)]
    found = [part for part in parts if part]
    return "Found " + " · ".join(found) if found else NOTHING_FOUND_MESSAGE


def _fact_sheet_summary(fact_sheet: FactSheetModel) -> dict[str, Any]:
    return {
        "total_files": fact_sheet.total_files,
        "primary_language": fact_sheet.primary_language.value,
        "build_tool": fact_sheet.build_tool.value,
        "test_frameworks": fact_sheet.test_frameworks.value,
        "automation_tools": fact_sheet.automation_tools.value,
        "bdd_tool": fact_sheet.bdd_tool.value,
    }


def _describe_findings(findings: AnalyzerFindingsModel) -> str:
    """One timeline line, e.g. "Analysis finished · Page Object Model · 6 important paths · 2 open questions"."""
    parts = [ANALYSIS_FINISHED_MESSAGE]
    pattern = ", ".join(_values(findings.architecture_pattern))
    if pattern:
        parts.append(_shorten(pattern, MAX_SUMMARY_VALUE_CHARS))
    parts.append(_count(len(findings.important_paths), "important path"))
    if findings.open_questions:
        parts.append(_count(len(findings.open_questions), "open question"))
    return " · ".join(parts)


def _test_selector(state: MasterState) -> str | None:
    """Generate mode: the generated tests only, unless the user asked for the whole suite. Test mode: the user's."""
    generation = state.get("generation")
    if not generation:
        return state.get("test_selector")
    if state.get("run_scope") == RunScope.ALL.value:
        return None
    return generation.get("selector")


def _describe_generation(outcome: GenerationOutcomeModel) -> str:
    """One timeline line, e.g. "Wrote tests for 3 cases in 2 files · 1 case left out (no locator)"."""
    message = (
        f"Wrote tests for {_count(len(outcome.generated_cases), 'case')} in {_count(len(outcome.files), 'file')}"
    )
    if outcome.skipped_cases:
        message += f" · {_count(len(outcome.skipped_cases), 'case')} left out (no locator)"
    if outcome.error:
        message += f" · {outcome.error}"
    return message


def _generation_summary(outcome: GenerationOutcomeModel) -> dict[str, Any]:
    return {
        "summary": outcome.summary,
        "files": [{"path": change.path, "change": change.change} for change in outcome.files[:MAX_REPORTED_FILES]],
        "generated_cases": outcome.generated_cases,
        "skipped_cases": [case.model_dump() for case in outcome.skipped_cases],
        "selector": outcome.selector,
        "reverted_files": outcome.reverted_files[:MAX_REPORTED_FILES],
        "stopped": outcome.error is not None,
    }


def _describe_test_run(number: int, result: TestRunResultModel) -> str:
    """One timeline line, e.g. "Run 1: 5 passed, 2 failed · Some tests failed on their checks"."""
    if result.total:
        counts = [f"{result.passed} passed"]
        if result.failed or result.errors:
            counts.append(f"{result.failed + result.errors} failed")
        if result.skipped:
            counts.append(f"{result.skipped} skipped")
        found = ", ".join(counts)
    else:
        found = "no test results"
    return f"Run {number}: {found} · {result.classification.reason}"


def _test_run_summary(number: int, result: TestRunResultModel) -> dict[str, Any]:
    return {
        "attempt": number,
        "kind": result.classification.kind.value,
        "healable": result.classification.healable,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "skipped": result.skipped,
    }


def _describe_heal(outcome: HealOutcomeModel) -> str:
    """One timeline line, e.g. "The healer changed 1 file: pom.xml"."""
    if outcome.blocker:
        message = f"The healer is blocked: {_first_line(outcome.blocker)}"
    elif outcome.error:
        message = outcome.error
        if outcome.changes:
            message += f" (kept {_count(len(outcome.changes), 'changed file')})"
    elif not outcome.changes:
        message = HEALER_NO_CHANGES_MESSAGE
    else:
        paths = [change.path for change in outcome.changes]
        shown = ", ".join(paths[:3]) + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")
        message = f"The healer changed {_count(len(paths), 'file')}: {shown}"
    if outcome.reverted_files:
        message += f" ({_count(len(outcome.reverted_files), 'forbidden edit')} rolled back)"
    return message


def _heal_summary(number: int, outcome: HealOutcomeModel) -> dict[str, Any]:
    return {
        "attempt": number,
        "summary": outcome.summary,
        "changes": [{"path": change.path, "change": change.change} for change in outcome.changes[:MAX_REPORTED_FILES]],
        "reverted_files": outcome.reverted_files[:MAX_REPORTED_FILES],
        "violations": len(outcome.violations),
        "stopped": outcome.error is not None,
        "blocked": outcome.blocker is not None,
        "seconds": round(outcome.seconds, 1),
    }


def _worth_remembering(heal: HealOutcomeModel) -> bool:
    """A heal that did nothing at all (e.g. the model could not be reached) teaches nothing."""
    return bool(heal.changes or heal.reverted_files or heal.blocker) or heal.summary != NO_SUMMARY


def _describe_memories(memories: list[HealMemoryModel]) -> str:
    worked = sum(1 for memory in memories if memory.result in SUCCESSFUL_RESULTS)
    return MEMORIES_MESSAGE.format(
        count=len(memories), heals="heal" if len(memories) == 1 else "heals", worked=worked, failed=len(memories) - worked
    )


def _memory_summary(memory: HealMemoryModel) -> dict[str, Any]:
    return {
        "id": memory.id,
        "run_id": memory.run_id,
        "score": round(memory.score, 3) if memory.score is not None else None,
        "result": memory.result.value,
        "problem": memory.failure_evidence[0] if memory.failure_evidence else memory.failure_kind,
    }


def _first_line(text: str) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return line if len(line) <= MAX_DETAIL_CHARS else line[: MAX_DETAIL_CHARS - 1] + "…"


def _values(fact: FactModel) -> list[str]:
    if fact.value is None:
        return []
    return [fact.value] if isinstance(fact.value, str) else list(fact.value)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
