"""The master agent's stages. Each one marks its stage, does the work, saves its output and reports a summary.

A stage saves its output on the run before it returns, so a run interrupted later resumes after it.
Blocking work (unzipping, scanning files, the analyzer agent) runs in worker threads; the event loop stays free.
"""

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.AnalyzerAgent.AnalyzerAgent import AnalyzerAgent
from app.agents.AnalyzerAgent.FactSheetCollector import collect_fact_sheet
from app.models.analyzerModel import AnalyzerFindingsModel, FactModel, FactSheetModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType, RunStage
from app.services.modelConnectionService import ModelConnectionService
from app.services.projectProfileService import ProjectProfileService
from app.services.runService import RunService
from app.services.sandboxService import SandboxService

from .MasterState import MasterState

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
}
WORKSPACE_READY_MESSAGE = "Project files are ready"
NOTHING_FOUND_MESSAGE = "No known language, build tool or test framework found"
ANALYSIS_FINISHED_MESSAGE = "Analysis finished"
PROFILE_SAVED_MESSAGE = "Project profile saved"

FactCollector = Callable[[Path], FactSheetModel]
AnalyzerFactory = Callable[[LlmConfigModel], AnalyzerAgent]


class MasterNodes:
    def __init__(
        self,
        run_service: RunService | None = None,
        sandbox_service: SandboxService | None = None,
        model_connection_service: ModelConnectionService | None = None,
        profile_service: ProjectProfileService | None = None,
        fact_collector: FactCollector = collect_fact_sheet,
        analyzer_factory: AnalyzerFactory = AnalyzerAgent,
    ) -> None:
        self.run_service = run_service or RunService()
        self.sandbox_service = sandbox_service or SandboxService()
        self.model_connection_service = model_connection_service or ModelConnectionService()
        self.profile_service = profile_service or ProjectProfileService()
        self.fact_collector = fact_collector
        self.analyzer_factory = analyzer_factory

    async def prepare_workspace(self, state: MasterState) -> MasterState:
        stage = RunStage.PREPARING_WORKSPACE
        await self._start_stage(state, stage)
        project_dir = str(
            await asyncio.to_thread(self.sandbox_service.prepare_project_dir, state["project_id"], state["run_id"])
        )
        await self.run_service.save_outputs(state["run_id"], project_dir=project_dir)
        await self._finish_stage(state, stage, WORKSPACE_READY_MESSAGE)
        return {"project_dir": project_dir}

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
        bridge = _AnalyzerBridge(asyncio.get_running_loop(), self.run_service, run_id, state["org_id"])

        def run_analyzer() -> AnalyzerFindingsModel:
            with self.sandbox_service.open_readonly(project_dir, bridge.on_started) as session:
                return self.analyzer_factory(llm_config).analyze(session.workspace, fact_sheet, bridge.on_event)

        cancelled = False
        try:
            findings = await asyncio.to_thread(run_analyzer)
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

    async def _start_stage(self, state: MasterState, stage: RunStage) -> None:
        await self.run_service.set_stage(state["run_id"], stage)
        await self.run_service.append_event(
            state["run_id"],
            state["org_id"],
            RunEventType.STAGE_STARTED,
            STAGE_STARTED_MESSAGES[stage],
            data={"stage": stage.value},
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
    """The run stopped (timeout or shutdown) while its sandbox was starting: abandon the analysis."""


class _AnalyzerBridge:
    """Connects the analyzer's threads to the event loop, which owns the database client.

    Agent events are saved one at a time, in the order they happen. Once closed, events of an abandoned analysis are
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
        """Analyzer thread: record the container before the analysis starts, so a restarted server can stop it."""
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
        """Analyzer threads: hand the event to the event loop and return at once (fire-and-forget)."""
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
        # Never raises: a lost timeline entry must not break the analysis, and nobody awaits the result.
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


def _values(fact: FactModel) -> list[str]:
    if fact.value is None:
        return []
    return [fact.value] if isinstance(fact.value, str) else list(fact.value)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
