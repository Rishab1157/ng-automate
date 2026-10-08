"""Runs the master agent in the background, a few runs at a time.

The run document records how far a run got, so the executor keeps no state across restarts: on startup it resumes
every unfinished run, and on shutdown it stops the running ones without marking them finished.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.agents.HealerAgent.HealLoopPolicy import describe_test_report
from app.config import settings
from app.core.exceptions import ErrorCode, ErrorMessages, NgAutomateException, NotFoundError
from app.models.runModel import RunEventType, RunModel, RunStatus, TestReportModel

from .MasterState import initial_state
from .RunControl import RunControl

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from app.services.runCommandService import RunCommandService
    from app.services.runService import RunService
    from app.services.sandboxService import SandboxService

logger = logging.getLogger(__name__)

# To move into ErrorCode / ErrorMessages.
RUN_TIMEOUT_ERROR_CODE = "RUN_TIMEOUT"
RUN_TIMEOUT_MESSAGE = "The run took longer than {seconds} seconds and was stopped"

RUN_COMPLETED_MESSAGE = "Run completed: the project profile is ready"
TEST_RUN_COMPLETED_MESSAGE = "Run completed: {summary}"

_FINISHED_STATUSES = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class RunExecutor:
    def __init__(
        self,
        max_concurrent: int = settings.MAX_CONCURRENT_RUNS,
        timeout_seconds: float = settings.RUN_TIMEOUT_SECONDS,
        run_service: RunService | None = None,
        sandbox_service: SandboxService | None = None,
        graph_factory: Callable[[], CompiledStateGraph] | None = None,
        command_service: RunCommandService | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        # Production defaults are created on first use: they pull in LangGraph and the agent SDKs.
        self._run_service = run_service
        self._sandbox_service = sandbox_service
        self._command_service = command_service
        self._graph_factory = graph_factory
        self._graph: CompiledStateGraph | None = None
        # The event loop keeps only weak references to tasks: without these, a running task could be garbage collected.
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._slots: asyncio.Semaphore | None = None
        self._slots_loop: asyncio.AbstractEventLoop | None = None

    @property
    def run_service(self) -> RunService:
        if self._run_service is None:
            from app.services.runService import RunService

            self._run_service = RunService()
        return self._run_service

    @property
    def command_service(self) -> RunCommandService:
        if self._command_service is None:
            from app.services.runCommandService import RunCommandService

            self._command_service = RunCommandService(self.run_service)
        return self._command_service

    @property
    def sandbox_service(self) -> SandboxService:
        if self._sandbox_service is None:
            from app.services.sandboxService import SandboxService

            self._sandbox_service = SandboxService()
        return self._sandbox_service

    async def submit(self, run_id: str) -> None:
        """Start the run in the background. A run that is already executing here is left alone."""
        if self._is_live(run_id):
            logger.info("Run %s is already executing", run_id)
            return
        task = asyncio.create_task(self._execute(run_id), name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda done: self._forget(run_id, done))

    async def resume_interrupted(self) -> int:
        """On startup: restart every queued or running run, after stopping a container a crash left behind."""
        resumed = 0
        for run in await self.run_service.find_unfinished():
            if self._is_live(run.id):
                continue
            if run.sandbox_container_id:
                await self._stop_container(run.id, run.sandbox_container_id)
            await self.submit(run.id)
            resumed += 1
        return resumed

    async def shutdown(self) -> None:
        """Stop the runs executing here. They stay unfinished and resume on the next start."""
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Stopped %d run(s); they resume on the next start", len(tasks))

    async def _execute(self, run_id: str) -> None:
        async with self._concurrency_slots():
            try:
                run = await self.run_service.get_by_id(run_id)
            except NotFoundError:
                logger.warning("Run %s no longer exists", run_id)
                return
            if run.status in _FINISHED_STATUSES:
                return
            try:
                await self._run_graph(run)
            finally:
                await self._release_resources(run_id)

    async def _run_graph(self, run: RunModel) -> None:
        task = asyncio.current_task()
        # The run's whole time budget, from now. Paused time is added back (extend_deadline).
        deadline = asyncio.timeout(self.timeout_seconds)

        def stop(reason: str) -> None:
            if task is not None:
                task.cancel()

        def extend_deadline(seconds: float) -> None:
            when = deadline.when()
            if when is not None and not deadline.expired():
                deadline.reschedule(when + seconds)

        with RunControl(run.id, self.command_service, stop, extend_deadline) as control:
            poller = asyncio.create_task(control.poll(), name=f"run-{run.id}-commands")
            try:
                await self._execute_graph(run, deadline)
            except asyncio.CancelledError:
                if control.stop_reason is None:
                    logger.info("Run %s interrupted; it resumes on the next start", run.id)
                    raise
                # Stopped by the user (or paused too long): the run ends here as cancelled.
                if task is not None:
                    task.uncancel()
                await self._cancel(run, control.stop_reason)
            finally:
                poller.cancel()
                await asyncio.gather(poller, return_exceptions=True)

    async def _execute_graph(self, run: RunModel, deadline: asyncio.Timeout) -> None:
        try:
            await self.run_service.mark_running(run.id)
            graph = self._get_graph()
            state = initial_state(run)
            async with deadline:
                final_state = await graph.ainvoke(state)
            await self.run_service.mark_completed(run.id)
            await self._report_completed(run, final_state)
        except asyncio.CancelledError:
            raise
        except NgAutomateException as error:
            await self._fail(run, error.error_code.value, error.message)
        except Exception as error:
            # Only our own deadline means "too slow"; any other TimeoutError is a bug like any other.
            if isinstance(error, TimeoutError) and deadline is not None and deadline.expired():
                logger.warning("Run %s timed out after %s seconds", run.id, self.timeout_seconds)
                await self._fail(
                    run, RUN_TIMEOUT_ERROR_CODE, RUN_TIMEOUT_MESSAGE.format(seconds=f"{self.timeout_seconds:g}")
                )
            else:
                logger.exception("Run %s failed with an unexpected error", run.id)
                await self._fail(run, ErrorCode.INTERNAL_SERVER_ERROR.value, ErrorMessages.INTERNAL_ERROR)

    async def _report_completed(self, run: RunModel, final_state: dict[str, Any]) -> None:
        """The run-completed event: what the run produced, in one line."""
        data: dict[str, Any] = {"profile_id": final_state.get("profile_id")}
        message = RUN_COMPLETED_MESSAGE
        if final_state.get("test_report"):
            report = TestReportModel.model_validate(final_state["test_report"])
            message = TEST_RUN_COMPLETED_MESSAGE.format(summary=describe_test_report(report))
            data["test_report"] = report.model_dump(mode="json")
        await self.run_service.append_event(run.id, run.org_id, RunEventType.RUN_COMPLETED, message, data=data)

    async def _cancel(self, run: RunModel, reason: str) -> None:
        await self.run_service.mark_cancelled(run.id, reason)
        await self.run_service.append_event(
            run.id, run.org_id, RunEventType.RUN_CANCELLED, reason, data={"code": ErrorCode.RUN_CANCELLED.value}
        )

    async def _fail(self, run: RunModel, code: str, message: str) -> None:
        await self.run_service.mark_failed(run.id, code, message)
        await self.run_service.append_event(run.id, run.org_id, RunEventType.RUN_FAILED, message, data={"code": code})

    async def _release_resources(self, run_id: str) -> None:
        """Stop a container the run still records (this also ends a timed-out analysis thread) and delete the run's
        copy of the project once the run is finished. Never raises: another error may be on its way out."""
        try:
            run = await self.run_service.get_by_id(run_id)
            if run.sandbox_container_id:
                await self._stop_container(run_id, run.sandbox_container_id)
            if run.status in _FINISHED_STATUSES:
                await asyncio.to_thread(self.sandbox_service.remove_run_dir, run_id)
        except Exception:
            logger.exception("Could not clean up after run %s", run_id)

    async def _stop_container(self, run_id: str, container_id: str) -> None:
        try:
            await asyncio.to_thread(self.sandbox_service.stop_container, container_id)
        except Exception:
            # The run must not stay stuck on a container Docker cannot stop; the log keeps the id for manual cleanup.
            logger.exception("Could not stop container %s of run %s", container_id, run_id)
        await self.run_service.set_sandbox_container(run_id, None)

    def _get_graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = (self._graph_factory or self._build_default_graph)()
        return self._graph

    def _build_default_graph(self) -> CompiledStateGraph:
        from .MasterGraph import build_master_graph
        from .MasterNodes import MasterNodes

        return build_master_graph(MasterNodes(run_service=self.run_service, sandbox_service=self.sandbox_service))

    def _concurrency_slots(self) -> asyncio.Semaphore:
        # asyncio primitives belong to one event loop, and tests start several loops in one process.
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(self.max_concurrent)
            self._slots_loop = loop
        return self._slots

    def _is_live(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def _forget(self, run_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            del self._tasks[run_id]
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Run %s stopped with an unexpected error", run_id, exc_info=error)


run_executor = RunExecutor()
