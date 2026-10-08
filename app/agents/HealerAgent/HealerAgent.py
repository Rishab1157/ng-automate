"""The healer: fixes technical failures (build, dependencies, environment) so the tests can run.

An OpenHands agent works in the writable test sandbox with a terminal and decides itself how to fix the failure
(edit code or configuration, install dependencies, download tools or drivers). Every change is diffed against a
snapshot taken before; a change that breaks the guards (a test deleted, skipped or weakened; a build file that no
longer parses; a backup copy left behind) is rolled back file by file, also when the agent stops early: no edit is
ever left unchecked.

When the environment stops it for good, the agent says so ("BLOCKED: ...") and quotes the error. The claim is
accepted only when that error really appeared in the output of a command it ran (and is not just the test failure
it was asked to fix): a model cannot end the loop by simply saying it is stuck.
"""

import logging
import re
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.event import ObservationEvent
from openhands.sdk.llm import content_to_str
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool

from app.agents.AnalyzerAgent.AgentEventMapper import MAX_MESSAGE_CHARS, AgentEvent, clip
from app.agents.HealerAgent.HealerGuards import check_changes
from app.agents.HealerAgent.HealerPrompts import BLOCKED_MARKER, build_heal_prompt
from app.config import settings
from app.core.exceptions import ErrorMessages, HealError
from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import HealOutcomeModel
from app.models.healMemoryModel import HealMemoryModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventLevel, RunEventType
from app.models.testRunModel import TestCommandModel, TestRunResultModel
from app.utils.AgentConversationUtils import (
    AgentControl,
    EventForwarder,
    ask,
    build_condenser,
    close_conversation,
    secret_values,
)
from app.utils.LlmInstance import build_llm
from app.utils.ProjectSnapshot import ProjectSnapshot

logger = logging.getLogger(__name__)

HEALER_TOOL_NAMES = (FileEditorTool.name, TerminalTool.name, GlobTool.name, GrepTool.name)
MAX_SUMMARY_CHARS = 2000
MAX_BLOCKER_CHARS = 1000
NO_SUMMARY = "The healer finished without a summary."
REVERTED_NOTICE = "Rolled back {path}: the change broke the healer's rules ({rules})"
BLOCKER_REJECTED_NOTICE = (
    "The healer said it is blocked, but the error it quoted is not in the output of any command it ran: it goes on"
)
# A small model sometimes ends its turn with a plan and no edit: it is told once to go on.
NUDGE_MESSAGE = (
    "You have not changed any project file. If the failure is not fixed yet, fix it now (edit files with the "
    "file_editor tool, or install what is missing), check it with the build or test command, then reply with your "
    "summary. If it is already fixed, just reply with your summary."
)
# A git repository the agent created (to diff or stash its edits) is not part of the project.
GIT_DIR = ".git"
# A quoted error line shorter than this proves nothing (it would match almost any output).
MIN_EVIDENCE_CHARS = 20
# Terminal output kept per heal to check a blocker's evidence (the newest is kept).
MAX_RECORDED_OUTPUT_CHARS = 500_000
_BLOCKED_START = re.compile(rf"[\s*#>_`]*{BLOCKED_MARKER[:-1]}[\s*_`]*:", re.IGNORECASE)
_BACKTICKED = re.compile(r"`([^`]+)`")
_SPACES = re.compile(r"\s+")


class HealerAgent:
    def __init__(
        self,
        llm_config: LlmConfigModel,
        max_iterations: int = settings.HEALER_MAX_ITERATIONS,
        timeout_seconds: float = settings.HEALER_TIME_BUDGET_SECONDS,
    ) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    def heal(
        self,
        workspace: BaseWorkspace,
        project_dir: Path,
        result: TestRunResultModel,
        command: TestCommandModel,
        fact_sheet: FactSheetModel,
        on_event: Callable[[AgentEvent], None] | None = None,
        previous: Sequence[HealOutcomeModel] = (),
        control: AgentControl | None = None,
        time_limit: float | None = None,
        memories: Sequence[HealMemoryModel] = (),
    ) -> HealOutcomeModel:
        """Blocking: call from a worker thread. `project_dir` is the host copy mounted in the sandbox.

        `previous`: this run's earlier fixes, so the healer does not repeat one that did not work. `memories`: similar
        heals from earlier runs of the organization (context; the agent decides). `time_limit`: what is left of the
        run's healing budget (default: the healer's own timeout); paused time does not count. Never raises HealError:
        when the agent stops early, the outcome says why in `error`.
        """
        started = time.monotonic()
        paused_before = control.paused_seconds if control is not None else 0.0
        snapshot = ProjectSnapshot.capture(project_dir)
        forwarder = EventForwarder(on_event, secret_values(self.llm_config))
        outputs = CommandOutputLog()
        had_git_dir = (snapshot.root / GIT_DIR).exists()
        conversation = Conversation(
            agent=self._build_agent(),
            workspace=workspace,
            callbacks=[forwarder, outputs],
            max_iteration_per_run=self.max_iterations,
            visualizer=None,
            # No extra LLM call to name the conversation: the shared local model is slow.
            autotitle=False,
        )
        prompt = build_heal_prompt(result, command, fact_sheet, previous, memories)
        deadline = started + (self.timeout_seconds if time_limit is None else time_limit)
        error: str | None = None
        answer = ""
        try:
            answer = ask(conversation, prompt, deadline, _heal_failed, control)
            stopped = control is not None and control.is_stopped
            if not _claims_blocked(answer) and not snapshot.changes() and not stopped:
                answer = ask(conversation, NUDGE_MESSAGE, deadline, _heal_failed, control) or answer
        except HealError as failure:
            error = failure.message
        finally:
            close_conversation(conversation)
            if not had_git_dir:
                shutil.rmtree(snapshot.root / GIT_DIR, ignore_errors=True)

        answer = answer.strip()
        blocker = None
        rejected = False
        if _claims_blocked(answer):
            if _proven(answer, outputs.text(), result.output_tail):
                blocker = _BLOCKED_START.sub("", answer, count=1).strip()[:MAX_BLOCKER_CHARS]
            else:
                rejected = True
                forwarder.notify(AgentEvent(RunEventType.AGENT_ERROR, RunEventLevel.INFO, BLOCKER_REJECTED_NOTICE))
        paused = (control.paused_seconds - paused_before) if control is not None else 0.0
        outcome = self._outcome(snapshot, forwarder, answer or NO_SUMMARY, error)
        return outcome.model_copy(update={
            "blocker": blocker,
            "blocker_rejected": rejected,
            "seconds": max(time.monotonic() - started - paused, 0.0),
        })

    def _outcome(
        self, snapshot: ProjectSnapshot, forwarder: EventForwarder, summary: str, error: str | None
    ) -> HealOutcomeModel:
        """What changed since the snapshot, with every change that breaks a guard rolled back."""
        changes = snapshot.changes()
        violations = check_changes(snapshot, changes)
        reverted = sorted({violation.file for violation in violations})
        for path in reverted:
            snapshot.restore(path)
            rules = ", ".join(sorted({v.rule for v in violations if v.file == path}))
            forwarder.notify(AgentEvent(
                RunEventType.AGENT_ERROR,
                RunEventLevel.INFO,
                clip(REVERTED_NOTICE.format(path=path, rules=rules), MAX_MESSAGE_CHARS),
                {"file": path, "violations": [v.model_dump() for v in violations if v.file == path]},
            ))
        kept = [change for change in changes if change.path not in reverted]
        logger.info("Healer changed %d file(s), %d rolled back%s", len(changes), len(reverted), " (stopped early)" if error else "")
        return HealOutcomeModel(
            summary=summary[:MAX_SUMMARY_CHARS],
            changes=kept,
            violations=violations,
            reverted_files=reverted,
            error=error,
        )

    def _build_agent(self) -> Agent:
        llm = build_llm(self.llm_config)
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in HEALER_TOOL_NAMES],
            condenser=build_condenser(llm, self.llm_config.num_ctx),
        )


class CommandOutputLog:
    """Conversation callback: keeps the terminal output the agent saw, to check a blocker's evidence. Never raises."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._size = 0

    def __call__(self, event: Any) -> None:
        try:
            if isinstance(event, ObservationEvent) and event.tool_name == TerminalTool.name:
                text = "".join(content_to_str(event.observation.to_llm_content))
                self._parts.append(text)
                self._size += len(text)
                while self._size > MAX_RECORDED_OUTPUT_CHARS and len(self._parts) > 1:
                    self._size -= len(self._parts.pop(0))
        except Exception:
            logger.debug("Could not record a command output", exc_info=True)

    def text(self) -> str:
        return "\n".join(self._parts)


def _claims_blocked(answer: str) -> bool:
    return _BLOCKED_START.match(answer) is not None


def _proven(answer: str, command_output: str, failure_output: str) -> bool:
    """A quoted line of the answer appeared in a command's output, and is new: not the test failure itself."""
    seen, failure = _flat(command_output), _flat(failure_output)
    for line in _BLOCKED_START.sub("", answer, count=1).splitlines():
        for candidate in [line, *_BACKTICKED.findall(line)]:
            quoted = _flat(candidate.strip().strip("`>*-\"' "))
            if len(quoted) >= MIN_EVIDENCE_CHARS and quoted in seen and quoted not in failure:
                return True
    return False


def _flat(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def _heal_failed(reason: str) -> HealError:
    return HealError(ErrorMessages.HEAL_FAILED.format(reason=reason))
