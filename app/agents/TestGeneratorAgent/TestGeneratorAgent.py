"""The test generator: an OpenHands agent that turns the user's test cases into tests in the project's own style.

It works in the writable test sandbox, a few cases at a time in one conversation. A locator the test data does not
give is looked up in the project or on the live page, or assumed and marked for review (never a reason to leave a case
out). Its edits go through the same guards as the healer's (existing tests may not be weakened, build files must still
parse, no backup copies) and it may not delete files. It only writes the tests: a test run runs them.
"""

import logging
import shutil
import time
from collections.abc import Callable
from typing import Any
from pathlib import Path

from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool

from app.agents.AnalyzerAgent.AgentEventMapper import MAX_MESSAGE_CHARS, AgentEvent, clip
from app.agents.HealerAgent.HealerGuards import check_changes
from app.services.sandboxService.SandboxService import SANDBOX_PROJECT_DIR
from app.agents.TestGeneratorAgent.GeneratedTestPlacement import (
    GENERATED_DIR,
    placement_rule,
    runnable_cases,
    unverified_locators,
)
from app.agents.TestGeneratorAgent.ProjectTestBriefing import project_test_briefing
from app.agents.TestGeneratorAgent.TestGeneratorPrompts import build_generate_prompt, build_next_batch_prompt
from app.config import settings
from app.core.exceptions import ErrorMessages, GenerationError, ValidationError
from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventLevel, RunEventType
from app.models.testDataModel import TestDataSetModel
from app.models.testGeneratorModel import GenerationOutcomeModel, SkippedCaseModel
from app.models.testRunModel import TestCommandModel
from app.utils.AgentConversationUtils import (
    AgentControl,
    EventForwarder,
    ask,
    build_condenser,
    close_conversation,
    secret_values,
)
from app.utils.LlmInstance import build_llm
from app.utils.ProjectSnapshot import ProjectSnapshot, project_fingerprint

logger = logging.getLogger(__name__)

GENERATOR_TOOL_NAMES = (FileEditorTool.name, TerminalTool.name, GlobTool.name, GrepTool.name)
MAX_SUMMARY_CHARS = 2000
NO_SUMMARY = "The test generator finished without a summary."
REVERTED_NOTICE = "Rolled back {path}: the change broke the generator's rules ({rules})"
# A small model often replies with a plan instead of a tool call, which ends its turn before it wrote anything.
NUDGE_NOTICE = "The generator stopped without writing the tests for these cases: asking it to continue ({attempt} of {total})"
NUDGE_MESSAGE = (
    "You have not written the tests for these cases yet. Continue now: create or extend the test files with the "
    "file_editor tool, following the rules above. Do not stop to explain or plan; reply only when the files are written."
)
BATCH_NOTICE = "Writing tests for cases {first}-{last} of {total}"
CHECK_NOTICE = "Checking that the tests build: {command}"
BUILDS_NOTICE = "The tests build"
BUILD_ERRORS_NOTICE = "The tests do not build: asking the generator to fix the errors ({attempt} of {total})"
STILL_BROKEN_NOTICE = "The tests still do not build after {total} fixes: the test run's healer gets the errors"
NOT_CHECKED_NOTICE = (
    "The project does not build before any test is added, so the generated tests cannot be checked by building them"
)
FIX_MESSAGE = """The project does not build with the tests you wrote. Fix these errors in the files you wrote (never in the
project's own files), then reply with a short summary:
```
{errors}
```
Usual causes: a missing import; a method, field or constructor that does not exist (use only what the base class, page
objects and helpers really have: view them); a wrong type; or text that is not code, such as the line numbers that
`view` shows, copied into a file. A file of yours that is badly broken is easiest to rewrite: `rm` it in the terminal,
then `create` it again with its whole corrected content."""
MAX_BUILD_ERROR_LINES = 40
MAX_BUILD_ERROR_CHARS = 4000
NOT_REACHED = "the test generator stopped before it finished this case"
GIT_DIR = ".git"
# One install or build of the tests (a first Maven build downloads the project's dependencies).
BUILD_CHECK_TIMEOUT_SECONDS = 900


class TestGeneratorAgent:
    __test__ = False  # not a pytest test class

    def __init__(
        self,
        llm_config: LlmConfigModel,
        max_iterations: int = settings.GENERATOR_MAX_ITERATIONS,
        timeout_seconds: float = settings.GENERATOR_TIMEOUT_SECONDS,
        nudges: int = settings.GENERATOR_NUDGES,
        batch_size: int = settings.GENERATOR_BATCH_SIZE,
        fix_rounds: int = settings.GENERATOR_FIX_ROUNDS,
    ) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.nudges = nudges
        self.batch_size = max(1, batch_size)
        self.fix_rounds = max(0, fix_rounds)

    def generate(
        self,
        workspace: BaseWorkspace,
        project_dir: Path,
        data_set: TestDataSetModel,
        fact_sheet: FactSheetModel,
        findings: AnalyzerFindingsModel | None,
        command: TestCommandModel,
        on_event: Callable[[AgentEvent], None] | None = None,
        control: AgentControl | None = None,
    ) -> GenerationOutcomeModel:
        """Blocking: call from a worker thread. `project_dir` is the host copy mounted in the sandbox.

        Raises ValidationError when no case can be generated. Never raises GenerationError: when the agent stops
        early, the outcome says why in `error` (what it wrote so far is still checked and kept).
        """
        cases, skipped = runnable_cases(data_set.cases)
        if not cases:
            raise ValidationError(ErrorMessages.NO_RUNNABLE_TEST_CASES)

        snapshot = ProjectSnapshot.capture(project_dir)
        had_git_dir = (snapshot.root / GIT_DIR).exists()
        briefing = project_test_briefing(snapshot.root, fact_sheet.test_dirs, GENERATED_DIR)
        forwarder = EventForwarder(on_event, secret_values(self.llm_config))
        conversation = Conversation(
            agent=self._build_agent(),
            workspace=workspace,
            callbacks=[forwarder],
            max_iteration_per_run=self.max_iterations,
            visualizer=None,
            # No extra LLM call to name the conversation: the shared local model is slow.
            autotitle=False,
        )
        batches = [cases[start:start + self.batch_size] for start in range(0, len(cases), self.batch_size)]
        error: str | None = None
        answers: list[str] = []
        done = 0  # cases whose batch the agent finished
        builds: bool | None = None
        try:
            # Only a project that builds before anything is added can show whether the generated tests build.
            checked = self.fix_rounds > 0 and _builds_before(workspace, command, forwarder)
            for batch in batches:
                if control is not None and control.is_stopped:
                    break
                if len(batches) > 1:
                    forwarder.notify(AgentEvent(
                        RunEventType.AGENT_MESSAGE,
                        RunEventLevel.INFO,
                        BATCH_NOTICE.format(first=done + 1, last=done + len(batch), total=len(cases)),
                        {"cases": [case.id for case in batch], "first": done + 1, "total": len(cases)},
                    ))
                prompt = (
                    build_generate_prompt(
                        batch, fact_sheet, findings, command, placement_rule(fact_sheet), len(cases), briefing
                    )
                    if done == 0
                    else build_next_batch_prompt(batch, done + 1, len(cases))
                )
                answers.append(self._write_batch(conversation, prompt, snapshot.root, forwarder, control))
                if checked:
                    builds = self._fix_build(conversation, workspace, command, forwarder, control)
                done += len(batch)
        except GenerationError as failure:
            error = failure.message
        finally:
            close_conversation(conversation)
            if not had_git_dir:
                shutil.rmtree(snapshot.root / GIT_DIR, ignore_errors=True)

        changes = snapshot.changes()
        violations = check_changes(snapshot, changes, allow_deletes=False, additions_only=True)
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
        logger.info("Generator wrote %d file(s) for %d of %d case(s), %d rolled back", len(kept), done, len(cases), len(reverted))
        skipped += [(case.id, NOT_REACHED) for case in cases[done:]]
        summary = "\n\n".join(answer.strip() for answer in answers if answer.strip())
        return GenerationOutcomeModel(
            summary=(summary or NO_SUMMARY)[:MAX_SUMMARY_CHARS],
            files=kept,
            generated_cases=[case.id for case in cases[:done]],
            skipped_cases=[SkippedCaseModel(case_id=case_id, reason=reason) for case_id, reason in skipped],
            violations=violations,
            reverted_files=reverted,
            unverified_locators=unverified_locators(snapshot.root, kept),
            builds=builds,
            error=error,
        )

    def _write_batch(
        self,
        conversation: BaseConversation,
        prompt: str,
        project_root: Path,
        forwarder: EventForwarder,
        control: AgentControl | None,
    ) -> str:
        """Send one batch and let the agent work; nudge it when it stops before changing any file. Its final answer."""
        before = project_fingerprint(project_root)
        deadline = time.monotonic() + self.timeout_seconds
        answer = ask(conversation, prompt, deadline, _generation_failed, control)
        for attempt in range(1, self.nudges + 1):
            if (control is not None and control.is_stopped) or project_fingerprint(project_root) != before:
                break
            forwarder.notify(AgentEvent(
                RunEventType.AGENT_ERROR,
                RunEventLevel.INFO,
                NUDGE_NOTICE.format(attempt=attempt, total=self.nudges),
            ))
            answer = ask(conversation, NUDGE_MESSAGE, deadline, _generation_failed, control)
        return answer

    def _fix_build(
        self,
        conversation: BaseConversation,
        workspace: BaseWorkspace,
        command: TestCommandModel,
        forwarder: EventForwarder,
        control: AgentControl | None,
    ) -> bool:
        """Build the tests; while they do not build, hand the errors back to the agent. Whether they build at the end."""
        deadline = time.monotonic() + self.timeout_seconds
        for attempt in range(1, self.fix_rounds + 1):
            errors = _build_errors(workspace, command, forwarder)
            if errors is None:
                forwarder.notify(AgentEvent(RunEventType.AGENT_MESSAGE, RunEventLevel.INFO, BUILDS_NOTICE))
                return True
            if control is not None and control.is_stopped:
                return False
            forwarder.notify(AgentEvent(
                RunEventType.AGENT_ERROR,
                RunEventLevel.INFO,
                BUILD_ERRORS_NOTICE.format(attempt=attempt, total=self.fix_rounds),
                {"errors": errors},
            ))
            ask(conversation, FIX_MESSAGE.format(errors=errors), deadline, _generation_failed, control)
        if _build_errors(workspace, command, forwarder) is None:
            forwarder.notify(AgentEvent(RunEventType.AGENT_MESSAGE, RunEventLevel.INFO, BUILDS_NOTICE))
            return True
        forwarder.notify(AgentEvent(RunEventType.AGENT_ERROR, RunEventLevel.INFO, STILL_BROKEN_NOTICE.format(total=self.fix_rounds)))
        return False

    def _build_agent(self) -> Agent:
        llm = build_llm(self.llm_config)
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in GENERATOR_TOOL_NAMES],
            condenser=build_condenser(llm, self.llm_config.num_ctx),
        )


def _builds_before(workspace: BaseWorkspace, command: TestCommandModel, forwarder: EventForwarder) -> bool:
    """Install what the project declares, then check that it builds as it is (no check for this stack: False)."""
    if not command.check:
        return False
    if command.setup:
        _execute(workspace, command.setup)
    if _build_errors(workspace, command, forwarder) is None:
        return True
    forwarder.notify(AgentEvent(RunEventType.AGENT_ERROR, RunEventLevel.INFO, NOT_CHECKED_NOTICE))
    return False


def _build_errors(workspace: BaseWorkspace, command: TestCommandModel, forwarder: EventForwarder) -> str | None:
    """Build the tests without running them: None when they build, else the errors (the lines that name them)."""
    assert command.check is not None
    forwarder.notify(AgentEvent(
        RunEventType.AGENT_MESSAGE, RunEventLevel.DETAIL, CHECK_NOTICE.format(command=command.check),
    ))
    result = _execute(workspace, command.check)
    if result.exit_code == 0 and not result.timeout_occurred:
        return None
    output = f"{result.stdout}\n{result.stderr}".strip()
    lines = [line for line in output.splitlines() if "error" in line.lower()][:MAX_BUILD_ERROR_LINES]
    text = "\n".join(lines) if lines else output[-MAX_BUILD_ERROR_CHARS:]
    return text[:MAX_BUILD_ERROR_CHARS] or "The build failed without printing an error."


def _execute(workspace: BaseWorkspace, shell_command: str) -> Any:
    return workspace.execute_command(shell_command, cwd=SANDBOX_PROJECT_DIR, timeout=BUILD_CHECK_TIMEOUT_SECONDS)


def _generation_failed(reason: str) -> GenerationError:
    return GenerationError(ErrorMessages.GENERATION_FAILED.format(reason=reason))
