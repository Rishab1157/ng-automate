"""The test generator: an OpenHands agent that turns the user's test cases into tests in the project's own style.

It works in the writable test sandbox. Cases with a step that has no locator are left out (a locator is never
guessed). Its edits go through the same guards as the healer's (existing tests may not be weakened, build files must
still parse, no backup copies) and it may not delete files. Afterwards the generated tests are tagged or placed so
the master can run just them.
"""

import logging
import shutil
import time
from collections.abc import Callable
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
from app.agents.TestGeneratorAgent.GeneratedTestPlacement import (
    generated_selector,
    placement_rule,
    runnable_cases,
    tag_untagged_features,
    unverified_locators,
)
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
NOT_REACHED = "the test generator stopped before it finished this case"
GIT_DIR = ".git"


class TestGeneratorAgent:
    __test__ = False  # not a pytest test class

    def __init__(
        self,
        llm_config: LlmConfigModel,
        max_iterations: int = settings.GENERATOR_MAX_ITERATIONS,
        timeout_seconds: float = settings.GENERATOR_TIMEOUT_SECONDS,
        nudges: int = settings.GENERATOR_NUDGES,
        batch_size: int = settings.GENERATOR_BATCH_SIZE,
    ) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.nudges = nudges
        self.batch_size = max(1, batch_size)

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
        try:
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
                    build_generate_prompt(batch, fact_sheet, findings, command, placement_rule(fact_sheet), len(cases))
                    if done == 0
                    else build_next_batch_prompt(batch, done + 1, len(cases))
                )
                answers.append(self._write_batch(conversation, prompt, snapshot.root, forwarder, control))
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
        if tag_untagged_features(snapshot.root, kept):
            kept = [change for change in snapshot.changes() if change.path not in reverted]
        logger.info("Generator wrote %d file(s) for %d of %d case(s), %d rolled back", len(kept), done, len(cases), len(reverted))
        skipped += [(case.id, NOT_REACHED) for case in cases[done:]]
        summary = "\n\n".join(answer.strip() for answer in answers if answer.strip())
        return GenerationOutcomeModel(
            summary=(summary or NO_SUMMARY)[:MAX_SUMMARY_CHARS],
            files=kept,
            generated_cases=[case.id for case in cases[:done]],
            skipped_cases=[SkippedCaseModel(case_id=case_id, reason=reason) for case_id, reason in skipped],
            selector=generated_selector(fact_sheet, kept),
            violations=violations,
            reverted_files=reverted,
            unverified_locators=unverified_locators(snapshot.root, kept),
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

    def _build_agent(self) -> Agent:
        llm = build_llm(self.llm_config)
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in GENERATOR_TOOL_NAMES],
            condenser=build_condenser(llm, self.llm_config.num_ctx),
        )


def _generation_failed(reason: str) -> GenerationError:
    return GenerationError(ErrorMessages.GENERATION_FAILED.format(reason=reason))
