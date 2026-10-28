"""Healer agent: the prompt, the agent loop and the test-integrity guards, with a fake conversation.

The fake conversation "is" the agent: its run() edits the project folder the way the real agent would through
its tools, then answers with a summary. No Docker, no LLM.
"""

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.event import MessageEvent, ObservationEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.terminal import TerminalObservation, TerminalTool
from pydantic import SecretStr

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent, map_openhands_event
from app.agents.HealerAgent.HealerAgent import (
    BLOCKER_REJECTED_NOTICE,
    HEALER_TOOL_NAMES,
    NO_SUMMARY,
    REVIEW_NOTICE,
    CommandOutputLog,
    HealerAgent,
)
from app.agents.HealerAgent.HealerPrompts import MAX_OUTPUT_IN_PROMPT, build_heal_prompt
from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource
from app.models.healerModel import FileChangeModel, GuardViolationModel, HealOutcomeModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType
from app.models.testRunModel import FailureClassificationModel, FailureKind, TestCommandModel, TestRunResultModel
from app.utils.AgentPromptRules import TERMINAL_RULE
from app.utils.LlmInstance import MODEL_NOT_ANSWERING

healer_module = importlib.import_module("app.agents.HealerAgent.HealerAgent")

API_KEY = "sk-test-secret-0123456789"
COMMAND = TestCommandModel(tool="maven", command="mvn -B -ntp test", report_globs=["target/surefire-reports/TEST-*.xml"])
DEPENDENCY_OUTPUT = (
    "[ERROR] Failed to execute goal on project demo: Could not resolve dependencies for project demo:demo:jar:1.0: "
    "The following artifacts could not be resolved: org.testng:testng:jar:99.0.0"
)
TEST_SOURCE = """package tests;

public class LoginTest {
    @Test
    public void validLogin() {
        login("admin", "secret");
        Assert.assertEquals(title(), "Dashboard");
    }
}
"""


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=["pom.xml"], confidence=Confidence.HIGH)


FACT_SHEET = FactSheetModel(
    total_files=3,
    language_files={"Java": 2},
    primary_language=_fact("Java"),
    build_tool=_fact("Maven"),
    test_frameworks=_fact(["TestNG"]),
    automation_tools=_fact(["Selenium"]),
    bdd_tool=_fact(None),
    marker_files=["pom.xml"],
    test_dirs=["src/test"],
    feature_file_count=0,
    top_level_tree=["pom.xml", "src/"],
)


def _result(output: str = DEPENDENCY_OUTPUT, kind: FailureKind = FailureKind.DEPENDENCY_FAILURE) -> TestRunResultModel:
    return TestRunResultModel(
        command=COMMAND.command, exit_code=1, duration_seconds=12.0, total=0, passed=0, failed=0, errors=0, skipped=0,
        output_tail=output,
        classification=FailureClassificationModel(
            kind=kind, healable=True, reason="A dependency could not be resolved",
            evidence=["Could not resolve dependencies for project demo:demo:jar:1.0"],
        ),
    )


def _project(root: Path) -> Path:
    (root / "src" / "test" / "java" / "tests").mkdir(parents=True)
    (root / "pom.xml").write_text("<project>\n  <testng.version>99.0.0</testng.version>\n</project>\n", encoding="utf-8")
    (root / "src" / "test" / "java" / "tests" / "LoginTest.java").write_text(TEST_SOURCE, encoding="utf-8")
    return root


TEST_FILE = "src/test/java/tests/LoginTest.java"


class FakeConversation:
    def __init__(self, edit: Callable[[], None] | None, answer: str | None, error: BaseException | None,
                 agent: Any, outputs: list[str] | None = None, **kwargs: Any) -> None:
        self.edit, self.answer, self.error, self.outputs = edit, answer, error, outputs or []
        self.agent = agent
        self.kwargs = kwargs
        self.state = SimpleNamespace(events=[])
        self.messages: list[str] = []
        self.closed = False

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def run(self) -> None:
        if self.edit is not None:
            self.edit()
        for number, output in enumerate(self.outputs):
            # What the terminal printed for a command the agent ran.
            observation = TerminalObservation.from_text(output, command=f"command {number}", exit_code=1)
            event = ObservationEvent(observation=observation, action_id=f"a{number}", tool_name=TerminalTool.name,
                                     tool_call_id=f"t{number}")
            for callback in self.kwargs["callbacks"]:
                callback(event)
        if self.answer is not None:
            event = MessageEvent(source="agent", llm_message=Message(role="assistant", content=[TextContent(text=self.answer)]))
            for callback in [*self.kwargs["callbacks"], self.state.events.append]:
                callback(event)
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def conversation(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[FakeConversation]]:
    def install(edit: Callable[[], None] | None = None, answer: str | None = "Fixed the TestNG version in pom.xml.",
                error: BaseException | None = None, outputs: list[str] | None = None) -> list[FakeConversation]:
        created: list[FakeConversation] = []

        def factory(agent: Any, **kwargs: Any) -> FakeConversation:
            created.append(FakeConversation(edit, answer, error, agent, outputs=outputs, **kwargs))
            return created[-1]

        monkeypatch.setattr(healer_module, "Conversation", factory)
        return created

    return install


def _healer(**kwargs: Any) -> HealerAgent:
    return HealerAgent(LlmConfigModel(model="openai/gpt-4o", api_key=SecretStr(API_KEY)), **kwargs)


def _heal(project: Path, events: list[AgentEvent] | None = None, **kwargs: Any) -> Any:
    heal_kwargs = {key: kwargs.pop(key) for key in ("time_limit", "control", "previous") if key in kwargs}
    return _healer(**kwargs).heal(
        LocalWorkspace(working_dir=str(project)), project, _result(), COMMAND, FACT_SHEET,
        on_event=events.append if events is not None else None, **heal_kwargs,
    )


# ---------------------------------------------------------------- prompt


def test_prompt_has_the_failure_command_stack_and_rules() -> None:
    prompt = build_heal_prompt(_result(), COMMAND, FACT_SHEET)

    assert "mvn -B -ntp test" in prompt
    assert "dependency_failure" in prompt
    assert "Java · Maven · TestNG · Selenium" in prompt
    assert "org.testng:testng:jar:99.0.0" in prompt
    assert "NEVER delete, skip or disable tests" in prompt
    assert "try another one" in prompt and "BLOCKED:" in prompt


def test_prompt_keeps_only_the_end_of_long_output_and_no_code_fence_breaks() -> None:
    output = "start-marker " + "x" * MAX_OUTPUT_IN_PROMPT + " ```end``` marker"

    prompt = build_heal_prompt(_result(output), COMMAND, FACT_SHEET)

    assert "start-marker" not in prompt
    assert "'''end''' marker" in prompt
    assert "Known fixes" not in prompt


USER_BLOCK_HEADER = (
    "The user's instructions for this run (they come first: where they differ from the rules below, follow them; "
    "never delete or weaken an existing test):"
)


def test_the_users_instructions_come_right_before_the_rules_only_when_given() -> None:
    prompt = build_heal_prompt(_result(), COMMAND, FACT_SHEET, instructions=["Keep TestNG 7.x.", "  "])

    # After the facts of the failure, before the rules; a blank instruction is left out.
    assert f"\n{USER_BLOCK_HEADER}\n- Keep TestNG 7.x.\n\nFind the cause yourself" in prompt
    assert prompt.index("Key lines:") < prompt.index(USER_BLOCK_HEADER)
    for nothing in ((), ["", "  "]):
        assert "The user's instructions" not in build_heal_prompt(_result(), COMMAND, FACT_SHEET, instructions=nothing)


def test_a_model_that_does_not_answer_stops_the_heal_before_it_starts(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = conversation()
    monkeypatch.setattr(healer_module, "check_model", lambda workspace, config: MODEL_NOT_ANSWERING.format(seconds=90))

    outcome = _heal(_project(tmp_path))

    assert outcome.error == f"The healer stopped: {MODEL_NOT_ANSWERING.format(seconds=90)}"
    assert created == [] and outcome.changes == []


def test_scratch_output_is_kept_out_of_the_project() -> None:
    # A real run: an agent ran its build as `... | tee result.txt` and left result.txt in the project root.
    flat = " ".join(build_heal_prompt(_result(), COMMAND, FACT_SHEET).split())

    assert "saved command output (`| tee`, `> file`) outside the project (for example in /tmp)" in flat


def test_the_healer_is_told_the_terminal_can_check_an_answer_that_looks_wrong() -> None:
    # A real run: the grep tool answered "No files found" for files that were there, and the agent believed it.
    assert TERMINAL_RULE in build_heal_prompt(_result(), COMMAND, FACT_SHEET)


def test_the_quicker_check_is_offered_only_when_the_project_has_one() -> None:
    with_check = COMMAND.model_copy(update={"check": "mvn -B -ntp -q -DskipTests test-compile"})

    without = build_heal_prompt(_result(), COMMAND, FACT_SHEET)
    offered = build_heal_prompt(_result(), with_check, FACT_SHEET)

    assert "- You may run the test command to check your fix.\n" in without and "quicker" not in without
    assert (
        "- You may run the test command to check your fix (for a build or load error, the quicker check "
        "`mvn -B -ntp -q -DskipTests test-compile` is enough: it builds or loads the tests without running them).\n"
    ) in offered


def test_a_playwright_prompt_has_the_generic_rules_and_no_other_stacks_recipes() -> None:
    # The rules say how to work; the project and its failure give the details (a javac recipe once reached every stack).
    facts = FACT_SHEET.model_copy(update={
        "language_files": {"TypeScript": 2}, "primary_language": _fact("TypeScript"), "build_tool": _fact("npm"),
        "test_frameworks": _fact(["Playwright Test"]), "automation_tools": _fact(["Playwright"]),
        "marker_files": ["package.json", "playwright.config.ts"],
    })
    command = TestCommandModel(
        tool="playwright", command="npx playwright test", setup="npm ci", check="npx playwright test --list"
    )
    error = "tests/ngauto/login.spec.ts:3:27 - error TS2307: Cannot find module '../pages/LoginPage'"
    result = TestRunResultModel(
        command=command.command, exit_code=1, duration_seconds=2.0, total=0, passed=0, failed=0, errors=0, skipped=0,
        output_tail=error,
        classification=FailureClassificationModel(
            kind=FailureKind.BUILD_FAILURE, healable=True, reason="r", evidence=[error]
        ),
    )

    prompt = build_heal_prompt(result, command, facts)
    flat = " ".join(prompt.split())

    assert "TypeScript · npm · Playwright Test · Playwright" in prompt
    for other in ("mvn", "cannot find symbol", "insert_line", "Java/JS/TS"):
        assert other not in prompt
    assert "(for a build or load error, the quicker check `npx playwright test --list` is enough" in prompt
    # The user watches the browser live: never headless, maximized where the framework allows it.
    assert "Never make the browser headless" in flat and "maximized window" in flat
    assert "A name, type or module it cannot find is usually a missing import" in flat
    assert "(not `insert`: its first line loses its indentation)" in flat
    assert "(in that file's comment syntax)" in prompt and "(sudo works)" in prompt
    assert "never write a new test, or a copy of a test under another name, to get around a failure" in flat
    assert "never add build tools, compilers or type packages just to check your fix" in flat


# ---------------------------------------------------------------- heal()


def test_legit_fix_is_kept_and_reported(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)

    def fix() -> None:
        pom = project / "pom.xml"
        pom.write_text(pom.read_text(encoding="utf-8").replace("99.0.0", "7.10.2"), encoding="utf-8")

    created = conversation(edit=fix)

    outcome = _heal(project, max_iterations=9)

    assert outcome.summary == "Fixed the TestNG version in pom.xml."
    assert [(c.path, c.change) for c in outcome.changes] == [("pom.xml", "modified")]
    assert "+  <testng.version>7.10.2</testng.version>" in outcome.changes[0].diff
    assert outcome.violations == []
    [fake] = created
    assert fake.kwargs["max_iteration_per_run"] == 9
    assert fake.kwargs["visualizer"] is None
    assert fake.closed
    assert "mvn -B -ntp test" in fake.messages[0]
    assert {tool.name for tool in fake.agent.tools} == set(HEALER_TOOL_NAMES)


def test_removed_assertion_is_kept_and_reported(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # Nothing is ever rolled back (the user's rule): the user is asked to review the change instead.
    project = _project(tmp_path)
    test_file = project / TEST_FILE
    weakened = TEST_SOURCE.replace('        Assert.assertEquals(title(), "Dashboard");\n', "")

    def cheat() -> None:
        (project / "pom.xml").write_text("<project>\n  <testng.version>7.10.2</testng.version>\n</project>\n", encoding="utf-8")
        test_file.write_text(weakened, encoding="utf-8")

    conversation(edit=cheat)
    events: list[AgentEvent] = []

    outcome = _heal(project, events)

    assert test_file.read_text(encoding="utf-8") == weakened
    assert [v.rule for v in outcome.violations] == ["assertion_removed"]
    assert [c.path for c in outcome.changes] == ["pom.xml", TEST_FILE]
    [notice] = [e for e in events if e.type == RunEventType.AGENT_ERROR]
    assert notice.message == REVIEW_NOTICE.format(path=TEST_FILE, rules="assertion_removed")


def test_skipped_test_is_kept_and_reported(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    test_file = project / TEST_FILE
    skipped = TEST_SOURCE.replace("@Test", "@Test(enabled = false)")
    conversation(edit=lambda: test_file.write_text(skipped, encoding="utf-8"))

    outcome = _heal(project)

    assert test_file.read_text(encoding="utf-8") == skipped
    assert "test_skipped" in {v.rule for v in outcome.violations}
    assert [c.path for c in outcome.changes] == [TEST_FILE]


def test_deleted_test_file_stays_deleted_and_is_reported(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    conversation(edit=lambda: (project / TEST_FILE).unlink())

    outcome = _heal(project)

    assert not (project / TEST_FILE).exists()
    assert [(c.path, c.change) for c in outcome.changes] == [(TEST_FILE, "deleted")]
    assert {v.rule for v in outcome.violations} == {"assertion_removed", "test_removed"}


def test_harmless_test_file_edit_is_kept(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    test_file = project / TEST_FILE
    fixed = TEST_SOURCE.replace("package tests;\n", "package tests;\n\nimport org.testng.Assert;\nimport org.testng.annotations.Test;\n")
    conversation(edit=lambda: test_file.write_text(fixed, encoding="utf-8"))

    outcome = _heal(project)

    assert test_file.read_text(encoding="utf-8") == fixed
    assert [c.path for c in outcome.changes] == [TEST_FILE]
    assert outcome.violations == []


def test_no_answer_gives_a_default_summary(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    conversation(answer=None)

    assert _heal(_project(tmp_path)).summary == NO_SUMMARY


def test_step_limit_still_returns_the_changes(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    limit = ConversationRunError(
        "c-1", RuntimeError("limit"),
        conversation_error=ConversationErrorEvent(source="environment", code="MaxIterationsReached", detail="limit"),
    )
    conversation(edit=lambda: (project / "pom.xml").write_text("<project/>\n", encoding="utf-8"), answer=None, error=limit)

    outcome = _heal(project)

    assert [c.path for c in outcome.changes] == ["pom.xml"]


def test_model_failure_is_reported_on_the_outcome(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    failure = ConversationRunError(
        "c-1", RuntimeError("auth"),
        conversation_error=ConversationErrorEvent(source="environment", code="AuthenticationError", detail=API_KEY),
    )
    created = conversation(answer=None, error=failure)

    outcome = _heal(_project(tmp_path))

    assert outcome.error is not None and outcome.error.startswith("The healer stopped: ")
    assert API_KEY not in outcome.error
    assert outcome.summary == NO_SUMMARY
    assert created[0].closed


def test_edits_made_before_the_agent_stopped_are_still_checked(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    test_file = project / TEST_FILE
    ignored = TEST_SOURCE.replace("@Test", "@Ignore @Test")
    timeout = ConversationRunError("c-1", TimeoutError("run timed out"))
    conversation(edit=lambda: test_file.write_text(ignored, encoding="utf-8"), answer=None, error=timeout)

    outcome = _heal(project)

    assert outcome.error == "The healer stopped: the agent did not finish in time"
    assert test_file.read_text(encoding="utf-8") == ignored
    assert [c.path for c in outcome.changes] == [TEST_FILE]
    assert "test_skipped" in {v.rule for v in outcome.violations}


def test_the_heal_deadline_comes_from_the_healer_timeout(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines: list[float] = []
    real_ask = healer_module.ask

    def recording_ask(conversation: Any, message: str, deadline: float, failure: Any, *rest: Any) -> str:
        deadlines.append(deadline - time.monotonic())
        return real_ask(conversation, message, deadline, failure, *rest)

    monkeypatch.setattr(healer_module, "ask", recording_ask)
    conversation()

    _heal(_project(tmp_path), timeout_seconds=120)

    assert 110 < deadlines[0] <= 120


def test_broken_build_file_is_kept_and_reported(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    pom = project / "pom.xml"
    broken = "003cproject003e\n  <testng.version>7.10.2</testng.version>\n"
    conversation(edit=lambda: pom.write_text(broken, encoding="utf-8"))

    outcome = _heal(project)

    assert pom.read_text(encoding="utf-8") == broken
    assert [v.rule for v in outcome.violations] == ["build_file_broken"]
    assert [c.path for c in outcome.changes] == ["pom.xml"]


def test_backup_copy_is_kept_and_reported_and_git_repo_made_by_the_agent_is_deleted(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)

    def tidy_up_badly() -> None:
        (project / "pom.xml.backup").write_text((project / "pom.xml").read_text(encoding="utf-8"), encoding="utf-8")
        (project / ".git" / "objects").mkdir(parents=True)
        (project / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

    conversation(edit=tidy_up_badly)

    outcome = _heal(project)

    assert (project / "pom.xml.backup").exists()
    assert not (project / ".git").exists()  # the agent's own tool, never part of the project
    assert [c.path for c in outcome.changes] == ["pom.xml.backup"]
    assert [v.rule for v in outcome.violations] == ["stray_file"]


def test_earlier_fixes_are_described_in_the_prompt(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    created = conversation()
    earlier = HealOutcomeModel(
        summary="Changed the TestNG version to 7.12.0.",
        changes=[FileChangeModel(path="pom.xml", change="modified")],
        violations=[GuardViolationModel(file="pom.xml", rule="build_file_broken", line="003cdependency003e")],
    )
    project = _project(tmp_path)

    _healer().heal(
        LocalWorkspace(working_dir=str(project)), project, _result(), COMMAND, FACT_SHEET, previous=[earlier]
    )

    prompt = created[0].messages[0]
    assert "Earlier fixes in this run did not make the tests pass" in prompt
    assert (
        "Fix 1: Changed the TestNG version to 7.12.0. Changed: pom.xml. Broke the rules in pom.xml (build_file_broken)."
        in prompt
    )


def test_the_users_instructions_reach_the_healers_task_message(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    created = conversation()
    given, not_given = _project(tmp_path / "given"), _project(tmp_path / "not-given")

    _healer().heal(
        LocalWorkspace(working_dir=str(given)), given, _result(), COMMAND, FACT_SHEET,
        instructions="Keep the TestNG version the project uses.",
    )
    _healer().heal(LocalWorkspace(working_dir=str(not_given)), not_given, _result(), COMMAND, FACT_SHEET)

    assert f"{USER_BLOCK_HEADER}\n- Keep the TestNG version the project uses.\n" in created[0].messages[0]
    assert "The user's instructions" not in created[1].messages[0]


def test_agent_that_changed_nothing_is_nudged_once(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    from app.agents.HealerAgent.HealerAgent import NUDGE_MESSAGE

    created = conversation(answer="I will look at the pom next.")

    _heal(_project(tmp_path))

    assert created[0].messages[1:] == [NUDGE_MESSAGE]


# ---------------------------------------------------------------- blocker, time


NO_NETWORK = "curl: (6) Could not resolve host: storage.googleapis.com"


def test_a_proven_blocker_is_accepted(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    created = conversation(
        answer=f"BLOCKED: the sandbox has no network, so the driver cannot be downloaded.\n`{NO_NETWORK}`",
        outputs=["Downloading...", NO_NETWORK],
    )

    outcome = _heal(_project(tmp_path))

    assert outcome.blocker is not None and outcome.blocker.startswith("the sandbox has no network")
    assert not outcome.blocker_rejected
    assert created[0].messages[1:] == []  # a blocked agent is not nudged


def test_a_blocker_without_matching_command_output_is_rejected(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    conversation(answer=f"**BLOCKED:** no network\n{NO_NETWORK}", outputs=["BUILD SUCCESS"])
    events: list[AgentEvent] = []

    outcome = _heal(_project(tmp_path), events)

    assert outcome.blocker is None and outcome.blocker_rejected
    assert BLOCKER_REJECTED_NOTICE in [event.message for event in events]


def test_quoting_the_test_failure_itself_is_not_a_blocker(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # The agent ran the tests again and quotes the failure it was asked to fix: that is not new information.
    conversation(answer=f"BLOCKED: cannot resolve it\n{DEPENDENCY_OUTPUT}", outputs=[DEPENDENCY_OUTPUT])

    outcome = _heal(_project(tmp_path))

    assert outcome.blocker is None and outcome.blocker_rejected


def test_the_time_limit_sets_the_deadline_and_the_working_time_is_measured(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines: list[float] = []
    real_ask = healer_module.ask

    def recording_ask(conversation: Any, message: str, deadline: float, failure: Any, *rest: Any) -> str:
        deadlines.append(deadline - time.monotonic())
        return real_ask(conversation, message, deadline, failure, *rest)

    monkeypatch.setattr(healer_module, "ask", recording_ask)
    project = _project(tmp_path)
    conversation(edit=lambda: (project / "pom.xml").write_text("<project/>\n", encoding="utf-8"))

    outcome = _heal(project, timeout_seconds=3600, time_limit=90)

    assert 80 < deadlines[0] <= 90
    assert 0 <= outcome.seconds < 30


def test_the_healer_sees_the_setup_that_installs_the_projects_dependencies() -> None:
    from app.agents.HealerAgent.HealerPrompts import build_heal_prompt
    from app.models.testRunModel import FailureClassificationModel, FailureKind, TestCommandModel, TestRunResultModel

    command = TestCommandModel(tool="pytest", command="python -m pytest -q", setup="python -m pip install -q -r requirements.txt")
    result = TestRunResultModel(
        command=command.command, exit_code=1, duration_seconds=1.0, total=0, passed=0, failed=0, errors=0, skipped=0,
        cases=[], output_tail="ERROR: No matching distribution found for selenium==99.0.0",
        classification=FailureClassificationModel(kind=FailureKind.DEPENDENCY_FAILURE, healable=True, reason="r"),
    )

    prompt = build_heal_prompt(result, command, FACT_SHEET)

    assert "python -m pip install -q -r requirements.txt && python -m pytest -q" in prompt


def test_a_failure_inside_the_tests_shows_the_failing_tests_not_the_build_log() -> None:
    from app.models.testRunModel import FailureClassificationModel, FailureKind, TestCommandModel, TestRunResultModel

    def prompt_for(kind: FailureKind) -> str:
        result = TestRunResultModel(
            command="mvn -B -ntp test", exit_code=1, duration_seconds=1.0, total=2, passed=1, failed=1, errors=0,
            skipped=0, cases=[], output_tail="WARNING: Unable to find CDP implementation matching 153",
            classification=FailureClassificationModel(
                kind=kind, healable=True, reason="r", evidence=["com.demo.CartTest.remove: element not found (at ...)"]
            ),
        )
        return build_heal_prompt(result, TestCommandModel(tool="maven", command="mvn -B -ntp test"), FACT_SHEET)

    locator = prompt_for(FailureKind.LOCATOR_FAILURE)
    dependency = prompt_for(FailureKind.DEPENDENCY_FAILURE)

    # A warning in the log once sent the healer to pom.xml for a test that checked an element that had gone.
    assert "Unable to find CDP" not in locator and "com.demo.CartTest.remove" in locator
    assert "Start in those files" in locator
    assert "Unable to find CDP" in dependency  # for build-level failures the log is the evidence


def test_a_long_output_from_the_agent_server_is_read_without_saving_it_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent server's observation names its own folder for a long output's full text, as a relative path.
    monkeypatch.chdir(tmp_path)
    observation = TerminalObservation.from_text("x" * 40_000, command="mvn test", exit_code=1).model_copy(
        update={"full_output_save_dir": "workspace/conversations/abc/observations"}
    )
    event = ObservationEvent(observation=observation, action_id="a1", tool_name=TerminalTool.name, tool_call_id="t1")
    outputs = CommandOutputLog()

    outputs(event)
    mapped = map_openhands_event(event)

    assert "<response clipped>" in outputs.text() and mapped is not None and "xxx" in mapped.data["output"]
    assert list(tmp_path.iterdir()) == []
