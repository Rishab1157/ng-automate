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

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.HealerAgent.HealerAgent import BLOCKER_REJECTED_NOTICE, HEALER_TOOL_NAMES, NO_SUMMARY, HealerAgent
from app.agents.HealerAgent.HealerPrompts import MAX_OUTPUT_IN_PROMPT, build_heal_prompt
from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource
from app.models.healerModel import GuardViolationModel, HealOutcomeModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType
from app.models.testRunModel import FailureClassificationModel, FailureKind, TestCommandModel, TestRunResultModel

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
    assert outcome.violations == [] and outcome.reverted_files == []
    [fake] = created
    assert fake.kwargs["max_iteration_per_run"] == 9
    assert fake.kwargs["visualizer"] is None
    assert fake.closed
    assert "mvn -B -ntp test" in fake.messages[0]
    assert {tool.name for tool in fake.agent.tools} == set(HEALER_TOOL_NAMES)


def test_removed_assertion_is_rolled_back_but_other_fixes_stay(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    test_file = project / TEST_FILE

    def cheat() -> None:
        (project / "pom.xml").write_text("<project>\n  <testng.version>7.10.2</testng.version>\n</project>\n", encoding="utf-8")
        test_file.write_text(TEST_SOURCE.replace('        Assert.assertEquals(title(), "Dashboard");\n', ""), encoding="utf-8")

    conversation(edit=cheat)
    events: list[AgentEvent] = []

    outcome = _heal(project, events)

    assert test_file.read_text(encoding="utf-8") == TEST_SOURCE
    assert outcome.reverted_files == [TEST_FILE]
    assert [v.rule for v in outcome.violations] == ["assertion_removed"]
    assert [c.path for c in outcome.changes] == ["pom.xml"]
    [notice] = [e for e in events if e.type == RunEventType.AGENT_ERROR]
    assert TEST_FILE in notice.message and "assertion_removed" in notice.message


def test_skipped_test_is_rolled_back(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    test_file = project / TEST_FILE
    conversation(edit=lambda: test_file.write_text(TEST_SOURCE.replace("@Test", "@Test(enabled = false)"), encoding="utf-8"))

    outcome = _heal(project)

    assert test_file.read_text(encoding="utf-8") == TEST_SOURCE
    assert "test_skipped" in {v.rule for v in outcome.violations}
    assert outcome.changes == []


def test_deleted_test_file_is_restored(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    conversation(edit=lambda: (project / TEST_FILE).unlink())

    outcome = _heal(project)

    assert (project / TEST_FILE).read_text(encoding="utf-8") == TEST_SOURCE
    assert outcome.reverted_files == [TEST_FILE]
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
    timeout = ConversationRunError("c-1", TimeoutError("run timed out"))
    conversation(
        edit=lambda: test_file.write_text(TEST_SOURCE.replace("@Test", "@Ignore @Test"), encoding="utf-8"),
        answer=None,
        error=timeout,
    )

    outcome = _heal(project)

    assert outcome.error == "The healer stopped: the agent did not finish in time"
    assert test_file.read_text(encoding="utf-8") == TEST_SOURCE
    assert outcome.reverted_files == [TEST_FILE]


def test_the_heal_deadline_comes_from_the_healer_timeout(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines: list[float] = []
    real_ask = healer_module.ask

    def recording_ask(conversation: Any, message: str, deadline: float, failure: Any, control: Any = None) -> str:
        deadlines.append(deadline - time.monotonic())
        return real_ask(conversation, message, deadline, failure, control)

    monkeypatch.setattr(healer_module, "ask", recording_ask)
    conversation()

    _heal(_project(tmp_path), timeout_seconds=120)

    assert 110 < deadlines[0] <= 120


def test_broken_build_file_is_rolled_back(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)
    pom = project / "pom.xml"
    original = pom.read_text(encoding="utf-8")
    conversation(edit=lambda: pom.write_text("003cproject003e\n  <testng.version>7.10.2</testng.version>\n", encoding="utf-8"))

    outcome = _heal(project)

    assert pom.read_text(encoding="utf-8") == original
    assert outcome.reverted_files == ["pom.xml"]
    assert [v.rule for v in outcome.violations] == ["build_file_broken"]
    assert outcome.changes == []


def test_backup_copy_is_removed_and_git_repo_made_by_the_agent_is_deleted(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)

    def tidy_up_badly() -> None:
        (project / "pom.xml.backup").write_text((project / "pom.xml").read_text(encoding="utf-8"), encoding="utf-8")
        (project / ".git" / "objects").mkdir(parents=True)
        (project / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")

    conversation(edit=tidy_up_badly)

    outcome = _heal(project)

    assert not (project / "pom.xml.backup").exists()
    assert not (project / ".git").exists()
    assert outcome.reverted_files == ["pom.xml.backup"]
    assert [v.rule for v in outcome.violations] == ["stray_file"]


def test_earlier_fixes_are_described_in_the_prompt(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    created = conversation()
    earlier = HealOutcomeModel(
        summary="Changed the TestNG version to 7.12.0.",
        changes=[],
        violations=[GuardViolationModel(file="pom.xml", rule="build_file_broken", line="003cdependency003e")],
        reverted_files=["pom.xml"],
    )
    project = _project(tmp_path)

    _healer().heal(
        LocalWorkspace(working_dir=str(project)), project, _result(), COMMAND, FACT_SHEET, previous=[earlier]
    )

    prompt = created[0].messages[0]
    assert "Earlier fixes in this run did not make the tests pass" in prompt
    assert "Fix 1: Changed the TestNG version to 7.12.0. Rolled back pom.xml (build_file_broken)." in prompt


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

    def recording_ask(conversation: Any, message: str, deadline: float, failure: Any, control: Any = None) -> str:
        deadlines.append(deadline - time.monotonic())
        return real_ask(conversation, message, deadline, failure, control)

    monkeypatch.setattr(healer_module, "ask", recording_ask)
    project = _project(tmp_path)
    conversation(edit=lambda: (project / "pom.xml").write_text("<project/>\n", encoding="utf-8"))

    outcome = _heal(project, timeout_seconds=3600, time_limit=90)

    assert 80 < deadlines[0] <= 90
    assert 0 <= outcome.seconds < 30
