"""Test generator agent: prompt, batches, guards and assumed locators, with a fake conversation (no Docker, no LLM)."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.workspace import LocalWorkspace
from pydantic import SecretStr

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.TestGeneratorAgent.ProjectTestBriefing import project_test_briefing
from app.agents.TestGeneratorAgent.TestGeneratorAgent import (
    GENERATOR_TOOL_NAMES,
    NUDGE_MESSAGE,
    NUDGE_NOTICE,
    REVIEW_NOTICE,
    TestGeneratorAgent,
)
from app.config import settings
from app.core.exceptions import ErrorMessages, ValidationError
from app.models.analyzerModel import AnalyzerFindingsModel, Confidence, FactModel, FactSheetModel, FactSource, ImportantPathModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestStepModel
from app.models.testRunModel import TestCommandModel
from app.utils.LlmInstance import MODEL_NOT_ANSWERING
from tests.test_healer_agent import FakeConversation

generator_module = importlib.import_module("app.agents.TestGeneratorAgent.TestGeneratorAgent")

COMMAND = TestCommandModel(tool="maven", command="mvn -B -ntp test", report_globs=[])
FEATURE = "src/test/resources/features/ngauto/login.feature"
STEPS = "src/test/java/com/demo/steps/LoginSteps.java"
EXISTING_STEPS = """package com.demo.steps;

public class LoginSteps {
    @Then("the title is {string}")
    public void title(String expected) {
        Assert.assertEquals(driver.getTitle(), expected);
    }
}
"""


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=["pom.xml"], confidence=Confidence.HIGH)


FACT_SHEET = FactSheetModel(
    total_files=4, language_files={"Java": 3}, primary_language=_fact("Java"), build_tool=_fact("Maven"),
    test_frameworks=_fact(["TestNG"]), automation_tools=_fact(["Selenium"]), bdd_tool=_fact("Cucumber"),
    marker_files=["pom.xml"], test_dirs=["src/test"], feature_file_count=1, top_level_tree=["pom.xml", "src/"],
)
FINDINGS = AnalyzerFindingsModel(
    project_summary=_fact("Demo shop tests"),
    architecture_pattern=_fact("Page Object Model with Cucumber"),
    test_command=_fact("mvn test"),
    reporting_tools=_fact(["ExtentReports"]),
    important_paths=[ImportantPathModel(path="src/test/java/com/demo/pages", role="page objects")],
)
LOGIN = TestCaseSpecModel(
    id="LOGIN-1",
    title="Valid login",
    tags=["smoke"],
    steps=[
        TestStepModel(action="open", target="https://shop.example/login"),
        TestStepModel(action="type", target="id=username", value="admin"),
        TestStepModel(action="verify_text", target="css=h1", expected="Dashboard"),
    ],
)
NO_LOCATOR = TestCaseSpecModel(id="LOGOUT-1", title="Logout", steps=[TestStepModel(action="click", needs_locator=True)])
DATA = TestDataSetModel(source_format=TestDataFormat.JSON, cases=[LOGIN, NO_LOCATOR])


def _project(root: Path) -> Path:
    (root / "src/test/java/com/demo/steps").mkdir(parents=True)
    (root / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    (root / STEPS).write_text(EXISTING_STEPS, encoding="utf-8")
    return root


@pytest.fixture
def conversation(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[FakeConversation]]:
    def install(edit: Callable[[], None] | None = None, answer: str | None = "Wrote login.feature.",
                error: BaseException | None = None) -> list[FakeConversation]:
        created: list[FakeConversation] = []

        def factory(agent: Any, **kwargs: Any) -> FakeConversation:
            created.append(FakeConversation(edit, answer, error, agent, **kwargs))
            return created[-1]

        monkeypatch.setattr(generator_module, "Conversation", factory)
        return created

    return install


def _generate(project: Path, data: TestDataSetModel = DATA, events: list[AgentEvent] | None = None) -> Any:
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o", api_key=SecretStr("sk-test-0123456789")), max_iterations=5)
    return agent.generate(
        LocalWorkspace(working_dir=str(project)), project, data, FACT_SHEET, FINDINGS, COMMAND,
        on_event=events.append if events is not None else None,
    )


def test_generated_feature_is_kept_as_written(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)

    def write_feature() -> None:
        (project / FEATURE).parent.mkdir(parents=True)
        (project / FEATURE).write_text("Feature: Login\n  Scenario: LOGIN-1 Valid login\n", encoding="utf-8")

    created = conversation(edit=write_feature)

    outcome = _generate(project)

    assert outcome.generated_cases == ["LOGIN-1", "LOGOUT-1"]  # LOGOUT-1's locator is assumed, never a reason to drop it
    assert outcome.skipped_cases == []
    assert [(f.path, f.change) for f in outcome.files] == [(FEATURE, "added")]
    assert (project / FEATURE).read_text(encoding="utf-8").startswith("Feature: Login")  # never rewritten afterwards
    assert outcome.summary == "Wrote login.feature."
    [fake] = created
    prompt = fake.messages[0]
    assert "Case LOGIN-1: Valid login [tags: smoke]" in prompt
    assert 'type target=id=username value="admin"' in prompt
    assert "Case LOGOUT-1: Logout" in prompt
    assert "NGAUTO-UNVERIFIED" in prompt and "Never leave a case out" in prompt
    assert "Preconditions" in prompt
    assert "src/test/java/com/demo/pages: page objects" in prompt
    assert "A sample of the project's own tests and shared code" in prompt
    assert "class LoginSteps" in prompt  # the project's own helper is inlined, so the model can reuse its API
    assert "new folder named 'ngauto'" in prompt  # the one fixed place: the rest is the framework's own
    assert "new .feature file" not in prompt  # no per-stack placement, also in a Cucumber project
    assert {tool.name for tool in fake.agent.tools} == set(GENERATOR_TOOL_NAMES)


def test_weakening_an_existing_step_or_deleting_a_file_is_kept_and_reported(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # Nothing is ever rolled back (the user's rule): the changes stay, and the user is asked to review them.
    project = _project(tmp_path)
    weakened = EXISTING_STEPS.replace("        Assert.assertEquals(driver.getTitle(), expected);\n", "")

    def misbehave() -> None:
        (project / STEPS).write_text(weakened, encoding="utf-8")
        (project / "pom.xml").unlink()

    conversation(edit=misbehave)
    events: list[AgentEvent] = []

    outcome = _generate(project, events=events)

    assert (project / STEPS).read_text(encoding="utf-8") == weakened
    assert not (project / "pom.xml").exists()
    assert {(v.file, v.rule) for v in outcome.violations} == {("pom.xml", "file_deleted"), (STEPS, "assertion_removed")}
    assert [(f.path, f.change) for f in outcome.files] == [("pom.xml", "deleted"), (STEPS, "modified")]
    assert [e.message for e in events if e.type == RunEventType.AGENT_ERROR] == [
        REVIEW_NOTICE.format(path="pom.xml", rules="file_deleted"),
        REVIEW_NOTICE.format(path=STEPS, rules="assertion_removed"),
    ]


def test_a_model_that_does_not_answer_stops_the_generator_before_it_starts(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = conversation()
    not_answering = MODEL_NOT_ANSWERING.format(seconds=90)
    monkeypatch.setattr(generator_module, "check_model", lambda workspace, config: not_answering)

    outcome = _generate(_project(tmp_path))

    assert outcome.error == f"The test generator stopped: {MODEL_NOT_ANSWERING.format(seconds=90)}"
    assert created == [] and outcome.files == []


def test_stopping_early_keeps_what_was_written(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)

    def write_feature() -> None:
        (project / FEATURE).parent.mkdir(parents=True)
        (project / FEATURE).write_text("@ngauto\nFeature: Login\n", encoding="utf-8")

    conversation(edit=write_feature, answer=None, error=ConversationRunError("c", TimeoutError("slow")))

    outcome = _generate(project)

    assert outcome.error == "The test generator stopped: the agent did not finish in time"
    assert [f.path for f in outcome.files] == [FEATURE]


def test_nothing_to_generate_when_no_case_has_steps(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    created = conversation()
    empty = TestCaseSpecModel(id="EMPTY-1", title="Nothing", steps=[])

    with pytest.raises(ValidationError) as caught:
        _generate(_project(tmp_path), TestDataSetModel(source_format=TestDataFormat.CSV, cases=[empty]))

    assert caught.value.message == ErrorMessages.NO_RUNNABLE_TEST_CASES
    assert created == []


def test_generator_that_stops_to_explain_is_told_to_continue(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    runs: list[int] = []

    def write_on_the_second_turn() -> None:
        runs.append(1)
        if len(runs) == 2:
            (project / FEATURE).parent.mkdir(parents=True)
            (project / FEATURE).write_text("@ngauto\nFeature: Login\n", encoding="utf-8")

    created = conversation(edit=write_on_the_second_turn, answer="Now I understand the structure.")
    events: list[AgentEvent] = []

    outcome = _generate(project, events=events)

    assert [f.path for f in outcome.files] == [FEATURE]
    assert created[0].messages[1:] == [NUDGE_MESSAGE]
    assert [e.message for e in events if e.type == RunEventType.AGENT_ERROR] == [
        NUDGE_NOTICE.format(attempt=1, total=settings.GENERATOR_NUDGES)
    ]


def test_generator_that_never_writes_is_nudged_a_limited_number_of_times(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    created = conversation(answer="Let me think about it.")

    outcome = _generate(_project(tmp_path))

    assert outcome.files == []
    assert created[0].messages[1:] == [NUDGE_MESSAGE] * settings.GENERATOR_NUDGES


def test_changes_to_the_projects_own_files_are_kept(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # The new tests may need the framework changed (a locator, a step, a dependency). In a real run, rolling such
    # changes back after the build check left generated tests that no longer compiled: they all stay.
    project = _project(tmp_path)
    page = project / "src/test/java/com/demo/pages/LoginPage.java"
    page.parent.mkdir(parents=True)
    original = 'public class LoginPage {\n  private final By username = By.id("username");\n}\n'
    page.write_text(original, encoding="utf-8")
    pom = project / "pom.xml"
    pom.write_text("<project>\n  <dependencies>\n  </dependencies>\n</project>\n", encoding="utf-8")
    dependency = "    <dependency><groupId>io.cucumber</groupId><artifactId>cucumber-java</artifactId></dependency>\n"
    with_dependency = pom.read_text(encoding="utf-8").replace("  </dependencies>", dependency + "  </dependencies>")
    grown = EXISTING_STEPS.replace("}\n}\n", '}\n\n    @When("the user opens {string}")\n    public void open(String url) { driver.get(url); }\n}\n')
    new_steps = "src/test/java/com/demo/steps/ngauto/CartSteps.java"
    changed_page = original.replace('By.id("username")', 'By.id("user-name")')

    def edit() -> None:
        page.write_text(changed_page, encoding="utf-8")
        (project / STEPS).write_text(grown, encoding="utf-8")
        pom.write_text(with_dependency, encoding="utf-8")
        (project / new_steps).parent.mkdir(parents=True)
        (project / new_steps).write_text("public class CartSteps {}\n", encoding="utf-8")

    conversation(edit=edit)

    outcome = _generate(project)

    assert page.read_text(encoding="utf-8") == changed_page
    assert (project / STEPS).read_text(encoding="utf-8") == grown
    assert pom.read_text(encoding="utf-8") == with_dependency
    assert outcome.violations == []
    assert [f.path for f in outcome.files] == [
        "pom.xml", "src/test/java/com/demo/pages/LoginPage.java", STEPS, new_steps,
    ]


GENERATED = "src/test/java/com/demo/ngauto"
WITH_A_TEST = "public class {name} {{\n    @Test\n    public void run() {{}}\n}}\n"


def _write_generated_classes(project: Path) -> Callable[[], None]:
    def write() -> None:
        (project / GENERATED).mkdir(parents=True)
        for name in ("TD_001_ValidLogin", "TD_002_LogoutTest", "Runner"):
            (project / GENERATED / f"{name}.java").write_text(WITH_A_TEST.format(name=name), encoding="utf-8")
        (project / GENERATED / "LoginFlow.java").write_text("public class LoginFlow {}\n", encoding="utf-8")  # no test
    return write


def test_a_new_test_class_that_a_plain_maven_run_skips_is_reported_and_kept(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # A real run: 8 of 11 generated classes were not named like tests (*Test), so a plain `mvn test` never ran them.
    project = _project(tmp_path)
    conversation(edit=_write_generated_classes(project))
    events: list[AgentEvent] = []

    outcome = _generate(project, events=events)

    skipped = [f"{GENERATED}/Runner.java", f"{GENERATED}/TD_001_ValidLogin.java"]
    assert [(v.file, v.rule) for v in outcome.violations] == [(path, "not_run_by_default") for path in skipped]
    notices = [e.message for e in events if e.type == RunEventType.AGENT_ERROR]
    assert notices == [REVIEW_NOTICE.format(path=path, rules="not_run_by_default") for path in skipped]
    assert all((project / path).exists() for path in skipped)  # only reported: never renamed


def test_gradle_finds_test_classes_by_their_content_so_their_names_are_not_reported(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    conversation(edit=_write_generated_classes(project))
    gradle = COMMAND.model_copy(update={"tool": "gradle", "command": "gradle test --no-daemon"})
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5)

    outcome = agent.generate(LocalWorkspace(working_dir=str(project)), project, DATA, FACT_SHEET, FINDINGS, gradle)

    assert outcome.violations == []


def test_command_output_left_in_the_project_root_is_reported_and_kept(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # A real run: the agent ran its build as `... | tee result.txt` and left result.txt in the project root.
    project = _project(tmp_path)
    (project / "src/test/resources").mkdir(parents=True)
    (project / "src/test/resources/logback-test.xml").write_text("<file>test.log</file>\n", encoding="utf-8")

    def write() -> None:
        _writes_feature(project)()
        (project / "result.txt").write_text("[INFO] BUILD SUCCESS\n", encoding="utf-8")
        (project / "requirements.txt").write_text("selenium==4.25.0\n", encoding="utf-8")  # a real project change
        (project / "test.log").write_text("12:00 INFO started\n", encoding="utf-8")  # by the project's own logging

    conversation(edit=write)

    outcome = _generate(project)

    assert [(v.file, v.rule) for v in outcome.violations] == [("result.txt", "scratch_file")]
    assert (project / "result.txt").exists()


def _cases(count: int) -> TestDataSetModel:
    return TestDataSetModel(source_format=TestDataFormat.JSON, cases=[
        LOGIN.model_copy(update={"id": f"CASE-{number}", "title": f"Case {number}"}) for number in range(1, count + 1)
    ])


def _batched(project: Path, data: TestDataSetModel, events: list[AgentEvent]) -> Any:
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5, batch_size=2)
    return agent.generate(LocalWorkspace(working_dir=str(project)), project, data, FACT_SHEET, FINDINGS, COMMAND,
                          on_event=events.append)


def test_cases_go_to_the_agent_a_few_at_a_time_in_one_conversation(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    turns: list[int] = []

    def write_one_feature_per_turn() -> None:
        turns.append(1)
        path = project / f"src/test/resources/features/ngauto/part{len(turns)}.feature"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"@ngauto\nFeature: Part {len(turns)}\n", encoding="utf-8")

    created = conversation(edit=write_one_feature_per_turn, answer="Wrote a feature.")
    events: list[AgentEvent] = []

    outcome = _batched(project, _cases(5), events)

    [fake] = created  # one conversation: the agent keeps what it learned about the project
    first, second, third = fake.messages
    assert "CASE-1" in first and "CASE-2" in first and "CASE-3" not in first
    assert "(These are cases 1-2 of 5; the next ones follow once these are written.)" in first
    assert "next test cases (3-4 of 5)" in second and "CASE-3" in second and "CASE-4" in second
    assert "Case CASE-2:" not in second and "How to work" not in second
    # A small model breaks files it edits (str_replace): every later batch goes into new files instead.
    assert "NEW test files" in second and "do not edit the test files you already wrote" in " ".join(second.split())
    # Where the work stands comes from the files, not from the model's memory (the condenser may drop it).
    assert "Where the work stands" in second and "src/test/resources/features/ngauto/part1.feature" in second
    assert "next test cases (5-5 of 5)" in third and "CASE-5" in third
    assert [e.message for e in events if e.type == RunEventType.AGENT_MESSAGE and "Writing tests" in e.message] == [
        "Writing tests for cases 1-2 of 5", "Writing tests for cases 3-4 of 5", "Writing tests for cases 5-5 of 5",
    ]
    assert outcome.generated_cases == ["CASE-1", "CASE-2", "CASE-3", "CASE-4", "CASE-5"]
    assert len(outcome.files) == 3 and outcome.error is None


def test_a_batch_that_writes_nothing_is_nudged_on_its_own(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    turns: list[int] = []

    def write_only_on_the_first_turn() -> None:
        turns.append(1)
        if len(turns) == 1:
            (project / FEATURE).parent.mkdir(parents=True)
            (project / FEATURE).write_text("@ngauto\nFeature: Login\n", encoding="utf-8")

    created = conversation(edit=write_only_on_the_first_turn, answer="Done.")

    _batched(project, _cases(3), [])

    # The second batch changed nothing although the first did: it is told to continue.
    assert created[0].messages[2:] == [NUDGE_MESSAGE] * settings.GENERATOR_NUDGES


def test_cases_after_a_failed_batch_are_reported_as_not_generated(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    holder: list[list[FakeConversation]] = []

    def write_then_time_out() -> None:
        [fake] = holder[0]
        (project / FEATURE).parent.mkdir(parents=True, exist_ok=True)
        (project / FEATURE).write_text(f"@ngauto\nFeature: Login {len(fake.messages)}\n", encoding="utf-8")
        if len(fake.messages) == 2:  # the second batch times out
            fake.error = ConversationRunError("c", TimeoutError("slow"))

    holder.append(conversation(edit=write_then_time_out, answer="Wrote it."))

    outcome = _batched(project, _cases(5), [])

    assert outcome.error == "The test generator stopped: the agent did not finish in time"
    assert outcome.generated_cases == ["CASE-1", "CASE-2"]
    assert [(case.case_id, case.reason) for case in outcome.skipped_cases] == [
        (f"CASE-{number}", "the test generator stopped before it finished this case") for number in (3, 4, 5)
    ]
    assert [f.path for f in outcome.files] == [FEATURE]  # what it wrote is kept


def test_briefing_inlines_the_projects_tests_but_not_the_generated_folder(tmp_path: Path) -> None:
    project = _project(tmp_path)
    generated = project / "src/test/java/tests/ngauto/OldTest.java"
    generated.parent.mkdir(parents=True)
    generated.write_text("package tests.ngauto;\nclass OldTest {}\n", encoding="utf-8")

    briefing = project_test_briefing(project, ["src/test"], "ngauto")

    assert "src/test/java/com/demo/steps/LoginSteps.java" in briefing
    assert "class LoginSteps" in briefing
    assert "OldTest" not in briefing


def test_briefing_keeps_a_base_class_when_many_files_share_the_test_folder(tmp_path: Path) -> None:
    (tmp_path / "src/test/java").mkdir(parents=True)
    (tmp_path / "src/test/java/BaseThing.java").write_text("class BaseThing {}\n", encoding="utf-8")
    for number in range(30):
        (tmp_path / f"src/test/java/Extra{number:02d}.java").write_text("class Extra {}\n", encoding="utf-8")

    briefing = project_test_briefing(tmp_path, ["src/test"], "ngauto")

    assert "src/test/java/BaseThing.java" in briefing


class BuildWorkspace(LocalWorkspace):
    """A sandbox whose build check gives scripted results, in order (the last one repeats)."""

    def __init__(self, project: Path, builds: list[tuple[int, str]]) -> None:
        super().__init__(working_dir=str(project))
        object.__setattr__(self, "_builds", list(builds))
        object.__setattr__(self, "_commands", [])

    def execute_command(self, command: str, cwd: str | None = None, timeout: float = 30.0) -> Any:  # type: ignore[override]
        from types import SimpleNamespace

        self._commands.append(command)
        if command == CHECKED.check:
            exit_code, output = self._builds.pop(0) if len(self._builds) > 1 else self._builds[0]
            return SimpleNamespace(exit_code=exit_code, stdout=output, stderr="", timeout_occurred=False)
        return SimpleNamespace(exit_code=0, stdout="", stderr="", timeout_occurred=False)


CHECKED = COMMAND.model_copy(update={"check": "mvn -B -ntp -q -DskipTests test-compile"})
COMPILE_ERROR = (
    "[INFO] Compiling 5 source files\n"
    "[ERROR] /workspace/project/src/test/java/com/demo/ngauto/CartTest.java:[13,9] cannot find symbol\n"
    "[ERROR]   symbol:   class LoginPage\n"
)


def _generate_checked(project: Path, workspace: BuildWorkspace, events: list[AgentEvent]) -> Any:
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5, fix_rounds=2)
    return agent.generate(workspace, project, DATA, FACT_SHEET, FINDINGS, CHECKED, on_event=events.append)


def _writes_feature(project: Path) -> Callable[[], None]:
    def write() -> None:
        (project / FEATURE).parent.mkdir(parents=True, exist_ok=True)
        (project / FEATURE).write_text("Feature: Login\n", encoding="utf-8")
    return write


def test_build_errors_go_back_to_the_generator_until_the_tests_build(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    # Before: builds. After the batch: a compile error. After the fix: builds.
    workspace = BuildWorkspace(project, [(0, ""), (1, COMPILE_ERROR), (0, "")])
    created = conversation(edit=_writes_feature(project))
    events: list[AgentEvent] = []

    outcome = _generate_checked(project, workspace, events)

    [fake] = created
    assert len(fake.messages) == 2
    assert "that problem is not your task" not in fake.messages[0]  # the project built before
    fix = fake.messages[1]
    assert "The check `mvn -B -ntp -q -DskipTests test-compile` fails with the files you wrote" in fix
    assert "cannot find symbol" in fix and "class LoginPage" in fix
    assert "Where the work stands" in fix and FEATURE in fix
    assert "[INFO] Compiling" not in fix  # only the lines that name the errors
    assert outcome.builds is True
    notices = [e.message for e in events if e.type in (RunEventType.AGENT_ERROR, RunEventType.AGENT_MESSAGE)]
    assert "The tests do not build: asking the generator to fix the errors (1 of 2)" in notices
    assert "The tests build" in notices


def test_a_project_that_did_not_build_before_is_not_checked(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    workspace = BuildWorkspace(project, [(1, COMPILE_ERROR)])  # broken before anything was added
    created = conversation(edit=_writes_feature(project))
    events: list[AgentEvent] = []

    outcome = _generate_checked(project, workspace, events)

    assert len(created[0].messages) == 1  # the project's own errors never go to the generator
    assert outcome.builds is None
    assert any("does not build before any test is added" in e.message for e in events)


def test_the_generator_is_told_that_a_project_which_did_not_build_before_is_not_its_task(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # A real run: on a project that never built, the generator tried to fix that build (sed on pom.xml, downloaded jars)
    # instead of only writing its tests. A later batch must not announce a check that never runs, either.
    project = _project(tmp_path)
    workspace = BuildWorkspace(project, [(1, COMPILE_ERROR)])
    created = conversation(edit=_writes_feature(project))
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5, fix_rounds=2, batch_size=1)

    agent.generate(workspace, project, DATA, FACT_SHEET, FINDINGS, CHECKED)

    first, second = created[0].messages[:2]  # one case per batch
    assert "that problem is not your task" in first
    assert "NG Automate checks your files after each batch" not in first
    assert "Where the work stands" in second and "has not run yet" not in second


def test_tests_that_never_build_are_left_to_the_healer(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    workspace = BuildWorkspace(project, [(0, ""), (1, COMPILE_ERROR)])
    created = conversation(edit=_writes_feature(project))
    events: list[AgentEvent] = []

    outcome = _generate_checked(project, workspace, events)

    assert len(created[0].messages) == 3  # the batch, then 2 fixes
    assert outcome.builds is False
    assert outcome.summary.startswith("The build check still failed at the end")  # before the agent's own words
    assert any("still do not build after 2 fixes" in e.message for e in events)
    assert [f.path for f in outcome.files] == [FEATURE]  # what it wrote is kept for the healer


NODE_MODULE_ERROR = (
    "Error: Cannot find module '../fixtures/test'\n"
    "Require stack:\n"
    "- /workspace/project/tests/ngauto/td-001-valid-login.spec.ts\n"
    "- /workspace/project/node_modules/playwright/lib/transform/transform.js\n"
)


def test_a_build_error_keeps_the_lines_that_name_its_file(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # A real Playwright check: the file with the wrong import is in the require stack, after the error line.
    project = _project(tmp_path)
    workspace = BuildWorkspace(project, [(0, ""), (1, NODE_MODULE_ERROR), (0, "")])
    created = conversation(edit=_writes_feature(project))

    _generate_checked(project, workspace, [])

    fix = created[0].messages[1]
    assert "Cannot find module '../fixtures/test'" in fix
    assert "/workspace/project/tests/ngauto/td-001-valid-login.spec.ts" in fix


def test_a_case_written_twice_is_reported_and_the_next_batch_is_told_to_keep_one(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    turns: list[int] = []

    def write_case_one_again() -> None:
        # The same case under another name in every turn, as a small model did after a condensation.
        turns.append(1)
        path = project / f"src/test/resources/features/ngauto/case_1_v{len(turns)}.feature"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Feature: CASE-1 Case 1\n", encoding="utf-8")

    created = conversation(edit=write_case_one_again, answer="Wrote a feature.")
    events: list[AgentEvent] = []

    _batched(project, _cases(5), events)

    third = created[0].messages[2]
    assert "CASE-1 is in 2 of your files" in third and "keep the best file" in third
    # Nothing follows the last batch, so the copies it made go back once more before the generator finishes.
    finish = created[0].messages[3]
    assert finish.startswith("Before you finish") and "CASE-1 is in 3 of your files" in finish
    assert len(created[0].messages) == 4
    notices = [e.message for e in events if e.type == RunEventType.AGENT_ERROR]
    assert any(notice.startswith("Some test cases are in more than one generated file (CASE-1)") for notice in notices)


def test_feature_files_asked_for_in_a_project_without_bdd_get_the_generic_rules_and_a_check_that_they_run(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    # A real run: on this Maven + TestNG project the generator wrote the feature files into src/test/java (the build
    # copies no feature files from there) and no runner, and the build check passed, so not one scenario ran. Where
    # such files go is the framework's business; the agent must check that its test command finds and runs them.
    project = _project(tmp_path)
    created = conversation(edit=_writes_feature(project), answer="Wrote a feature.")
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5)
    testng = FACT_SHEET.model_copy(update={"bdd_tool": _fact(None), "feature_file_count": 0})

    agent.generate(LocalWorkspace(working_dir=str(project)), project, DATA, testng, FINDINGS, COMMAND,
                   instructions="Write the new tests as Cucumber BDD feature files with step definitions.")

    head, rules = created[0].messages[0].split("How to work", 1)
    assert "Write the new tests as Cucumber BDD feature files" in head  # the user's instructions come first
    assert "Put your new test files only in a new folder named 'ngauto'" in rules
    assert "Every other file you need goes where this framework expects it" in rules
    assert "the project's own test command would find and run your new tests" in rules
    assert "src/test" not in rules and "Class names end with" not in rules  # no Maven recipe


def test_the_users_instructions_come_before_the_rules_in_every_batch(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    created = conversation(edit=_writes_feature(project), answer="Wrote a feature.")
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5, batch_size=2)

    agent.generate(LocalWorkspace(working_dir=str(project)), project, _cases(3), FACT_SHEET, FINDINGS, COMMAND,
                   instructions="Write the new tests as BDD feature files.")

    first, second = created[0].messages[:2]  # then nudges: this fake writes the same file every turn
    assert "The user's instructions for this run (they come first" in first
    assert first.index("Write the new tests as BDD feature files.") < first.index("How to work")
    assert "- Write the new tests as BDD feature files." in second
