"""Test generator agent: prompt, skipped cases, guards and the selector, with a fake conversation (no Docker, no LLM)."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.workspace import LocalWorkspace
from pydantic import SecretStr

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.TestGeneratorAgent.TestGeneratorAgent import (
    GENERATOR_TOOL_NAMES,
    NUDGE_MESSAGE,
    NUDGE_NOTICE,
    TestGeneratorAgent,
)
from app.config import settings
from app.core.exceptions import ErrorMessages, ValidationError
from app.models.analyzerModel import AnalyzerFindingsModel, Confidence, FactModel, FactSheetModel, FactSource, ImportantPathModel
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestStepModel
from app.models.testRunModel import TestCommandModel
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


def test_generated_feature_is_kept_tagged_and_selected(conversation: Callable[..., list[FakeConversation]], tmp_path: Path) -> None:
    project = _project(tmp_path)

    def write_feature() -> None:
        (project / FEATURE).parent.mkdir(parents=True)
        (project / FEATURE).write_text("Feature: Login\n  Scenario: LOGIN-1 Valid login\n", encoding="utf-8")

    created = conversation(edit=write_feature)

    outcome = _generate(project)

    assert outcome.generated_cases == ["LOGIN-1", "LOGOUT-1"]  # LOGOUT-1's locator is assumed, never a reason to drop it
    assert outcome.skipped_cases == []
    assert [(f.path, f.change) for f in outcome.files] == [(FEATURE, "added")]
    assert (project / FEATURE).read_text(encoding="utf-8").startswith("@ngauto\nFeature: Login")
    assert outcome.selector == "@ngauto"
    assert outcome.summary == "Wrote login.feature."
    [fake] = created
    prompt = fake.messages[0]
    assert "Case LOGIN-1: Valid login [tags: smoke]" in prompt
    assert 'type target=id=username value="admin"' in prompt
    assert "Case LOGOUT-1: Logout" in prompt
    assert "NGAUTO-UNVERIFIED" in prompt and "Never leave a case out" in prompt
    assert "Preconditions" in prompt
    assert "src/test/java/com/demo/pages: page objects" in prompt
    assert "@ngauto" in prompt
    assert {tool.name for tool in fake.agent.tools} == set(GENERATOR_TOOL_NAMES)


def test_weakening_an_existing_step_or_deleting_a_file_is_rolled_back(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)

    def misbehave() -> None:
        (project / STEPS).write_text(EXISTING_STEPS.replace("        Assert.assertEquals(driver.getTitle(), expected);\n", ""), encoding="utf-8")
        (project / "pom.xml").unlink()

    conversation(edit=misbehave)

    outcome = _generate(project)

    assert (project / STEPS).read_text(encoding="utf-8") == EXISTING_STEPS
    assert (project / "pom.xml").is_file()
    assert outcome.reverted_files == ["pom.xml", STEPS]
    assert {v.rule for v in outcome.violations} == {"assertion_removed", "existing_code_changed", "file_deleted"}
    assert outcome.files == [] and outcome.selector is None


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


def test_changing_an_existing_locator_is_rolled_back_but_new_code_is_kept(
    conversation: Callable[..., list[FakeConversation]], tmp_path: Path
) -> None:
    project = _project(tmp_path)
    page = project / "src/test/java/com/demo/pages/LoginPage.java"
    page.parent.mkdir(parents=True)
    original = 'public class LoginPage {\n  private final By username = By.id("username");\n}\n'
    page.write_text(original, encoding="utf-8")
    grown = EXISTING_STEPS.replace("}\n}\n", '}\n\n    @When("the user opens {string}")\n    public void open(String url) { driver.get(url); }\n}\n')

    def edit() -> None:
        page.write_text(original.replace('By.id("username")', 'By.id("user-name")'), encoding="utf-8")
        (project / STEPS).write_text(grown, encoding="utf-8")

    conversation(edit=edit)

    outcome = _generate(project)

    assert page.read_text(encoding="utf-8") == original
    assert (project / STEPS).read_text(encoding="utf-8") == grown
    assert outcome.reverted_files == ["src/test/java/com/demo/pages/LoginPage.java"]
    assert [v.rule for v in outcome.violations] == ["existing_code_changed"]
    assert [f.path for f in outcome.files] == [STEPS]


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
    assert "CASE-2" not in second and "Rules:" not in second
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
