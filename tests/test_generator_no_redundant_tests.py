"""No redundant tests, the generator: cases that a test an earlier generation wrote in the 'ngauto' folder already
covers are left out before the agent starts, so it writes only the new ones. A fake conversation plays the agent and a
fake structured call plays the judge (no Docker, no LLM)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openhands.sdk.workspace import LocalWorkspace

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.agents.TestGeneratorAgent.TestGeneratorAgent import NOTHING_NEW_SUMMARY, TestGeneratorAgent
from app.models.llmModel import LlmConfigModel
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestStepModel
from app.models.tokenUsageModel import TokenUsageModel
from tests.test_healer_agent import FakeConversation
from tests.test_test_generator_agent import (  # noqa: F401 - conversation is a fixture
    COMMAND,
    DATA,
    FACT_SHEET,
    FINDINGS,
    LOGIN,
    _project,
    conversation,
)

Conversations = Callable[..., list[FakeConversation]]

# What an earlier generation wrote for LOGIN-1 (types "admin", checks "Dashboard"): its own name, no case id.
EARLIER = "src/test/java/com/demo/ngauto/SignInTest.java"
EARLIER_TEXT = (
    "package com.demo.ngauto;\n\npublic class SignInTest {\n    @Test\n    public void signInShowsTheDashboard() {\n"
    '        login("admin");\n        Assert.assertEquals(heading(), "Dashboard");\n    }\n}\n'
)
# The same login as a story in prose (any knowledge-base format): no values for plain code to compare.
STORY = TestCaseSpecModel(id="TD-001", title="An admin signs in and lands on the dashboard", steps=[
    TestStepModel(action="step", text="The admin signs in"),
    TestStepModel(action="verify_text", text="The dashboard is shown"),
])
NEW = TestCaseSpecModel(id="TD-002", title="A user resets the password", steps=[
    TestStepModel(action="step", text="The user asks for a new password"),
    TestStepModel(action="verify_text", text="A reset mail is sent", expected="Check your inbox"),
])
YES = {"covered": True, "file": EARLIER, "test": "signInShowsTheDashboard", "reason": "same sign-in and dashboard check"}
NO = {"covered": False, "file": "", "test": "", "reason": "it does not reset a password"}


class FakeJudge:
    """Plays complete_json for the judge: an answer per case title (else "no"), or an error for every call."""

    def __init__(self, answers: dict[str, dict[str, Any]] | None = None, error: Exception | None = None) -> None:
        self.answers, self.error = answers or {}, error
        self.cases: list[str] = []

    def __call__(self, llm_config: Any, instructions: str, text: str, schema_name: str, schema: Any, parse: Any,
                 on_usage: Callable[[TokenUsageModel], None] | None = None, **kwargs: Any) -> Any:
        title = text.split("Title: ", 1)[1].split("\n", 1)[0]
        self.cases.append(title)
        if on_usage is not None:
            on_usage(TokenUsageModel(input_tokens=100, output_tokens=10, calls=1))
        if self.error is not None:
            raise self.error
        return parse(json.dumps(self.answers.get(title, NO)))


def _with_earlier_test(root: Path, text: str = EARLIER_TEXT, path: str = EARLIER) -> Path:
    project = _project(root)
    (project / path).parent.mkdir(parents=True, exist_ok=True)
    (project / path).write_text(text, encoding="utf-8")
    return project


def _generate(project: Path, cases: list[TestCaseSpecModel], judge: FakeJudge,
              events: list[AgentEvent] | None = None) -> Any:
    agent = TestGeneratorAgent(LlmConfigModel(model="openai/gpt-4o"), max_iterations=5, complete=judge)
    data = TestDataSetModel(source_format=TestDataFormat.TEXT, cases=cases)
    return agent.generate(LocalWorkspace(working_dir=str(project)), project, data, FACT_SHEET, FINDINGS, COMMAND,
                          on_event=events.append if events is not None else None)


def _write(project: Path, path: str = "src/test/java/com/demo/ngauto/ResetPasswordTest.java") -> Callable[[], None]:
    def write() -> None:
        (project / path).parent.mkdir(parents=True, exist_ok=True)
        (project / path).write_text("public class ResetPasswordTest {\n    @Test public void reset() {}\n}\n",
                                    encoding="utf-8")

    return write


def _messages(events: list[AgentEvent]) -> list[str]:
    return [event.message for event in events]


def test_generating_again_writes_only_the_new_cases(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path)
    created = conversation(edit=_write(project), answer="Wrote ResetPasswordTest.")
    judge = FakeJudge({STORY.title: YES})
    events: list[AgentEvent] = []

    outcome = _generate(project, [STORY, NEW], judge, events)

    prompt = created[0].messages[0]
    assert "Case TD-002: A user resets the password" in prompt and "TD-001" not in prompt  # never shown to the agent
    assert outcome.generated_cases == ["TD-002"]
    [covered] = outcome.skipped_cases
    assert (covered.case_id, covered.covered_by) == ("TD-001", EARLIER)
    assert covered.reason == (f"an earlier generation already tests it: signInShowsTheDashboard in {EARLIER} "
                              "(same sign-in and dashboard check)")
    assert (project / EARLIER).read_text(encoding="utf-8") == EARLIER_TEXT  # the earlier test is untouched
    messages = _messages(events)
    assert ("The 'ngauto' folder already has 1 test files from an earlier generation: checking which of the 2 cases "
            "they already cover") in messages
    assert f"TD-001: no new test, {covered.reason}" in messages


def test_when_every_case_is_already_tested_the_agent_never_starts(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path)
    created = conversation(edit=_write(project))

    outcome = _generate(project, [STORY], FakeJudge({STORY.title: YES}))

    assert created == []  # no conversation, no sandbox work
    assert outcome.summary == NOTHING_NEW_SUMMARY and outcome.files == [] and outcome.generated_cases == []
    assert [case.case_id for case in outcome.skipped_cases] == ["TD-001"]
    assert outcome.token_usage is not None and outcome.token_usage.calls == 1  # the judge's call is counted


def test_the_same_values_need_no_judge(conversation: Conversations, tmp_path: Path) -> None:
    # LOGIN-1 types "admin" and checks "Dashboard": the earlier test holds both, under another name and no id.
    project = _with_earlier_test(tmp_path)
    conversation(edit=_write(project))
    judge = FakeJudge()

    outcome = _generate(project, [LOGIN], judge)

    assert judge.cases == []
    [covered] = outcome.skipped_cases
    assert covered.reason == f"an earlier generation already tests it in {EARLIER} (it has all of the case's values and checks)"


def test_a_similar_but_different_case_is_written(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path, EARLIER_TEXT.replace('"Dashboard"', '"Home"'))
    created = conversation(edit=_write(project), answer="Wrote it.")
    judge = FakeJudge()  # the judge says no

    outcome = _generate(project, [LOGIN], judge)

    assert judge.cases == ["Valid login"]  # not all values match: the judge decides, and says no
    assert "Case LOGIN-1" in created[0].messages[0]
    assert outcome.generated_cases == ["LOGIN-1"] and outcome.skipped_cases == []


def test_an_answer_that_does_not_hold_writes_the_case(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path)
    created = conversation(edit=_write(project), answer="Wrote it.")
    made_up = {**YES, "test": "testThatDoesNotExist"}
    events: list[AgentEvent] = []

    outcome = _generate(project, [STORY], FakeJudge({STORY.title: made_up}), events)

    assert "Case TD-001" in created[0].messages[0]
    assert outcome.generated_cases == ["TD-001"] and outcome.skipped_cases == []
    assert (f"Could not tell whether TD-001 is already tested ({EARLIER} has no test named 'testThatDoesNotExist'): "
            "its test is written") in _messages(events)


def test_a_failing_judge_never_stops_the_generation(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path)
    created = conversation(edit=_write(project), answer="Wrote it.")
    events: list[AgentEvent] = []

    outcome = _generate(project, [STORY, NEW], FakeJudge(error=ValueError("the model call failed (Timeout)")), events)

    assert "Case TD-001" in created[0].messages[0] and "Case TD-002" in created[0].messages[0]
    assert outcome.generated_cases == ["TD-001", "TD-002"] and outcome.error is None
    assert any(m.startswith("Could not tell whether TD-001 is already tested (the check failed") for m in _messages(events))


def test_without_earlier_tests_nothing_is_judged(conversation: Conversations, tmp_path: Path) -> None:
    project = _project(tmp_path)  # a first generation: no 'ngauto' folder yet
    conversation(edit=_write(project), answer="Wrote it.")
    judge = FakeJudge({STORY.title: YES})

    outcome = _generate(project, [STORY], judge)

    assert judge.cases == [] and outcome.generated_cases == ["TD-001"]


def test_the_projects_own_tests_are_never_compared(conversation: Conversations, tmp_path: Path) -> None:
    # Only what NG Automate wrote counts: the user's own SignInTest does the same, and the case is still written.
    project = _with_earlier_test(tmp_path, path="src/test/java/com/demo/tests/SignInTest.java")
    conversation(edit=_write(project), answer="Wrote it.")
    judge = FakeJudge({STORY.title: YES})

    outcome = _generate(project, [STORY, LOGIN], judge)

    assert judge.cases == [] and outcome.generated_cases == ["TD-001", "LOGIN-1"]


def test_a_case_without_steps_is_still_reported_as_before(conversation: Conversations, tmp_path: Path) -> None:
    project = _with_earlier_test(tmp_path)
    empty = TestCaseSpecModel(id="TD-009", title="Nothing yet", steps=[])

    outcome = _generate(project, [STORY, empty], FakeJudge({STORY.title: YES}))

    assert [(case.case_id, case.covered_by) for case in outcome.skipped_cases] == [("TD-001", EARLIER), ("TD-009", None)]


def test_first_generations_are_unchanged(conversation: Conversations, tmp_path: Path) -> None:
    project = _project(tmp_path)
    created = conversation(edit=_write(project), answer="Wrote it.")

    outcome = _generate(project, DATA.cases, FakeJudge())

    assert len(created) == 1 and outcome.generated_cases == ["LOGIN-1", "LOGOUT-1"] and outcome.skipped_cases == []
