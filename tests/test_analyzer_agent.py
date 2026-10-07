"""Analyzer agent: answer parsing, prompts, event mapping and the agent loop.

The agent loop runs against a fake Conversation (no Docker, no LLM). The opt-in e2e test at the end
runs the real agent in the real sandbox against the default LLM.
"""

import importlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent, ObservationEvent, SystemPromptEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor import FileEditorAction, FileEditorObservation
from openhands.tools.glob import GlobAction, GlobObservation
from openhands.tools.grep import GrepAction, GrepObservation
from pydantic import SecretStr

from app.agents.AnalyzerAgent.AgentEventMapper import (
    AGENT_OUT_OF_STEPS,
    AGENT_TIMED_OUT,
    MAX_MESSAGE_CHARS,
    MAX_OUTPUT_CHARS,
    MODEL_AUTH_FAILED,
    AgentEvent,
    map_openhands_event,
)
from app.agents.AnalyzerAgent.AnalyzerAgent import NO_ANSWER, NO_JSON_OBJECT, AnalyzerAgent, parse_findings
from app.agents.AnalyzerAgent.AnalyzerPrompts import (
    ANSWER_EXAMPLE,
    SANDBOX_PROJECT_DIR,
    build_analysis_prompt,
    build_repair_prompt,
)
from app.config import settings
from app.core.exceptions import AnalysisError, ErrorCode
from app.models.analyzerModel import (
    AnalyzerFindingsModel,
    Confidence,
    FactModel,
    FactSheetModel,
    FactSource,
    ImportantPathModel,
)
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventLevel, RunEventType

# The module, not the class of the same name (a package re-export would shadow it on attribute access).
analyzer_module = importlib.import_module("app.agents.AnalyzerAgent.AnalyzerAgent")

API_KEY = "sk-test-secret-0123456789"


def _answer(**overrides: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "project_summary": {
            "value": "UI tests for the demo shop: login and checkout",
            "evidence": ["/workspace/project/README.md"],
            "confidence": "high",
        },
        "architecture_pattern": {
            "value": "Page Object Model",
            "source": "code",
            "evidence": ["src/main/java/pages/LoginPage.java:12"],
            "confidence": "medium",
        },
        "test_command": {"value": "mvn test", "evidence": ["pom.xml"], "confidence": "high"},
        "reporting_tools": {"value": ["Allure", "ExtentReports"], "evidence": ["pom.xml"], "confidence": "low"},
        "important_paths": [{"path": "/workspace/project/src/main/java/pages", "role": "page objects"}],
        "open_questions": ["Which environment do the tests run against?"],
    }
    answer.update(overrides)
    return answer


def _fenced(data: Any) -> str:
    return f"Here is the profile.\n```json\n{json.dumps(data, indent=2)}\n```\nTell me if you need more."


def _fact(value: str | list[str] | None, evidence: list[str] | None = None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=evidence or [], confidence=Confidence.HIGH)


@pytest.fixture
def fact_sheet() -> FactSheetModel:
    return FactSheetModel(
        total_files=12,
        language_files={"Java": 10, "Gherkin": 2},
        primary_language=_fact("Java", ["src/test/java/tests/LoginTest.java"]),
        build_tool=_fact("Maven", ["pom.xml"]),
        test_frameworks=_fact(["TestNG"], ["pom.xml"]),
        automation_tools=_fact(["Selenium"], ["pom.xml"]),
        bdd_tool=_fact("Cucumber", ["pom.xml"]),
        marker_files=["pom.xml", "testng.xml"],
        test_dirs=["src/test"],
        feature_file_count=2,
        top_level_tree=["README.md", "pom.xml", "src/", "src/main/", "src/test/", "testng.xml"],
    )


# ---------------------------------------------------------------- parse_findings


def test_parse_fenced_block_with_prose_around_it() -> None:
    findings = parse_findings(_fenced(_answer()))

    assert findings.test_command.value == "mvn test"
    assert findings.reporting_tools.value == ["Allure", "ExtentReports"]
    assert findings.open_questions == ["Which environment do the tests run against?"]


def test_parse_takes_the_last_fenced_block() -> None:
    draft = _answer(test_command={"value": "gradle test", "evidence": ["build.gradle"], "confidence": "low"})
    text = f"Draft:\n```json\n{json.dumps(draft)}\n```\nFinal:\n```JSON\n{json.dumps(_answer())}\n```"

    assert parse_findings(text).test_command.value == "mvn test"


def test_parse_bare_object_with_trailing_prose() -> None:
    text = f"Sure. {json.dumps(_answer())}\n\nI hope this helps, {{name}}. Ask me {{anything}}!"

    assert parse_findings(text).architecture_pattern.value == "Page Object Model"


def test_parse_ignores_braces_inside_json_strings() -> None:
    value = 'Runs with "-Denv={env}" } and { braces'
    answer = _answer(test_command={"value": value, "evidence": ["pom.xml"], "confidence": "high"})

    assert parse_findings(f"Answer: {json.dumps(answer)} done").test_command.value == value


def test_parse_fenced_block_with_trailing_text_inside_the_fence() -> None:
    text = f"```json\n{json.dumps(_answer())}\nThat is all.\n```"

    assert parse_findings(text).test_command.value == "mvn test"


def test_parse_forces_source_to_llm() -> None:
    findings = parse_findings(_fenced(_answer()))

    facts = [findings.project_summary, findings.architecture_pattern, findings.test_command, findings.reporting_tools]
    assert all(fact.source == FactSource.LLM for fact in facts)


def test_parse_strips_the_sandbox_project_prefix() -> None:
    answer = _answer(
        test_command={
            "value": "mvn test",
            "evidence": ["/workspace/project/pom.xml:42", "/workspace/project", "/etc/hosts", "testng.xml"],
            "confidence": "high",
        },
        important_paths=[
            {"path": "/workspace/project/src/test/java/steps", "role": "step definitions"},
            {"path": "src/test/resources", "role": "test data"},
        ],
    )

    findings = parse_findings(_fenced(answer))

    assert findings.project_summary.evidence == ["README.md"]
    assert findings.test_command.evidence == ["pom.xml:42", ".", "/etc/hosts", "testng.xml"]
    assert findings.important_paths == [
        ImportantPathModel(path="src/test/java/steps", role="step definitions"),
        ImportantPathModel(path="src/test/resources", role="test data"),
    ]


def test_parse_normalizes_confidence() -> None:
    answer = _answer(
        project_summary={"value": "UI tests", "evidence": ["README.md"], "confidence": " High "},
        architecture_pattern={"value": "Page Object Model", "evidence": ["src/pages/A.java"]},
        test_command={"value": "mvn test", "evidence": [], "confidence": "high"},
        reporting_tools={"value": None, "evidence": [], "confidence": "medium"},
    )

    findings = parse_findings(_fenced(answer))

    assert findings.project_summary.confidence == Confidence.HIGH
    assert findings.architecture_pattern.confidence == Confidence.LOW  # missing -> low
    assert findings.test_command.confidence == Confidence.LOW  # a value without evidence is not proven
    assert findings.reporting_tools.confidence == Confidence.MEDIUM  # "nothing found" needs no file


def test_parse_accepts_a_single_evidence_string() -> None:
    answer = _answer(test_command={"value": "mvn test", "evidence": "/workspace/project/pom.xml", "confidence": "high"})

    assert parse_findings(_fenced(answer)).test_command.evidence == ["pom.xml"]


def test_parse_drops_blank_evidence() -> None:
    answer = _answer(test_command={"value": "mvn test", "evidence": ["", "  ", " pom.xml "], "confidence": "high"})
    unproven = _answer(test_command={"value": "mvn test", "evidence": ["", " "], "confidence": "high"})

    assert parse_findings(_fenced(answer)).test_command.evidence == ["pom.xml"]
    assert parse_findings(_fenced(unproven)).test_command.confidence == Confidence.LOW


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        pytest.param("", "no final answer", id="empty"),
        pytest.param("   \n", "no final answer", id="blank"),
        pytest.param("It is a Maven project with TestNG.", "no JSON object found", id="prose"),
        pytest.param('```json\n{"project_summary": {"value": "x",}}\n```', "JSON is not valid", id="invalid"),
        pytest.param('Here: {"project_summary": {"value": "cut off', "not complete", id="cut-off"),
        pytest.param("```json\n[1, 2, 3]\n```", "not a JSON object", id="array"),
        pytest.param('{"a": ' + "[" * 100_000 + "]" * 100_000 + "}", "nested too deeply", id="deep"),
    ],
)
def test_parse_rejects_unusable_text(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_findings(text)


def test_parse_reports_missing_keys() -> None:
    answer = _answer()
    del answer["test_command"]

    with pytest.raises(ValueError, match="test_command"):
        parse_findings(_fenced(answer))


def test_parse_reports_invalid_values() -> None:
    answer = _answer(test_command={"value": "mvn test", "evidence": ["pom.xml"], "confidence": "certain"})

    with pytest.raises(ValueError, match="test_command.confidence"):
        parse_findings(_fenced(answer))


def test_parse_rejects_the_copied_answer_example() -> None:
    with pytest.raises(ValueError, match="example value"):
        parse_findings(_fenced(ANSWER_EXAMPLE))


# ---------------------------------------------------------------- prompts


def test_answer_example_has_exactly_the_model_keys() -> None:
    assert set(ANSWER_EXAMPLE) == set(AnalyzerFindingsModel.model_fields)
    for name, info in AnalyzerFindingsModel.model_fields.items():
        if info.annotation is FactModel:
            assert set(ANSWER_EXAMPLE[name]) == set(FactModel.model_fields)
    assert set(ANSWER_EXAMPLE["important_paths"][0]) == set(ImportantPathModel.model_fields)
    AnalyzerFindingsModel.model_validate(ANSWER_EXAMPLE)  # the shape itself is valid


def test_analysis_prompt_gives_facts_rules_and_format(fact_sheet: FactSheetModel) -> None:
    prompt = build_analysis_prompt(fact_sheet)

    assert fact_sheet.model_dump_json() in prompt
    assert SANDBOX_PROJECT_DIR in prompt
    assert "READ-ONLY" in prompt
    assert "never contradict" in prompt
    assert '"low"' in prompt and "open question" in prompt
    assert "```json" in prompt
    for key in AnalyzerFindingsModel.model_fields:
        assert f'"{key}"' in prompt


def test_repair_prompt_names_the_problem_and_repeats_the_format() -> None:
    prompt = build_repair_prompt("test_command: Field required")

    assert "test_command: Field required" in prompt
    assert "```json" in prompt
    assert '"open_questions"' in prompt


# ---------------------------------------------------------------- event mapper


def _action_event(tool_name: str, action: Any, *, arguments: str | None = None, thought: str = "") -> ActionEvent:
    if arguments is None:
        arguments = json.dumps(action.model_dump(mode="json", exclude_none=True)) if action else "{}"
    return ActionEvent(
        thought=[TextContent(text=thought)] if thought else [],
        action=action,
        tool_name=tool_name,
        tool_call_id="call-1",
        tool_call=MessageToolCall(id="call-1", name=tool_name, arguments=arguments, origin="completion"),
        llm_response_id="response-1",
    )


def _observation_event(tool_name: str, observation: Any) -> ObservationEvent:
    return ObservationEvent(observation=observation, action_id="action-1", tool_name=tool_name, tool_call_id="call-1")


def _agent_message(text: str) -> MessageEvent:
    return MessageEvent(source="agent", llm_message=Message(role="assistant", content=[TextContent(text=text)]))


def test_glob_action_is_a_short_info_event() -> None:
    event = map_openhands_event(_action_event("glob", GlobAction(pattern="**/pom.xml")))

    assert event == AgentEvent(
        RunEventType.AGENT_ACTION, RunEventLevel.INFO, "glob **/pom.xml", {"tool": "glob", "args": {"pattern": "**/pom.xml"}}
    )


def test_grep_action_shows_pattern_folder_and_filter() -> None:
    action = GrepAction(pattern="@Test", path="/workspace/project/src/test", include="*.java")

    event = map_openhands_event(_action_event("grep", action, thought="Find the test classes."))

    assert event is not None
    assert event.message == 'grep "@Test" in src/test (*.java)'
    assert event.data["args"] == {"pattern": "@Test", "path": "/workspace/project/src/test", "include": "*.java"}
    assert event.data["thought"] == "Find the test classes."


def test_view_action_shows_the_relative_path_and_lines() -> None:
    action = FileEditorAction(command="view", path="/workspace/project/src/test/LoginTest.java", view_range=[1, 80])

    event = map_openhands_event(_action_event("file_editor", action))

    assert event is not None
    assert event.message == "view src/test/LoginTest.java lines 1-80"
    assert event.data["tool"] == "file_editor"


def test_action_that_could_not_be_parsed_still_maps() -> None:
    bad = map_openhands_event(_action_event("glob", None, arguments='{"pattern": '))
    unknown = map_openhands_event(_action_event("glob", None, arguments='{"pattern": "*.py", "extra": 1}'))

    assert bad is not None and bad.message == "glob" and bad.data["args"] == {"arguments": '{"pattern": '}
    assert unknown is not None and unknown.message == "glob *.py"


def test_finish_action_maps_with_a_clipped_message() -> None:
    event = map_openhands_event(_action_event("finish", FinishAction(message="x" * 5000)))

    assert event is not None
    assert event.message == "finish"
    assert len(event.data["args"]["message"]) == MAX_MESSAGE_CHARS


def test_observation_is_a_detail_event_with_output() -> None:
    observation = GlobObservation.from_text(
        "/workspace/project/pom.xml\n/workspace/project/module/pom.xml",
        files=["/workspace/project/pom.xml", "/workspace/project/module/pom.xml"],
        pattern="**/pom.xml",
        search_path="/workspace/project",
    )

    event = map_openhands_event(_observation_event("glob", observation))

    assert event is not None
    assert (event.type, event.level) == (RunEventType.AGENT_OBSERVATION, RunEventLevel.DETAIL)
    assert event.message == "glob found 2 file(s)"
    assert event.data["tool"] == "glob"
    assert "module/pom.xml" in event.data["output"]


def test_grep_observation_counts_matching_files() -> None:
    observation = GrepObservation.from_text(
        "a.java", matches=["a.java"], pattern="@Test", search_path="/workspace/project", truncated=True
    )

    event = map_openhands_event(_observation_event("grep", observation))

    assert event is not None and event.message == "grep matched 1 file(s) (truncated)"


def test_long_observation_output_is_truncated() -> None:
    observation = FileEditorObservation.from_text("x" * 50_000, command="view", path="/workspace/project/big.txt")

    event = map_openhands_event(_observation_event("file_editor", observation))

    assert event is not None
    assert len(event.data["output"]) == MAX_OUTPUT_CHARS
    assert event.data["output"].endswith("…")
    assert event.message == "file_editor result"


def test_failed_observation_is_marked() -> None:
    observation = FileEditorObservation.from_text(
        "Read-only file system", is_error=True, command="create", path="/workspace/project/x.txt"
    )

    event = map_openhands_event(_observation_event("file_editor", observation))

    assert event is not None
    assert event.message == "file_editor failed"
    assert event.data["error"] is True


def test_agent_message_is_info_and_truncated() -> None:
    event = map_openhands_event(_agent_message("a" * 5000))

    assert event is not None
    assert (event.type, event.level) == (RunEventType.AGENT_MESSAGE, RunEventLevel.INFO)
    assert len(event.message) == MAX_MESSAGE_CHARS


def test_prompts_and_system_prompt_are_never_shown() -> None:
    user = MessageEvent(source="user", llm_message=Message(role="user", content=[TextContent(text="our prompt")]))
    system = SystemPromptEvent(system_prompt=TextContent(text="You are OpenHands agent..."), tools=[])

    assert map_openhands_event(user) is None
    assert map_openhands_event(system) is None
    assert map_openhands_event(_agent_message("   ")) is None


def test_agent_error_event() -> None:
    error = AgentErrorEvent(error="Tool 'terminal' not found.\nAvailable: glob", tool_name="terminal", tool_call_id="c1")

    event = map_openhands_event(error)

    assert event == AgentEvent(
        RunEventType.AGENT_ERROR,
        RunEventLevel.INFO,
        "terminal error: Tool 'terminal' not found.",
        {"tool": "terminal", "error": "Tool 'terminal' not found.\nAvailable: glob"},
    )


def test_conversation_error_shows_a_safe_reason_only() -> None:
    error = ConversationErrorEvent(
        source="environment", code="LLMAuthenticationError", detail=f"key {API_KEY} rejected by http://10.0.0.5"
    )

    event = map_openhands_event(error)

    assert event is not None
    assert (event.type, event.level) == (RunEventType.AGENT_ERROR, RunEventLevel.INFO)
    assert event.message == MODEL_AUTH_FAILED[0].upper() + MODEL_AUTH_FAILED[1:]
    assert event.data == {"code": "LLMAuthenticationError", "kind": "auth"}
    assert API_KEY not in repr(event) and "10.0.0.5" not in repr(event)


def test_step_limit_error_says_so() -> None:
    error = ConversationErrorEvent(source="environment", code="MaxIterationsReached", detail="limit (40)")

    event = map_openhands_event(error)

    assert event is not None and event.message.lower() == AGENT_OUT_OF_STEPS


@pytest.mark.parametrize(
    "event",
    [
        None,
        42,
        "text",
        object(),
        ActionEvent.model_construct(),
        ObservationEvent.model_construct(),
        MessageEvent.model_construct(source="agent"),
        AgentErrorEvent.model_construct(),
    ],
)
def test_mapper_never_raises(event: Any) -> None:
    assert map_openhands_event(event) is None


# ---------------------------------------------------------------- analyze() with a fake conversation


@dataclass
class Reply:
    """What one fake run() does: emit events, end with the agent's text (if any), then raise (if set)."""

    text: str | None = None
    error: BaseException | None = None
    events: list[Any] = field(default_factory=list)


class FakeConversation:
    """Stands in for openhands' Conversation. Like the SDK, it calls the callbacks without a guard and
    records an event only after the callbacks ran."""

    def __init__(self, replies: list[Reply | str | BaseException], agent: Any, **kwargs: Any) -> None:
        self.replies = list(replies)
        self.agent = agent
        self.kwargs = kwargs
        self.callbacks: list[Callable[[Any], None]] = list(kwargs["callbacks"])
        self.state = SimpleNamespace(events=[])
        self.messages: list[str] = []
        self.closed = False

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def run(self) -> None:
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, str):
            reply = Reply(text=reply)
        events = reply.events or [
            _action_event("glob", GlobAction(pattern="**/pom.xml")),
            _observation_event(
                "glob", GlobObservation.from_text("pom.xml", files=["pom.xml"], pattern="**/pom.xml", search_path=".")
            ),
        ]
        if reply.text is not None:
            events = [*events, _agent_message(reply.text)]
        for event in events:
            for callback in [*self.callbacks, self.state.events.append]:
                callback(event)
        if reply.error is not None:
            raise reply.error

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def conversations(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[FakeConversation]]:
    """Install the fake Conversation; returns the list of conversations it creates."""

    def install(*replies: Reply | str | BaseException) -> list[FakeConversation]:
        created: list[FakeConversation] = []

        def factory(agent: Any, **kwargs: Any) -> FakeConversation:
            created.append(FakeConversation(list(replies), agent, **kwargs))
            return created[-1]

        monkeypatch.setattr(analyzer_module, "Conversation", factory)
        return created

    return install


def _llm_config(**overrides: Any) -> LlmConfigModel:
    return LlmConfigModel(**{"model": "openai/gpt-4o", "api_key": SecretStr(API_KEY), **overrides})


def _run_error(code: str, detail: str = "details") -> ConversationRunError:
    event = ConversationErrorEvent(source="environment", code=code, detail=detail)
    return ConversationRunError("conversation-1", RuntimeError(f"{code}: {detail}"), conversation_error=event)


def test_analyze_returns_findings_on_the_first_try(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(_fenced(_answer()))
    workspace = LocalWorkspace(working_dir=str(tmp_path))

    findings = AnalyzerAgent(_llm_config(), max_iterations=7).analyze(workspace, fact_sheet)

    assert findings.test_command.value == "mvn test"
    [conversation] = created
    assert conversation.messages == [build_analysis_prompt(fact_sheet)]
    assert conversation.kwargs["workspace"] is workspace
    assert conversation.kwargs["max_iteration_per_run"] == 7
    assert conversation.kwargs["visualizer"] is None
    assert conversation.closed


def test_analyzer_tools_are_read_only(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(_fenced(_answer()))

    AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    agent = created[0].agent
    assert [tool.name for tool in agent.tools] == ["file_editor", "glob", "grep"]
    assert not any("terminal" in name.lower() for name in agent.include_default_tools)
    assert agent.mcp_config == {}


def test_condenser_keeps_the_history_inside_the_context_window(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(_fenced(_answer()), _fenced(_answer()))
    workspace = LocalWorkspace(working_dir=str(tmp_path))

    AnalyzerAgent(_llm_config(model="ollama_chat/devstral", num_ctx=32768)).analyze(workspace, fact_sheet)
    AnalyzerAgent(_llm_config()).analyze(workspace, fact_sheet)

    ollama, api = created[0].agent.condenser, created[1].agent.condenser
    assert ollama.max_tokens == int(32768 * 0.6)
    assert ollama.llm.usage_id == "condenser" and ollama.llm.model == "ollama_chat/devstral"
    assert api.max_tokens is None  # the SDK uses the model's known context window


def test_invalid_answer_gets_one_repair_round(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations("It is a Maven project with TestNG.", _fenced(_answer()))
    events: list[AgentEvent] = []

    findings = AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet, events.append)

    assert findings.test_command.value == "mvn test"
    assert created[0].messages == [
        build_analysis_prompt(fact_sheet),
        build_repair_prompt(NO_JSON_OBJECT),
    ]
    notices = [event for event in events if event.type == RunEventType.AGENT_ERROR]
    assert len(notices) == 1 and "asking the agent to correct it" in notices[0].message


def test_second_invalid_answer_fails_the_analysis(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    missing_key = _answer()
    del missing_key["reporting_tools"]
    created = conversations("No JSON here.", _fenced(missing_key))

    with pytest.raises(AnalysisError) as raised:
        AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    assert raised.value.error_code == ErrorCode.ANALYSIS_FAILED
    assert raised.value.message.startswith("The analyzer could not produce a valid profile: ")
    assert "reporting_tools" in raised.value.message
    assert len(created[0].messages) == 2
    assert created[0].closed


def test_on_event_receives_mapped_events_in_order(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    conversations(_fenced(_answer()))
    events: list[AgentEvent] = []

    AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet, events.append)

    assert [event.type for event in events] == [
        RunEventType.AGENT_ACTION,
        RunEventType.AGENT_OBSERVATION,
        RunEventType.AGENT_MESSAGE,
    ]
    assert events[0].message == "glob **/pom.xml"


def test_raising_on_event_does_not_break_the_run(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(_fenced(_answer()))
    calls: list[AgentEvent] = []

    def broken(event: AgentEvent) -> None:
        calls.append(event)
        raise RuntimeError("database is down")

    findings = AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet, broken)

    assert findings.test_command.value == "mvn test"
    assert len(calls) == 3
    assert len(created[0].state.events) == 3  # the SDK's own bookkeeping still saw every event


def test_api_key_never_reaches_on_event(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    leaked = FileEditorObservation.from_text(f"OPENAI_API_KEY={API_KEY}", command="view", path="/workspace/project/.env")
    conversations(Reply(text=_fenced(_answer()), events=[_observation_event("file_editor", leaked)]))
    events: list[AgentEvent] = []

    AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet, events.append)

    assert events[0].data["output"] == "OPENAI_API_KEY=***"
    assert all(API_KEY not in repr(event) for event in events)


def test_model_failure_becomes_a_safe_analysis_error(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(_run_error("LLMAuthenticationError", f"Incorrect API key provided: {API_KEY}"))

    with pytest.raises(AnalysisError) as raised:
        AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    assert raised.value.message.endswith(MODEL_AUTH_FAILED)
    assert API_KEY not in raised.value.message
    assert len(created[0].messages) == 1  # no repair round for a broken model
    assert created[0].closed


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ConversationRunError("c1", TimeoutError("Run timed out after 3600 seconds")), AGENT_TIMED_OUT),
        (ConversationRunError("c1", RuntimeError("Connection error: cannot connect to host")), "could not be reached"),
        (_run_error("ZeroDivisionError", "division by zero"), "unexpected error"),
    ],
)
def test_run_failures_have_clear_reasons(
    conversations: Callable[..., list[FakeConversation]],
    fact_sheet: FactSheetModel,
    tmp_path: Path,
    error: ConversationRunError,
    reason: str,
) -> None:
    conversations(error)

    with pytest.raises(AnalysisError, match=reason):
        AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)


@pytest.mark.parametrize(
    "stop",
    [
        _run_error("MaxIterationsReached", "Agent reached maximum iterations limit (40)."),
        ConversationRunError("c1", RuntimeError("Remote conversation got stuck")),
    ],
)
def test_agent_stopped_at_a_limit_gets_a_repair_round(
    conversations: Callable[..., list[FakeConversation]],
    fact_sheet: FactSheetModel,
    tmp_path: Path,
    stop: ConversationRunError,
) -> None:
    created = conversations(Reply(error=stop), _fenced(_answer()))

    findings = AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    assert findings.test_command.value == "mvn test"
    assert created[0].messages[1] == build_repair_prompt(NO_ANSWER)


def test_unexpected_errors_propagate_and_the_conversation_is_closed(
    conversations: Callable[..., list[FakeConversation]], fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    created = conversations(OSError("socket closed"))

    with pytest.raises(OSError, match="socket closed"):
        AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    assert created[0].closed


class FakeRemoteConversation(RemoteConversation):
    """A RemoteConversation without a server: records the timeout each run() gets."""

    def __init__(self, answer: str) -> None:  # no super().__init__: nothing to connect to
        self.answer = answer
        self.timeouts: list[float] = []
        self.fake_state = SimpleNamespace(events=[])
        self.closed = False

    @property
    def state(self) -> Any:  # type: ignore[override]
        return self.fake_state

    def send_message(self, message: Any, sender: str | None = None) -> None:
        pass

    def run(self, blocking: bool = True, poll_interval: float = 1.0, timeout: float = 3600.0) -> None:
        self.timeouts.append(timeout)
        self.fake_state.events.append(_agent_message(self.answer))

    def close(self) -> None:
        self.closed = True


def test_remote_runs_are_bounded_by_the_run_timeout(
    monkeypatch: pytest.MonkeyPatch, fact_sheet: FactSheetModel, tmp_path: Path
) -> None:
    remote = FakeRemoteConversation(_fenced(_answer()))
    monkeypatch.setattr(analyzer_module, "Conversation", lambda agent, **kwargs: remote)
    monkeypatch.setattr(settings, "RUN_TIMEOUT_SECONDS", 120)

    AnalyzerAgent(_llm_config()).analyze(LocalWorkspace(working_dir=str(tmp_path)), fact_sheet)

    [timeout] = remote.timeouts
    assert 100 < timeout <= 120
    assert remote.closed


# ---------------------------------------------------------------- opt-in: real sandbox + real LLM


def _write(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_prompt_and_sandbox_agree_on_the_project_folder() -> None:
    sandbox_module = importlib.import_module("app.services.sandboxService.SandboxService")

    assert getattr(sandbox_module, "SANDBOX_PROJECT_DIR", SANDBOX_PROJECT_DIR) == SANDBOX_PROJECT_DIR


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("NGAUTOMATE_E2E_TESTS") != "1", reason="needs Docker and the LLM: set NGAUTOMATE_E2E_TESTS=1")
def test_analyze_a_real_project_in_the_sandbox(tmp_path: Path) -> None:
    from dotenv import dotenv_values

    from app.services.sandboxService import SandboxService

    # conftest replaces LLM_MODEL with a fake name: read the real default model from .env.
    env = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    llm_config = LlmConfigModel(
        model=env["LLM_MODEL"],
        base_url=env.get("LLM_BASE_URL") or None,
        api_key=SecretStr(env["LLM_API_KEY"]) if env.get("LLM_API_KEY") else None,
        api_mode=env.get("LLM_API_MODE") or "auto",
        num_ctx=int(env["LLM_NUM_CTX"]) if env.get("LLM_NUM_CTX") else None,
        thinking=(env.get("LLM_THINKING") or "true").lower() == "true",
    )
    project = tmp_path / "project"
    _write(project, {
        "README.md": "# Demo shop UI tests\nSelenium + TestNG tests for the demo shop login.\nRun: mvn clean test\n",
        "pom.xml": (
            "<project><artifactId>shop-tests</artifactId><dependencies>"
            "<dependency><groupId>org.seleniumhq.selenium</groupId><artifactId>selenium-java</artifactId></dependency>"
            "<dependency><groupId>org.testng</groupId><artifactId>testng</artifactId></dependency>"
            "<dependency><groupId>io.qameta.allure</groupId><artifactId>allure-testng</artifactId></dependency>"
            "</dependencies></project>"
        ),
        "testng.xml": '<suite name="shop"><test name="login"><classes><class name="tests.LoginTest"/></classes></test></suite>',
        "src/main/java/pages/LoginPage.java": (
            "package pages;\npublic class LoginPage {\n  public void login(String user, String password) {}\n}\n"
        ),
        "src/test/java/tests/LoginTest.java": (
            "package tests;\nimport org.testng.annotations.Test;\nimport pages.LoginPage;\n"
            "public class LoginTest {\n  @Test public void validLogin() { new LoginPage().login(\"a\", \"b\"); }\n}\n"
        ),
    })
    fact_sheet = FactSheetModel(
        total_files=5,
        language_files={"Java": 2},
        primary_language=_fact("Java", ["src/test/java/tests/LoginTest.java"]),
        build_tool=_fact("Maven", ["pom.xml"]),
        test_frameworks=_fact(["TestNG"], ["pom.xml", "testng.xml"]),
        automation_tools=_fact(["Selenium"], ["pom.xml"]),
        bdd_tool=FactModel(value=None, source=FactSource.CODE, evidence=[], confidence=Confidence.LOW),
        marker_files=["pom.xml", "testng.xml"],
        test_dirs=["src/test"],
        feature_file_count=0,
        top_level_tree=["README.md", "pom.xml", "src/", "src/main/", "src/test/", "testng.xml"],
    )
    events: list[AgentEvent] = []

    with SandboxService().open_readonly(project) as session:
        findings = AnalyzerAgent(llm_config).analyze(session.workspace, fact_sheet, events.append)
    print(findings.model_dump_json(indent=2))
    print("\n".join(f"{event.type.value}: {event.message}" for event in events))

    assert findings.test_command.value
    assert all(fact.source == FactSource.LLM for fact in (findings.project_summary, findings.test_command))
    assert any(event.type == RunEventType.AGENT_ACTION for event in events)
    assert not any(SANDBOX_PROJECT_DIR in path for path in findings.test_command.evidence)
