"""The analyzer agent: an OpenHands agent that reads a project inside the sandbox and writes its findings.

It can only look: file_editor (view), glob and grep. There is no terminal tool, so it cannot run commands
or reach the network. The project is mounted read-only, so the editor's write commands fail.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.exceptions import ConversationRunError
from openhands.sdk.conversation.impl.remote_conversation import RemoteConversation
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.event.error_classification import classify_error
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from pydantic import ValidationError as PydanticValidationError

from app.agents.AnalyzerAgent.AgentEventMapper import (
    AGENT_TIMED_OUT,
    MAX_ITERATIONS_CODE,
    MAX_MESSAGE_CHARS,
    AgentEvent,
    clip,
    describe_agent_failure,
    map_openhands_event,
)
from app.agents.AnalyzerAgent.AnalyzerPrompts import (
    ANSWER_PLACEHOLDERS,
    build_analysis_prompt,
    build_repair_prompt,
    project_relative_path,
)
from app.config import settings
from app.core.exceptions import AnalysisError, ErrorMessages
from app.models.analyzerModel import AnalyzerFindingsModel, Confidence, FactModel, FactSheetModel, FactSource
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventLevel, RunEventType
from app.utils.LlmInstance import build_llm

logger = logging.getLogger(__name__)

# Look-only tools. Never add the terminal tool: it would let the agent run commands and reach the network.
ANALYZER_TOOL_NAMES = (FileEditorTool.name, GlobTool.name, GrepTool.name)
# Summarize old steps before the history fills this share of the context window. Ollama silently drops
# the start of a prompt that is too long, which would lose the task and the answer format.
_CONTEXT_WINDOW_SHARE = 0.6
_FACT_FIELDS = tuple(
    name for name, info in AnalyzerFindingsModel.model_fields.items() if info.annotation is FactModel
)
_FENCED_JSON = re.compile(r"```json[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT_START = re.compile(r'\{\s*["}]')
# Shorter values are not real secrets, and masking them would mangle ordinary words in the timeline.
_MIN_SECRET_LENGTH = 8
_SECRET_MASK = "***"

# Why an answer could not be used: told to the agent in the repair prompt, and to users through
# ErrorMessages.ANALYSIS_FAILED when the repaired answer is still unusable.
NO_ANSWER = "the agent gave no final answer"
NO_JSON_OBJECT = "no JSON object found in the answer"
JSON_INCOMPLETE = "the JSON object is not complete"
JSON_INVALID = "the JSON is not valid: {problem} (line {line}, column {column})"
JSON_TOO_DEEP = "the JSON is nested too deeply"
NOT_A_JSON_OBJECT = "the answer is not a JSON object"
EXAMPLE_VALUE_COPIED = 'the answer still contains the example value "{value}"'
# Timeline notice before the repair round.
REPAIR_NOTICE = "The answer could not be used ({reason}); asking the agent to correct it"


def parse_findings(text: str) -> AnalyzerFindingsModel:
    """The agent's final answer as findings. ValueError with a short reason when it is not usable.

    Every fact is marked as judged by the LLM, and a fact without evidence is at most "low" confidence.
    """
    data = _extract_json_object(text)
    for name in _FACT_FIELDS:
        if isinstance(data.get(name), dict):
            data[name] = _normalize_fact(data[name])
    if isinstance(data.get("important_paths"), list):
        data["important_paths"] = [_normalize_important_path(entry) for entry in data["important_paths"]]

    try:
        findings = AnalyzerFindingsModel.model_validate(data)
    except PydanticValidationError as error:
        raise ValueError(_describe_validation_error(error)) from None

    copied = sorted(_strings(findings.model_dump(mode="json")) & ANSWER_PLACEHOLDERS)
    if copied:
        raise ValueError(EXAMPLE_VALUE_COPIED.format(value=copied[0]))
    return findings


class AnalyzerAgent:
    def __init__(self, llm_config: LlmConfigModel, max_iterations: int = settings.ANALYZER_MAX_ITERATIONS) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations

    def analyze(
        self,
        workspace: BaseWorkspace,
        fact_sheet: FactSheetModel,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AnalyzerFindingsModel:
        """Blocking: the master runs it in a worker thread. One repair round when the answer is not usable."""
        forwarder = _EventForwarder(on_event, _secret_values(self.llm_config))
        conversation = Conversation(
            agent=self._build_agent(),
            workspace=workspace,
            callbacks=[forwarder],
            max_iteration_per_run=self.max_iterations,
            visualizer=None,
        )
        deadline = time.monotonic() + settings.RUN_TIMEOUT_SECONDS
        try:
            answer = _ask(conversation, build_analysis_prompt(fact_sheet), deadline)
            try:
                return parse_findings(answer)
            except ValueError as error:
                reason = str(error)

            logger.info("Analyzer answer was not usable (%s); asking the agent to correct it", reason)
            forwarder.notify(AgentEvent(
                RunEventType.AGENT_ERROR, RunEventLevel.INFO, clip(REPAIR_NOTICE.format(reason=reason), MAX_MESSAGE_CHARS)
            ))
            answer = _ask(conversation, build_repair_prompt(reason), deadline)
            try:
                return parse_findings(answer)
            except ValueError as error:
                raise AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason=str(error))) from None
        finally:
            _close(conversation)

    def _build_agent(self) -> Agent:
        llm = build_llm(self.llm_config)
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in ANALYZER_TOOL_NAMES],
            condenser=_build_condenser(llm, self.llm_config.num_ctx),
        )


class _EventForwarder:
    """The conversation callback: maps SDK events and hands them to on_event.

    It never raises. The SDK calls its callbacks one after another without a guard, and its own event
    bookkeeping (which the final answer is read from) runs after ours.
    """

    def __init__(self, on_event: Callable[[AgentEvent], None] | None, secrets: tuple[str, ...]) -> None:
        self.on_event = on_event
        self.secrets = secrets
        self._failed = False

    def __call__(self, event: Any) -> None:
        if self.on_event is not None:
            self.notify(map_openhands_event(event))

    def notify(self, event: AgentEvent | None) -> None:
        if event is None or self.on_event is None:
            return
        try:
            self.on_event(_mask_event(event, self.secrets))
        except Exception:
            if not self._failed:
                logger.exception("Analyzer on_event callback failed; the analysis continues")
            self._failed = True


def _ask(conversation: BaseConversation, message: str, deadline: float) -> str:
    """Send one message, let the agent work, return its final answer ("" when it gave none)."""
    conversation.send_message(message)
    try:
        if isinstance(conversation, RemoteConversation):
            # Without a timeout a remote run keeps polling for up to an hour, even after its sandbox is gone.
            conversation.run(timeout=max(deadline - time.monotonic(), 1.0))
        else:
            conversation.run()
    except ConversationRunError as error:
        if not _stopped_at_a_limit(error):
            logger.warning("Analyzer conversation failed", exc_info=True)
            raise AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason=_run_failure_reason(error))) from error
        logger.info("Analyzer agent stopped at a step limit; reading what it answered")
    return get_agent_final_response(conversation.state.events)


def _stopped_at_a_limit(error: ConversationRunError) -> bool:
    """Out of steps or stuck: the agent can still answer with what it found (a remote run raises for these)."""
    if error.conversation_error is not None:
        return error.conversation_error.code == MAX_ITERATIONS_CODE
    return "stuck" in str(error.original_exception).lower()


def _run_failure_reason(error: ConversationRunError) -> str:
    event = error.conversation_error
    if event is not None:
        return describe_agent_failure(event.code, event.classification.kind if event.classification else None)
    original = error.original_exception
    if isinstance(original, TimeoutError):
        return AGENT_TIMED_OUT
    code = type(original).__name__
    return describe_agent_failure(code, classify_error(code, str(original)).kind)


def _close(conversation: BaseConversation) -> None:
    try:
        conversation.close()
    except Exception:
        logger.warning("Could not close the analyzer conversation", exc_info=True)


def _build_condenser(llm: LLM, num_ctx: int | None) -> LLMSummarizingCondenser:
    return LLMSummarizingCondenser(
        llm=llm.model_copy(update={"usage_id": "condenser"}),
        max_tokens=int(num_ctx * _CONTEXT_WINDOW_SHARE) if num_ctx else None,
    )


def _secret_values(config: LlmConfigModel) -> tuple[str, ...]:
    key = config.api_key.get_secret_value() if config.api_key else ""
    return (key,) if len(key) >= _MIN_SECRET_LENGTH else ()


def _mask_event(event: AgentEvent, secrets: tuple[str, ...]) -> AgentEvent:
    """Defense in depth: the API key never reaches the timeline, even if the agent read it somewhere."""
    if not secrets:
        return event
    return AgentEvent(event.type, event.level, _mask(event.message, secrets), _mask(event.data, secrets))


def _mask(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _SECRET_MASK)
        return value
    if isinstance(value, list):
        return [_mask(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _mask(item, secrets) for key, item in value.items()}
    return value


def _extract_json_object(text: str) -> dict[str, Any]:
    """The last ```json block, else the last {...} object of the text."""
    if not text or not text.strip():
        raise ValueError(NO_ANSWER)
    blocks = _FENCED_JSON.findall(text)
    candidate = blocks[-1] if blocks else text
    try:
        try:
            value = json.loads(candidate)
        except ValueError:
            value = _last_json_object(candidate)
    except RecursionError:
        raise ValueError(JSON_TOO_DEEP) from None
    if not isinstance(value, dict):
        raise ValueError(NOT_A_JSON_OBJECT)
    return value


def _last_json_object(text: str) -> Any:
    spans = _object_spans(text)
    if not spans:
        if _OBJECT_START.search(text):
            raise ValueError(JSON_INCOMPLETE)
        raise ValueError(NO_JSON_OBJECT)
    errors: list[json.JSONDecodeError] = []
    for start, end in reversed(spans):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as error:
            errors.append(error)
    last = errors[0]
    raise ValueError(JSON_INVALID.format(problem=last.msg, line=last.lineno, column=last.colno))


def _object_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level {...} in the text, ignoring braces inside JSON strings.

    An object may only start at a "{" followed by '"' or "}", so braces in prose ("{name}") are skipped.
    """
    spans: list[tuple[int, int]] = []
    depth = start = 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if depth == 0:
            if char == "{" and _OBJECT_START.match(text, index):
                depth, start = 1, index
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
    return spans


def _normalize_fact(fact: dict[str, Any]) -> dict[str, Any]:
    fact = {**fact, "source": FactSource.LLM.value}
    evidence = fact.get("evidence")
    if evidence is None:
        evidence = []
    elif isinstance(evidence, str):
        evidence = [evidence]
    if isinstance(evidence, list):
        evidence = [
            project_relative_path(item.strip()) if isinstance(item, str) else item
            for item in evidence
            if not (isinstance(item, str) and not item.strip())
        ]
    fact["evidence"] = evidence

    confidence = fact.get("confidence")
    if isinstance(confidence, str):
        fact["confidence"] = confidence.strip().lower()
    if confidence is None or (fact.get("value") is not None and not evidence):
        fact["confidence"] = Confidence.LOW.value
    return fact


def _normalize_important_path(entry: Any) -> Any:
    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
        return {**entry, "path": project_relative_path(entry["path"].strip())}
    return entry


def _describe_validation_error(error: PydanticValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc']) or 'answer'}: {item['msg']}" for item in error.errors()[:3]
    ]
    more = f" (and {error.error_count() - 3} more problems)" if error.error_count() > 3 else ""
    return "; ".join(problems) + more


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value))
    return set()
