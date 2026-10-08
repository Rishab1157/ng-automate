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

from openhands.sdk import Agent, Conversation, Tool
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from pydantic import ValidationError as PydanticValidationError

from app.agents.AnalyzerAgent.AgentEventMapper import (
    MAX_MESSAGE_CHARS,
    AgentEvent,
    clip,
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
from app.utils.AgentConversationUtils import (
    AgentControl,
    EventForwarder,
    ask,
    build_condenser,
    close_conversation,
    secret_values,
)
from app.utils.LlmInstance import build_llm
from app.utils.StructuredOutput import complete_json

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
REPAIR_NOTICE = "The answer could not be used ({reason}); asking the agent to correct it (attempt {attempt} of {attempts})"
FORMATTER_NOTICE = "The agent could not produce valid JSON ({reason}); formatting its answer with strict JSON mode"
FORMATTER_INSTRUCTIONS = (
    "Convert the analysis below into one JSON object that matches the required schema. Keep the meaning and the "
    "cited file paths exactly; do not invent values or evidence. Use null for unknown values, an empty evidence list "
    "and confidence \"low\" for answers without evidence."
)


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
    def __init__(
        self,
        llm_config: LlmConfigModel,
        max_iterations: int = settings.ANALYZER_MAX_ITERATIONS,
        repair_attempts: int = settings.ANALYZER_REPAIR_ATTEMPTS,
        formatter: Callable[[str], AnalyzerFindingsModel] | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.repair_attempts = max(repair_attempts, 0)
        # Last resort for answers the agent could not fix; injectable for tests.
        self.formatter = formatter or self._format_with_schema

    def analyze(
        self,
        workspace: BaseWorkspace,
        fact_sheet: FactSheetModel,
        on_event: Callable[[AgentEvent], None] | None = None,
        control: AgentControl | None = None,
    ) -> AnalyzerFindingsModel:
        """Blocking: the master runs it in a worker thread.

        An answer is strictly validated. If it is not usable, the agent corrects it (up to `repair_attempts` rounds,
        each told the exact error), then a schema-constrained formatter is the last resort.
        """
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
        deadline = time.monotonic() + settings.RUN_TIMEOUT_SECONDS
        try:
            answer = ask(conversation, build_analysis_prompt(fact_sheet), deadline, _analysis_failed, control)
            reason = _usable_or_reason(answer)
            if reason is None:
                return parse_findings(answer)

            # Layer 2: the agent corrects its own answer, told the exact validation error each round.
            for attempt in range(1, self.repair_attempts + 1):
                logger.info("Analyzer answer not usable (%s); repair attempt %d", reason, attempt)
                forwarder.notify(_notice(REPAIR_NOTICE.format(reason=reason, attempt=attempt, attempts=self.repair_attempts)))
                answer = ask(conversation, build_repair_prompt(reason), deadline, _analysis_failed, control)
                reason = _usable_or_reason(answer)
                if reason is None:
                    return parse_findings(answer)

            # Layer 3: a schema-constrained LLM call turns the last answer into JSON of the right shape.
            forwarder.notify(_notice(FORMATTER_NOTICE.format(reason=reason)))
            try:
                return self.formatter(answer)
            except ValueError as error:
                raise AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason=str(error))) from None
        finally:
            close_conversation(conversation)

    def _format_with_schema(self, answer: str) -> AnalyzerFindingsModel:
        return complete_json(
            self.llm_config,
            FORMATTER_INSTRUCTIONS,
            answer or "(the agent gave no answer)",
            "analyzer_findings",
            AnalyzerFindingsModel.model_json_schema(),
            parse_findings,
        )

    def _build_agent(self) -> Agent:
        llm = build_llm(self.llm_config)
        return Agent(
            llm=llm,
            tools=[Tool(name=name) for name in ANALYZER_TOOL_NAMES],
            condenser=build_condenser(llm, self.llm_config.num_ctx),
        )


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


def _usable_or_reason(answer: str) -> str | None:
    """None when the answer validates, else the short reason it does not."""
    try:
        parse_findings(answer)
    except ValueError as error:
        return str(error)
    return None


def _notice(message: str) -> AgentEvent:
    return AgentEvent(RunEventType.AGENT_ERROR, RunEventLevel.INFO, clip(message, MAX_MESSAGE_CHARS))


def _analysis_failed(reason: str) -> AnalysisError:
    return AnalysisError(ErrorMessages.ANALYSIS_FAILED.format(reason=reason))
