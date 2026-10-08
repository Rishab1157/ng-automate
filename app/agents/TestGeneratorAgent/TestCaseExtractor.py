"""Turn a free-form test-data source (plain text, or JSON in any shape) into test cases with an LLM.

The source is split into numbered blocks (paragraphs of text, or the records of a JSON list) and read a few blocks
at a time with a strict JSON schema. Every case must name the blocks it came from, and every URL, value, expected
text and locator it carries must appear in those blocks: anything else is dropped and reported, never invented.
A step that names its element only in words keeps that as `target_hint`; the test generator finds the real locator
later (the project's page objects, or the live page), or leaves the case out.

Blocking (LLM calls): call from a worker thread.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestStepModel
from app.utils.StructuredOutput import complete_json
from app.utils.TestDataParser import NAVIGATION_ACTIONS, build_test_data_set

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 5000
MAX_CASES_PER_CHUNK = 12
MAX_STEPS_PER_CASE = 30
ACTIONS = [
    "open", "click", "tap", "type", "select", "check", "uncheck", "hover", "upload", "swipe", "scroll",
    "verify_text", "verify_visible", "verify_value", "verify_url", "wait", "other",
]

NO_CASES_FOUND = "no test cases were found in the source"
PART_FAILED = "part {part} of the source could not be read: {reason}"
UNGROUNDED = "case {case} step {step}: dropped {field} '{value}' (it is not in the source)"
NO_CHECK = "case {case}: the source states an expected result but the case has no verify step"
# Words that announce an expected result in manual test cases and acceptance criteria.
_EXPECTATION = re.compile(r"\b(expected|expect|then|should|must|verify|displayed|is shown)\b", re.IGNORECASE)
GROUNDED_FIELDS = ("url", "value", "expected", "locator")

INSTRUCTIONS = (
    "You turn test documentation into automated-test cases. The source is split into blocks marked [B1], [B2], ... "
    "Return one case per test case, scenario or acceptance criterion; list the block ids each case comes from. "
    "Keep the source's wording in each step's 'text'. Copy ids, URLs, values and expected texts exactly as written; "
    "use null when the source does not give one. 'target_hint' names the element in words (e.g. 'Login button'). "
    "'locator' must be null unless the source contains a real locator (id=..., css=..., an XPath, or #id). "
    "Every expected result (Expected:, Then ..., ... should ..., ... is displayed) MUST become its own verify step "
    "at the end: verify_text with 'expected' set to the exact expected text, verify_visible for an element or page "
    "that must appear, verify_url for an address. Never drop an expected result. "
    "Never invent URLs, locators, values or steps. List blocks that hold no test case in not_test_cases."
)


@dataclass(frozen=True)
class SourceBlock:
    id: str  # B1, B2, ...
    ref: str  # "lines 3-9" or "item 2"
    text: str


Complete = Callable[..., Any]


class TestCaseExtractor:
    __test__ = False  # not a pytest test class

    def __init__(
        self,
        llm_config: LlmConfigModel,
        complete: Complete = complete_json,
        timeout_seconds: int = settings.TEST_DATA_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.llm_config = llm_config
        self.complete = complete
        self.timeout_seconds = timeout_seconds

    def extract(self, text: str, source_format: TestDataFormat) -> TestDataSetModel:
        """Raises ValueError when no test case could be read at all."""
        blocks = split_blocks(text, source_format)
        raw_cases: list[dict[str, Any]] = []
        warnings: list[str] = []
        failures: list[str] = []
        for part, chunk in enumerate(chunk_blocks(blocks), start=1):
            try:
                reply = self.complete(
                    self.llm_config,
                    INSTRUCTIONS,
                    "\n\n".join(f"[{block.id}] ({block.ref})\n{block.text}" for block in chunk),
                    "test_cases",
                    _schema([block.id for block in chunk]),
                    _parse_reply,
                    timeout_seconds=self.timeout_seconds,
                )
            except ValueError as error:
                logger.warning("Could not read part %d of a test-data source: %s", part, error)
                failures.append(PART_FAILED.format(part=part, reason=str(error)[:200]))
                continue
            texts = {block.id: block for block in chunk}
            raw_cases.extend(_grounded_case(case, texts, warnings) for case in reply["cases"])

        cases = _unique(_dedupe([case for case in raw_cases if case["steps"]]))
        if not cases:
            raise ValueError(failures[0] if failures else NO_CASES_FOUND)
        data_set = build_test_data_set(cases, source_format)
        return data_set.model_copy(update={"warnings": [*failures, *warnings, *data_set.warnings]})


def split_blocks(text: str, source_format: TestDataFormat) -> list[SourceBlock]:
    if source_format == TestDataFormat.JSON:
        return _json_blocks(text)
    blocks: list[SourceBlock] = []
    start: int | None = None
    lines: list[str] = []

    def flush() -> None:
        if lines and start is not None:
            for piece_start, piece_lines in _pieces(start, lines):
                piece_end = piece_start + len(piece_lines) - 1
                ref = f"line {piece_start}" if piece_start == piece_end else f"lines {piece_start}-{piece_end}"
                blocks.append(SourceBlock(f"B{len(blocks) + 1}", ref, "\n".join(piece_lines)))

    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if start is None:
                start = number
            lines.append(line)
        else:
            flush()
            start, lines = None, []
    flush()
    return blocks


def chunk_blocks(blocks: list[SourceBlock]) -> list[list[SourceBlock]]:
    chunks: list[list[SourceBlock]] = []
    size = 0
    for block in blocks:
        if chunks and size + len(block.text) <= MAX_CHUNK_CHARS:
            chunks[-1].append(block)
            size += len(block.text)
        else:
            chunks.append([block])
            size = len(block.text)
    return chunks


def _pieces(start: int, lines: list[str]) -> list[tuple[int, list[str]]]:
    """A paragraph longer than a chunk, cut at line ends."""
    pieces: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_start = start
    size = 0
    for offset, line in enumerate(lines):
        if current and size + len(line) > MAX_CHUNK_CHARS:
            pieces.append((current_start, current))
            current, current_start, size = [], start + offset, 0
        current.append(line[:MAX_CHUNK_CHARS])
        size += len(line) + 1
    if current:
        pieces.append((current_start, current))
    return pieces


def _json_blocks(text: str) -> list[SourceBlock]:
    data = json.loads(text)
    records = _record_list(data)
    if records is None:
        return split_blocks(text, TestDataFormat.TEXT)
    blocks: list[SourceBlock] = []
    for index, record in enumerate(records, start=1):
        dumped = json.dumps(record, indent=1, ensure_ascii=False)
        blocks.append(SourceBlock(f"B{len(blocks) + 1}", f"item {index}", dumped[:MAX_CHUNK_CHARS]))
    return blocks


def _record_list(data: Any, depth: int = 0) -> list[Any] | None:
    """The largest list of objects in the JSON (exports nest it, e.g. {"suite": {"cases": [...]}})."""
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        return data
    if depth >= 4:
        return None
    candidates: list[list[Any]] = []
    values = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
    for value in values:
        found = _record_list(value, depth + 1)
        if found:
            candidates.append(found)
    return max(candidates, key=len) if candidates else None


def _schema(block_ids: list[str]) -> dict[str, Any]:
    nullable = {"type": ["string", "null"]}
    step = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "action", "target_hint", "locator", "url", "value", "expected"],
        "properties": {
            "text": {"type": "string"},
            "action": {"type": "string", "enum": ACTIONS},
            "target_hint": nullable,
            "locator": nullable,
            "url": nullable,
            "value": nullable,
            "expected": nullable,
        },
    }
    case = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_blocks", "id", "title", "preconditions", "steps", "tags"],
        "properties": {
            "source_blocks": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": block_ids}},
            "id": nullable,
            "title": {"type": "string"},
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "steps": {"type": "array", "minItems": 1, "maxItems": MAX_STEPS_PER_CASE, "items": step},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["cases", "not_test_cases"],
        "properties": {
            "cases": {"type": "array", "maxItems": MAX_CASES_PER_CHUNK, "items": case},
            "not_test_cases": {"type": "array", "items": {"type": "string", "enum": block_ids}},
        },
    }


def _parse_reply(reply: str) -> dict[str, Any]:
    data = json.loads(reply)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("expected an object with a 'cases' list")
    for case in data["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("steps"), list) or not case.get("title"):
            raise ValueError("every case needs a title and a list of steps")
    return data


def _grounded_case(case: dict[str, Any], blocks: dict[str, SourceBlock], warnings: list[str]) -> dict[str, Any]:
    cited = [blocks[block_id] for block_id in case.get("source_blocks") or [] if block_id in blocks]
    haystack = _normalized(" ".join(block.text for block in cited))
    label = case.get("id") or case.get("title") or "?"
    steps: list[dict[str, Any]] = []
    for number, step in enumerate(case.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        kept = dict(step)
        for field in GROUNDED_FIELDS:
            value = kept.get(field)
            if value and not _appears_in(value, haystack):
                warnings.append(UNGROUNDED.format(case=label, step=number, field=field, value=str(value)[:80]))
                kept[field] = None
        action = kept.get("action") or "other"
        steps.append({
            "action": action,
            "target": kept.get("url") if action in NAVIGATION_ACTIONS else kept.get("locator"),
            "target_hint": kept.get("target_hint"),
            "value": kept.get("value"),
            "expected": kept.get("expected"),
            "text": kept.get("text"),
        })
    if _EXPECTATION.search(haystack) and not any(str(step["action"]).startswith("verify") for step in steps):
        warnings.append(NO_CHECK.format(case=label))
    source_id = case.get("id")
    return {
        "id": source_id if source_id and _appears_in(source_id, haystack) else "",
        "title": str(case.get("title") or "").strip(),
        "preconditions": [str(item) for item in case.get("preconditions") or [] if str(item).strip()],
        "steps": steps,
        "tags": [str(tag) for tag in case.get("tags") or [] if str(tag).strip()],
        "source_ref": ", ".join(block.ref for block in cited) or None,
    }


def _dedupe(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[dict[str, Any]] = []
    for case in cases:
        key = (_normalized(case["title"]), tuple(_normalized(step.get("text") or "") for step in case["steps"]))
        if key not in seen:
            seen.add(key)
            unique.append(case)
    return unique


def _unique(cases: list[dict[str, Any]]) -> list[TestCaseSpecModel]:
    """Give every case an id: the source's own when it has one (made unique), else TD-001, TD-002, ..."""
    used: set[str] = set()
    result: list[TestCaseSpecModel] = []
    for index, case in enumerate(cases, start=1):
        base = case["id"] or f"TD-{index:03d}"
        case_id, suffix = base, 2
        while case_id in used:
            case_id, suffix = f"{base}-{suffix}", suffix + 1
        used.add(case_id)
        steps = [TestStepModel(**step) for step in case["steps"]]
        result.append(TestCaseSpecModel(
            id=case_id, title=case["title"] or case_id, preconditions=case["preconditions"], steps=steps,
            tags=case["tags"], source_ref=case["source_ref"],
        ))
    return result


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _appears_in(value: Any, haystack: str) -> bool:
    needle = _normalized(str(value)).strip("\"'")
    return bool(needle) and needle in haystack
