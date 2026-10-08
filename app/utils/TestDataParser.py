"""Read an uploaded test-data source: our own JSON/CSV format at once, anything else as a free-form source.

Our format is a list of test cases with steps (JSON, or CSV rows). JSON in any other shape and plain text (.txt, .md)
are free-form: an LLM turns them into test cases later (TestCaseExtractor). Locators come from the source. A step that
needs an element and names none (no locator, no element in words) is marked `needs_locator` and reported; it is never
guessed. Errors name the case or row so users can fix their file.
"""

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import ValidationError
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataSetModel, TestStepModel

MAX_TEST_DATA_BYTES = 5 * 1024 * 1024
# Free-form sources are read by an LLM: bigger ones would take hours with a small local model.
MAX_FREE_FORM_BYTES = 100 * 1024
MAX_CASES = 500
MAX_STEPS_PER_CASE = 200

# Actions that act on an element and so need a locator.
TARGETED_ACTIONS = frozenset({
    "click", "tap", "type", "select", "check", "uncheck", "hover", "verify_text", "verify_visible", "verify_value",
    "upload",
})
NAVIGATION_ACTIONS = frozenset({"open", "navigate"})
CSV_COLUMNS = ("case_id", "title", "action", "target", "value", "expected", "tags", "description")

STRUCTURED_EXTENSIONS = ("json", "csv")
TEXT_EXTENSIONS = ("txt", "md", "text", "feature")

UNSUPPORTED_FORMAT = "Test data must be a .json, .csv, .txt or .md file"
TOO_LARGE = f"Test data is larger than {MAX_TEST_DATA_BYTES // (1024 * 1024)} MB"
FREE_FORM_TOO_LARGE = (
    f"Free-form test data (text, or JSON not in the test-case format) must be at most {MAX_FREE_FORM_BYTES // 1024} KB"
)
EMPTY_SOURCE = "The test data file is empty"
NOT_UTF8 = "Test data must be UTF-8 text"
INVALID_JSON = "Test data is not valid JSON: {reason}"
JSON_SHAPE = 'Test data JSON must be a list of test cases or {"test_cases": [...]}'
CSV_MISSING_COLUMNS = "Test data CSV is missing the column(s): {columns}"
CASE_NOT_OBJECT = "Test case {index} must be an object"
CASE_FIELD_REQUIRED = "Test case {case}: '{field}' is required"
STEP_ACTION_REQUIRED = "Test case {case} step {step}: 'action' is required"
TOO_MANY_CASES = f"Test data has more than {MAX_CASES} test cases"
TOO_MANY_STEPS = f"Test case {{case}} has more than {MAX_STEPS_PER_CASE} steps"
DUPLICATE_CASE = "Test case id '{case}' is used more than once"
NO_CASES = "Test data contains no test cases"
NEEDS_LOCATOR = "case {case} step {step}: {action} needs a locator"
NEEDS_URL = "case {case} step {step}: {action} needs a URL"


@dataclass(frozen=True)
class FreeFormSource:
    """A source an LLM must read first: JSON in another shape, or plain text."""

    source_format: TestDataFormat
    text: str


def read_test_data(content: bytes, filename: str) -> TestDataSetModel | FreeFormSource:
    """Our own format -> its test cases; anything else we accept -> a free-form source. Raises ValidationError."""
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in (*STRUCTURED_EXTENSIONS, *TEXT_EXTENSIONS):
        raise ValidationError(UNSUPPORTED_FORMAT)
    if len(content) > MAX_TEST_DATA_BYTES:
        raise ValidationError(TOO_LARGE)
    text = _decode(content)
    if not text.strip():
        raise ValidationError(EMPTY_SOURCE)

    if extension == "csv":
        return _build(_csv_cases(text), TestDataFormat.CSV)
    if extension == "json":
        data = _load_json(text)
        raw_cases = _own_format_cases(data)
        if raw_cases is not None:
            return _build(raw_cases, TestDataFormat.JSON)
        return _free_form(json.dumps(data, indent=2, ensure_ascii=False), TestDataFormat.JSON)
    return _free_form(text, TestDataFormat.TEXT)


def parse_test_data(content: bytes, filename: str) -> TestDataSetModel:
    """Our own JSON/CSV format only (free-form sources are a ValidationError here)."""
    result = read_test_data(content, filename)
    if isinstance(result, FreeFormSource):
        raise ValidationError(JSON_SHAPE if result.source_format == TestDataFormat.JSON else UNSUPPORTED_FORMAT)
    return result


def build_test_data_set(cases: list[TestCaseSpecModel], source_format: TestDataFormat) -> TestDataSetModel:
    """Check edited or extracted cases (ids, limits) and recompute which steps still need an element."""
    raw = [case.model_dump() for case in cases]
    return _build(raw, source_format)


def _build(raw_cases: list[Any], source_format: TestDataFormat) -> TestDataSetModel:
    if not raw_cases:
        raise ValidationError(NO_CASES)
    if len(raw_cases) > MAX_CASES:
        raise ValidationError(TOO_MANY_CASES)

    warnings: list[str] = []
    cases: list[TestCaseSpecModel] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        case = _case(raw, index, warnings)
        if case.id in seen:
            raise ValidationError(DUPLICATE_CASE.format(case=case.id))
        seen.add(case.id)
        cases.append(case)
    return TestDataSetModel(source_format=source_format, cases=cases, warnings=warnings)


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError(NOT_UTF8) from None


def _free_form(text: str, source_format: TestDataFormat) -> FreeFormSource:
    if len(text.encode("utf-8")) > MAX_FREE_FORM_BYTES:
        raise ValidationError(FREE_FORM_TOO_LARGE)
    return FreeFormSource(source_format=source_format, text=text)


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValidationError(INVALID_JSON.format(reason=f"line {error.lineno}, column {error.colno}: {error.msg}")) from None


def _own_format_cases(data: Any) -> list[Any] | None:
    """The cases when the JSON is meant to be our format (cases with steps), else None (a free-form source).

    JSON that is clearly our format is parsed strictly, so its mistakes are reported instead of guessed around.
    """
    if isinstance(data, dict) and isinstance(data.get("test_cases"), list):
        return data["test_cases"]
    if isinstance(data, list) and all(isinstance(item, dict) and "steps" in item for item in data):
        return data  # an empty list too: "no test cases" says more than asking an LLM to read nothing
    return None


def _csv_cases(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    columns = {(name or "").strip().lower() for name in reader.fieldnames or []}
    missing = [column for column in ("case_id", "title", "action") if column not in columns]
    if missing:
        raise ValidationError(CSV_MISSING_COLUMNS.format(columns=", ".join(missing)))

    cases: dict[str, dict[str, Any]] = {}
    for row in reader:
        values = {(key or "").strip().lower(): (value or "").strip() for key, value in row.items() if key is not None}
        if not any(values.values()):
            continue
        case_id = values.get("case_id", "")
        case = cases.setdefault(case_id, {
            "id": case_id,
            "title": values.get("title", ""),
            "description": values.get("description") or None,
            "tags": [tag.strip() for tag in re.split(r"[;,]", values.get("tags", "")) if tag.strip()],
            "steps": [],
        })
        case["steps"].append({key: values.get(key) or None for key in ("action", "target", "value", "expected")})
        if len(cases) > MAX_CASES:
            raise ValidationError(TOO_MANY_CASES)
    return list(cases.values())


def _case(raw: Any, index: int, warnings: list[str]) -> TestCaseSpecModel:
    if not isinstance(raw, dict):
        raise ValidationError(CASE_NOT_OBJECT.format(index=index))
    case_id = _text(raw.get("id")) or f"TC-{index}"
    title = _text(raw.get("title"))
    if not title:
        raise ValidationError(CASE_FIELD_REQUIRED.format(case=case_id, field="title"))
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValidationError(CASE_FIELD_REQUIRED.format(case=case_id, field="steps"))
    if len(raw_steps) > MAX_STEPS_PER_CASE:
        raise ValidationError(TOO_MANY_STEPS.format(case=case_id))

    steps = [_step(step, case_id, number, warnings) for number, step in enumerate(raw_steps, start=1)]
    test_data = raw.get("test_data") if isinstance(raw.get("test_data"), dict) else {}
    return TestCaseSpecModel(
        id=case_id,
        title=title,
        description=_text(raw.get("description")) or None,
        preconditions=[_text(item) for item in raw.get("preconditions") or [] if _text(item)],
        steps=steps,
        test_data={str(key): _text(value) for key, value in test_data.items()},
        tags=[_text(tag) for tag in raw.get("tags") or [] if _text(tag)],
        source_ref=_text(raw.get("source_ref")) or None,
    )


def _step(raw: Any, case_id: str, number: int, warnings: list[str]) -> TestStepModel:
    raw = raw if isinstance(raw, dict) else {}
    action = re.sub(r"[\s-]+", "_", _text(raw.get("action")).lower())
    if not action:
        raise ValidationError(STEP_ACTION_REQUIRED.format(case=case_id, step=number))
    target = _text(raw.get("target")) or None
    target_hint = _text(raw.get("target_hint") or raw.get("element")) or None
    needs_locator = action in TARGETED_ACTIONS and target is None and target_hint is None
    if needs_locator:
        warnings.append(NEEDS_LOCATOR.format(case=case_id, step=number, action=action))
    elif action in NAVIGATION_ACTIONS and target is None:
        warnings.append(NEEDS_URL.format(case=case_id, step=number, action=action))
    return TestStepModel(
        action=action,
        target=target,
        target_hint=target_hint,
        value=_text(raw.get("value")) or None,
        expected=_text(raw.get("expected")) or None,
        text=_text(raw.get("text")) or None,
        needs_locator=needs_locator,
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()
