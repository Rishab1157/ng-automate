"""Read an uploaded test-data file. Nothing in its content is required.

The user's files are their knowledge base, written however they write them: stories, manual test cases, Gherkin, a
sheet of locators, notes, in any language. A file is refused only when it is empty or its syntax is broken (JSON that
does not parse, text that is not UTF-8, a file type no person reads as text). Everything else is accepted as it is.
JSON in our own shape (a list of cases with steps) or TXGENIE's shape is read at once, without a model; any other file
is a free-form source that an LLM reads later (TestCaseExtractor), and a file in our shape with something odd in it
is read that way too, never refused. A step that needs an element and names none is marked `needs_locator` and
reported (never guessed: the generator finds or assumes the locator). A case without an id gets the first TD-001,
TD-002, ... that no other case of the project has: tests are named after case ids.
"""

import csv
import io
import itertools
import json
import re
from collections.abc import Collection, Iterator
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
# A step whose action the file does not name: the generator reads the step's words.
OTHER_ACTION = "other"

STRUCTURED_EXTENSIONS = ("json", "csv")
# Any plain-text file a person reads without a tool: stories, manual cases, Gherkin, locator sheets, page sources, notes.
TEXT_EXTENSIONS = (
    "txt", "text", "md", "feature", "gherkin", "tsv", "yaml", "yml", "xml", "html", "htm", "properties", "ini",
)
ACCEPTED_EXTENSIONS = (*STRUCTURED_EXTENSIONS, *TEXT_EXTENSIONS)

UNSUPPORTED_FORMAT = (
    f"Test data must be a {', '.join(f'.{name}' for name in ACCEPTED_EXTENSIONS[:-1])} "
    f"or .{ACCEPTED_EXTENSIONS[-1]} file"
)
TOO_LARGE = f"Test data is larger than {MAX_TEST_DATA_BYTES // (1024 * 1024)} MB"
FREE_FORM_TOO_LARGE = (
    "Free-form test data (text, or JSON or CSV not in the test-case format) must be at most "
    f"{MAX_FREE_FORM_BYTES // 1024} KB"
)
EMPTY_SOURCE = "The test data file is empty"
EMPTY_CSV = "The test data CSV has no rows below its header"
NOT_UTF8 = "Test data must be UTF-8 text"
INVALID_JSON = "Test data is not valid JSON: {reason}"
INVALID_CSV = "Test data is not valid CSV: {reason}"
TOO_MANY_CASES = f"Test data has more than {MAX_CASES} test cases"
TOO_MANY_STEPS = f"Test case {{case}} has more than {MAX_STEPS_PER_CASE} steps"
NO_CASES = "Test data contains no test cases"
NEEDS_LOCATOR = "case {case} step {step}: {action} needs a locator"
NEEDS_URL = "case {case} step {step}: {action} needs a URL"


class NoTestCases(ValueError):
    """Every part of a free-form source was read and none holds a test case (a sheet of locators, a note with
    accounts): not a failure, the file is context for the test generator."""


@dataclass(frozen=True)
class FreeFormSource:
    """A source an LLM must read first: anything that is not JSON in our own or TXGENIE's shape."""

    source_format: TestDataFormat
    text: str


def read_test_data(
    content: bytes, filename: str, taken_ids: Collection[str] = ()
) -> TestDataSetModel | FreeFormSource:
    """A test-data file as the user wrote it: refused only when empty or its syntax is broken. JSON in our own or
    TXGENIE's shape -> its cases at once; anything else -> a free-form source the model reads. `taken_ids`: the case
    ids the project's other test data has, which the ids given to cases without one must not repeat."""
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in ACCEPTED_EXTENSIONS:
        raise ValidationError(UNSUPPORTED_FORMAT)
    if len(content) > MAX_TEST_DATA_BYTES:
        raise ValidationError(TOO_LARGE)
    text = _decode(content)
    if not text.strip():
        raise ValidationError(EMPTY_SOURCE)

    if extension == "csv":
        rows = _csv_rows(text)  # the syntax
        if not any(cell.strip() for row in rows[1:] for cell in row):
            raise ValidationError(EMPTY_CSV)
        return _free_form(text, TestDataFormat.CSV)
    if extension == "json":
        data = _load_json(text)  # the syntax
        if data in ([], {}):
            raise ValidationError(EMPTY_SOURCE)
        try:
            txgenie = _txgenie_cases(data, taken_ids)
            if txgenie is not None:
                return txgenie
            raw_cases = _own_format_cases(data)
            if raw_cases is not None:
                return _build(raw_cases, TestDataFormat.JSON, taken_ids)
        except ValidationError:
            pass  # our shape, but too much of it: the model reads it in parts like any other source
        return _free_form(json.dumps(data, indent=2, ensure_ascii=False), TestDataFormat.JSON)
    return _free_form(text, TestDataFormat.TEXT)


def build_test_data_set(
    cases: list[TestCaseSpecModel], source_format: TestDataFormat, taken_ids: Collection[str] = ()
) -> TestDataSetModel:
    """Edited or extracted cases as a data set: ids made unique, limits checked, and which steps still need an
    element worked out again."""
    raw = [case.model_dump() for case in cases]
    return _build(raw, source_format, taken_ids)


def new_case_ids(taken: Collection[str]) -> Iterator[str]:
    """TD-001, TD-002, ... without the ids in `taken`: the ids of cases their source gives none."""
    return (case_id for number in itertools.count(1) if (case_id := f"TD-{number:03d}") not in taken)


def _build(raw_cases: list[Any], source_format: TestDataFormat, taken_ids: Collection[str] = ()) -> TestDataSetModel:
    if not raw_cases:
        raise ValidationError(NO_CASES)
    if len(raw_cases) > MAX_CASES:
        raise ValidationError(TOO_MANY_CASES)

    warnings: list[str] = []
    cases: list[TestCaseSpecModel] = []
    seen: set[str] = set()
    new_ids = new_case_ids({*taken_ids, *_given_ids(raw_cases)})
    for raw in raw_cases:
        case = _case(raw, warnings, new_ids, seen)
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


def _csv_rows(text: str) -> list[list[str]]:
    try:
        return list(csv.reader(io.StringIO(text)))
    except csv.Error as error:
        raise ValidationError(INVALID_CSV.format(reason=error)) from None


def _txgenie_cases(data: Any, taken_ids: Collection[str] = ()) -> TestDataSetModel | None:
    """TXGENIE's generated test cases (steps with action / test_data / expected_result), else None.

    Their steps are sentences ("Click the Login button"), not our action words, so they are kept as written: the
    test generator reads them as they are. Nothing is guessed and no LLM is needed.
    """
    raw_cases = (data.get("test_cases") or data.get("testcases")) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list) or not raw_cases or not all(isinstance(case, dict) for case in raw_cases):
        return None
    if not any(
        isinstance(step, dict) and ("expected_result" in step or "test_data" in step)
        for case in raw_cases
        for step in case.get("steps") or []
    ):
        return None
    if len(raw_cases) > MAX_CASES:
        raise ValidationError(TOO_MANY_CASES)
    cases: list[TestCaseSpecModel] = []
    seen: set[str] = set()
    new_ids = new_case_ids({*taken_ids, *_given_ids(raw_cases)})
    for index, raw in enumerate(raw_cases, start=1):
        case = _txgenie_case(raw, index, new_ids, seen)
        seen.add(case.id)
        cases.append(case)
    return TestDataSetModel(source_format=TestDataFormat.JSON, cases=cases, warnings=[])


# TX Genie writes an element's locator at the end of a step's action: "Click Login (locator: #login-button)".
_TXGENIE_LOCATOR = re.compile(r"\(locator:\s*(?P<locator>.+)\)\s*$", re.IGNORECASE)


def _txgenie_locator(step: dict[str, Any]) -> str | None:
    """The step's locator: a `locator` field, or the one at the end of its action."""
    match = _TXGENIE_LOCATOR.search(_text(step.get("action")))
    return _text(step.get("locator")) or (match["locator"].strip() if match else None)


def _txgenie_case(raw: dict[str, Any], index: int, new_ids: Iterator[str], seen: Collection[str]) -> TestCaseSpecModel:
    case_id = _unique_id(_text(raw.get("id")) or next(new_ids), seen)
    raw_steps = [step for step in raw.get("steps") or [] if isinstance(step, dict) and _text(step.get("action"))]
    if len(raw_steps) > MAX_STEPS_PER_CASE:
        raise ValidationError(TOO_MANY_STEPS.format(case=case_id))
    raw_steps.sort(key=lambda step: step.get("step_number") if isinstance(step.get("step_number"), int) else 0)
    steps = [
        TestStepModel(
            action=_text(step.get("action")),
            target=_txgenie_locator(step),
            value=_free_text(step.get("test_data")) or None,
            expected=_free_text(step.get("expected_result")) or None,
            text=_text(step.get("action")),
        )
        for step in raw_steps
    ]
    preconditions = raw.get("preconditions")
    if isinstance(preconditions, str):
        preconditions = [line.strip(" -•\t") for line in preconditions.splitlines()]
    case_data = raw.get("test_data")
    test_data = (
        {str(key): _free_text(value) for key, value in case_data.items()}
        if isinstance(case_data, dict)
        else ({"data": _free_text(case_data)} if _free_text(case_data) else {})
    )
    labels = [raw.get(field) for field in ("priority", "test_type", "module", "business_area")]
    tags = [_text(tag) for tag in [*labels, *(raw.get("tags") or [])] if _text(tag)]
    return TestCaseSpecModel(
        id=case_id,
        title=_text(raw.get("title")) or f"Test case {index}",
        description=_text(raw.get("description")) or None,
        preconditions=[_text(item) for item in preconditions or [] if _text(item)],
        steps=steps,
        test_data=test_data,
        tags=list(dict.fromkeys(tags)),
    )


def _free_text(value: Any) -> str:
    """Test data or an expected result as text: TXGENIE sometimes sends objects or lists there."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False) if value else ""
    return _text(value)


def _own_format_cases(data: Any) -> list[Any] | None:
    """The cases when the JSON is whole in our own shape: a list (or {"test_cases": [...]}) of cases that each have a
    title and a list of steps with an action. Else None: the model reads it like any other file (it may be our shape
    with a piece missing; nothing is required, this only decides who reads it)."""
    cases = data.get("test_cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not all(_in_our_shape(case) for case in cases):
        return None
    return cases


def _in_our_shape(case: Any) -> bool:
    return (
        isinstance(case, dict)
        and bool(_text(case.get("title")))
        and isinstance(case.get("steps"), list)
        and all(isinstance(step, dict) and bool(_text(step.get("action"))) for step in case["steps"])
    )


def _given_ids(raw_cases: list[Any]) -> set[str]:
    """The ids the source gives its cases: never given to another case."""
    return {case_id for raw in raw_cases if isinstance(raw, dict) and (case_id := _text(raw.get("id")))}


def _unique_id(case_id: str, seen: Collection[str]) -> str:
    """The id, with a suffix when another case of the set already has it (TD-L02, TD-L02-2): tests are named after
    ids, so no two cases share one."""
    unique, suffix = case_id, 2
    while unique in seen:
        unique, suffix = f"{case_id}-{suffix}", suffix + 1
    return unique


def _case(raw: Any, warnings: list[str], new_ids: Iterator[str], seen: Collection[str]) -> TestCaseSpecModel:
    """A case as the file gives it. Nothing in it is required: a missing title comes from its first step or its id,
    steps given as one sentence are one step, and a step without an action is one the generator reads in words."""
    raw = raw if isinstance(raw, dict) else {"title": _text(raw)}
    case_id = _unique_id(_text(raw.get("id")) or next(new_ids), seen)
    raw_steps = raw.get("steps")
    if raw_steps is None:
        raw_steps = []
    elif not isinstance(raw_steps, list):
        raw_steps = [raw_steps]
    if len(raw_steps) > MAX_STEPS_PER_CASE:
        raise ValidationError(TOO_MANY_STEPS.format(case=case_id))
    steps = [_step(step, case_id, number, warnings) for number, step in enumerate(raw_steps, start=1)]
    title = _text(raw.get("title") or raw.get("name")) or next(
        (step.text or step.action for step in steps if step.text or step.action), case_id
    )
    test_data = raw.get("test_data") if isinstance(raw.get("test_data"), dict) else {}
    preconditions = raw.get("preconditions")
    if isinstance(preconditions, str):
        preconditions = [preconditions]
    return TestCaseSpecModel(
        id=case_id,
        title=title,
        description=_text(raw.get("description")) or None,
        preconditions=[_text(item) for item in preconditions or [] if _text(item)],
        steps=steps,
        test_data={str(key): _text(value) for key, value in test_data.items()},
        tags=[_text(tag) for tag in raw.get("tags") or [] if _text(tag)],
        source_ref=_text(raw.get("source_ref")) or None,
    )


def _step(raw: Any, case_id: str, number: int, warnings: list[str]) -> TestStepModel:
    """A step as the file gives it: a dict with our fields, or just its words."""
    raw = raw if isinstance(raw, dict) else {"text": _text(raw)}
    action = re.sub(r"[\s-]+", "_", _text(raw.get("action")).lower()) or OTHER_ACTION
    target = _text(raw.get("target")) or None
    target_hint = _text(raw.get("target_hint") or raw.get("element")) or None
    alternates = _alternate_locators(raw)
    # A step may list its locators as `locators: [...]`: the first is the primary, the rest are fallbacks.
    if target is None and action not in NAVIGATION_ACTIONS and alternates:
        target, alternates = alternates[0], alternates[1:]
    needs_locator = action in TARGETED_ACTIONS and target is None and target_hint is None
    if needs_locator:
        warnings.append(NEEDS_LOCATOR.format(case=case_id, step=number, action=action))
    elif action in NAVIGATION_ACTIONS and target is None:
        warnings.append(NEEDS_URL.format(case=case_id, step=number, action=action))
    return TestStepModel(
        action=action,
        target=target,
        target_hint=target_hint,
        alternate_locators=alternates,
        value=_text(raw.get("value")) or None,
        expected=_text(raw.get("expected")) or None,
        text=_text(raw.get("text")) or None,
        needs_locator=needs_locator,
    )


def _alternate_locators(raw: dict[str, Any]) -> list[str]:
    """The other locators the step gives for the same element, in the source's order."""
    values: list[Any] = []
    for key in ("alternate_locators", "alternates", "locators"):
        item = raw.get(key)
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, list):
            values.extend(item)
    return [text for item in values if (text := _text(item))]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()
