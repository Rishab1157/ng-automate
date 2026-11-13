"""Reading an uploaded test-data file: nothing in its content is required. Only an empty file, broken syntax or a
file type nobody reads as text is refused; JSON whole in our own (or TXGENIE's) shape is read at once, everything else
is a free-form source the model reads."""

import json
import re
from collections.abc import Collection

import pytest

from app.core.exceptions import ValidationError
from app.models.testDataModel import TestDataFormat
from app.utils.TestDataParser import (
    MAX_TEST_DATA_BYTES,
    STRUCTURED_EXTENSIONS,
    TEXT_EXTENSIONS,
    FreeFormSource,
    read_test_data,
)

CASES = [
    {
        "id": "LOGIN-1",
        "title": "Valid user logs in",
        "preconditions": ["user exists"],
        "steps": [
            {"action": "Open", "target": "https://shop.example.com/login"},
            {"action": "type", "target": "id=username", "value": "standard_user"},
            {"action": "type", "target": "id=password", "value": "${password}"},
            {"action": "Click", "target": "css=button[type=submit]"},
            {"action": "verify text", "target": "css=h1", "expected": "Dashboard"},
        ],
        "test_data": {"password": "secret", "retries": 3},
        "tags": ["smoke", "login"],
    },
    {"title": "Cart is empty", "steps": [{"action": "click"}, {"action": "verify-visible", "target": "id=empty"}]},
]


def _read_at_once(content: bytes, filename: str):
    """A file in our own shape, read without a model: the tests below check what that reading makes of it."""
    result = read_test_data(content, filename)
    assert not isinstance(result, FreeFormSource), "this file was expected to be read at once"
    return result


# ---------- refused: only empty files and broken syntax ----------

@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (b"[]", "x.xlsx", ".json, .csv, .txt, .text, .md, .feature, .gherkin, .tsv, .yaml, .yml, .xml, .html, .htm, "
                          ".properties or .ini file"),
        (b"{not json", "x.json", "line 1"),
        (b"[]", "x.json", "empty"),
        (b"{}", "x.json", "empty"),
        (b"", "x.txt", "empty"),
        (b"  \n\n", "x.md", "empty"),
        (b"\xff\xfe\x00", "x.json", "UTF-8"),
        (b"case_id,title,action\n", "x.csv", "no rows below its header"),
    ],
)
def test_only_an_empty_file_or_broken_syntax_is_refused(content: bytes, filename: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        read_test_data(content, filename)


def test_the_message_for_another_file_type_names_every_type_that_is_accepted() -> None:
    with pytest.raises(ValidationError) as caught:
        read_test_data(b"[]", "cases.xlsx")

    assert re.findall(r"\.(\w+)", caught.value.message) == [*STRUCTURED_EXTENSIONS, *TEXT_EXTENSIONS]


def test_a_file_larger_than_the_limit_is_refused() -> None:
    with pytest.raises(ValidationError, match="larger than"):
        read_test_data(b"x" * (MAX_TEST_DATA_BYTES + 1), "x.json")


# ---------- accepted as written: the model reads whatever the user wrote ----------

@pytest.mark.parametrize(("content", "filename"), [
    ("A shopper signs in and sees the products.", "story.txt"),
    ("Feature: Login\n  Scenario: ok\n    Given the shop\n", "login.feature"),
    ("element,locator\nLogin button,#login-button\n", "locators.csv"),
    ("case_id,title,action,target\nC1,Login,open,https://x.test\n", "our-old-columns.csv"),
    ("case_id,title,action\nC1,,click\n", "x.csv"),
    (json.dumps({"cases": [{"name": "Login"}]}), "x.json"),
    (json.dumps([{"steps": [{"action": "open"}]}]), "no-title.json"),
    (json.dumps([{"title": "t", "steps": "click the login button"}]), "steps-as-a-sentence.json"),
    (json.dumps([{"title": "t", "steps": [{"target": "id=x"}]}]), "step-without-action.json"),
    (json.dumps([{"title": "t", "steps": [{"action": "click", "target": "id=x"}] * 201}]), "too-many-steps.json"),
    (json.dumps({"locators": {"login": "#login-button"}, "users": ["standard_user"]}), "data.json"),
])
def test_nothing_in_a_files_content_is_required(content: str, filename: str) -> None:
    result = read_test_data(content.encode(), filename)

    assert isinstance(result, FreeFormSource) and result.text.strip()


def test_csv_keeps_its_text_and_leading_blank_lines_for_the_model() -> None:
    text = "\n\nTest Key,Summary,Step,Expected Result\nX-1,Login,Open the login page,The form is shown\n"

    result = read_test_data(text.encode(), "xray-export.CSV")

    assert result == FreeFormSource(source_format=TestDataFormat.CSV, text=text)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"Summary,Steps\n , \n,,\n", "no rows below its header"),
        ("Summary\n" + "x" * (100 * 1024) + "\n", "at most 100 KB"),
        ("case_id,title,action\nC1,T," + "x" * 140_000 + "\n", "not valid CSV"),  # was an unhandled error (500)
    ],
    ids=["blank-rows", "too-large", "unreadable"],
)
def test_a_csv_without_rows_too_large_or_unreadable_is_refused(content: bytes | str, message: str) -> None:
    data = content.encode() if isinstance(content, str) else content
    with pytest.raises(ValidationError, match=message):
        read_test_data(data, "x.csv")


# ---------- JSON whole in our own shape: read at once, without a model ----------

def test_json_list_and_object_shapes() -> None:
    as_list = _read_at_once(json.dumps(CASES).encode(), "cases.json")
    as_object = _read_at_once(json.dumps({"test_cases": CASES}).encode(), "cases.JSON")

    assert as_list == as_object
    assert as_list.source_format == TestDataFormat.JSON
    first = as_list.cases[0]
    assert [s.action for s in first.steps] == ["open", "type", "type", "click", "verify_text"]
    assert first.test_data == {"password": "secret", "retries": "3"}
    assert first.tags == ["smoke", "login"]


def test_missing_locator_is_flagged_not_guessed() -> None:
    result = _read_at_once(json.dumps(CASES).encode(), "cases.json")
    second = result.cases[1]

    assert second.id == "TD-001"  # it gives no id: the first one new in the project (here: no other test data)
    assert second.steps[0].needs_locator is True
    assert second.steps[0].target is None
    assert second.steps[1].action == "verify_visible"
    assert result.warnings == ["case TD-001 step 1: click needs a locator"]


def test_a_case_with_only_its_words_and_no_steps_is_a_case() -> None:
    result = _read_at_once(json.dumps([{"id": "QX-1", "title": "A locked-out user cannot sign in", "steps": []}]).encode(),
                           "cases.json")

    assert [(case.id, case.title, case.steps) for case in result.cases] == [("QX-1", "A locked-out user cannot sign in", [])]


def test_ids_the_file_does_not_give_are_new_in_the_project_and_given_ones_are_kept() -> None:
    # Every upload numbered its cases without an id from 1 (TC-1, TC-2, ...), so a project had several TC-1 tests.
    cases = [
        {"title": "Login", "steps": [{"action": "open", "target": "https://x.test"}]},
        {"id": "TD-001", "title": "Cart", "steps": [{"action": "open", "target": "https://x.test/cart"}]},
        {"title": "Logout", "steps": [{"action": "open", "target": "https://x.test"}]},
    ]
    txgenie = {
        "test_cases": [{"title": "Login", "steps": [{"action": "Open the site", "expected_result": "It opens"}]}],
    }
    taken = {"TD-001", "TD-002", "C1"}

    def ids(content: str, filename: str, taken_ids: Collection[str] = ()) -> list[str]:
        return [case.id for case in read_test_data(content.encode(), filename, taken_ids).cases]

    # A file's own id is never renamed, also when another upload of the project has it.
    assert ids(json.dumps(cases), "a.json") == ["TD-002", "TD-001", "TD-003"]
    assert ids(json.dumps(cases), "a.json", taken) == ["TD-003", "TD-001", "TD-004"]
    assert ids(json.dumps(txgenie), "b.json", taken) == ["TD-003"]


def test_an_id_the_file_repeats_gets_a_suffix_never_a_refusal() -> None:
    # Two rows of one scenario, copied with the same id: both are cases, named apart (tests are named after ids).
    cases = [{"id": "TD-001", "title": "t", "steps": [{"action": "open", "target": "u"}]}] * 3

    result = _read_at_once(json.dumps(cases).encode(), "x.json")

    assert [case.id for case in result.cases] == ["TD-001", "TD-001-2", "TD-001-3"]


def test_a_step_keeps_every_locator_given_for_its_element() -> None:
    cases = [
        {
            "id": "LOGIN-1",
            "title": "Login",
            "steps": [
                {"action": "click", "locators": ["id=submit", "css=button[type=submit]", "xpath=//button"]},
                {"action": "type", "target": "id=user", "alternate_locators": ["name=username"], "value": "admin"},
            ],
        }
    ]

    result = _read_at_once(json.dumps(cases).encode(), "cases.json")

    steps = result.cases[0].steps
    assert steps[0].target == "id=submit"
    assert steps[0].alternate_locators == ["css=button[type=submit]", "xpath=//button"]
    assert steps[0].needs_locator is False
    assert steps[1].target == "id=user"
    assert steps[1].alternate_locators == ["name=username"]


def test_a_txgenie_steps_locator_becomes_its_target() -> None:
    case = {"id": "TC001", "title": "Login", "steps": [
        {"step_number": 1, "action": "Click the Login button (locator: [data-test='login-button'])", "test_data": "",
         "expected_result": "The products page opens"},
        {"step_number": 2, "action": "Type the user name", "locator": "#user-name", "test_data": "standard_user",
         "expected_result": ""},
        {"step_number": 3, "action": "Check the title", "test_data": "", "expected_result": "Products"},
    ]}

    data_set = read_test_data(json.dumps({"test_cases": [case]}).encode(), "txgenie.json")

    assert [step.target for step in data_set.cases[0].steps] == ["[data-test='login-button']", "#user-name", None]
