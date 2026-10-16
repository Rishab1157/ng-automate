"""TestCaseExtractor: free-form text / JSON -> test cases, with every value checked against the source.

The LLM is a fake `complete` function: it receives what the real structured-output call would get and returns a reply.
"""

import json
from typing import Any

import pytest
from pydantic import SecretStr

from app.agents.TestGeneratorAgent.TestCaseExtractor import (
    MAX_CHUNK_CHARS,
    NO_CASES_FOUND,
    TestCaseExtractor,
    chunk_blocks,
    split_blocks,
)
from app.models.llmModel import LlmConfigModel
from app.models.testDataModel import TestDataFormat
from app.models.tokenUsageModel import TokenUsageModel

TEXT = """# Login feature - manual test cases

TC-101: Valid user can log in
1. Open https://www.saucedemo.com/
2. Enter "standard_user" in the Username field
3. Click the Login button
Expected: the Products page is shown

Release notes: version 2.3 fixed the footer links.
"""


def _step(text: str, action: str, **fields: Any) -> dict[str, Any]:
    step = {"text": text, "action": action, "target_hint": None, "locator": None, "alternate_locators": [],
            "url": None, "value": None, "expected": None}
    step.update(fields)
    return step


GOOD_CASE = {
    "source_blocks": ["B2"],
    "id": "TC-101",
    "title": "Valid user can log in",
    "preconditions": [],
    "tags": [],
    "steps": [
        _step("Open https://www.saucedemo.com/", "open", url="https://www.saucedemo.com/"),
        _step('Enter "standard_user" in the Username field', "type", target_hint="Username field", value="standard_user"),
        _step("Click the Login button", "click", target_hint="Login button"),
        _step("the Products page is shown", "verify_visible", target_hint="Products page", expected="Products"),
    ],
}


class FakeComplete:
    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, llm_config: Any, instructions: str, text: str, schema_name: str, schema: dict[str, Any],
                 parse: Any, timeout_seconds: int = 0, on_usage: Any = None) -> Any:
        self.calls.append({"text": text, "schema": schema, "timeout": timeout_seconds})
        if on_usage is not None:
            on_usage(TokenUsageModel(input_tokens=100, output_tokens=10, calls=1))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return parse(json.dumps(reply))


def _extractor(fake: FakeComplete) -> TestCaseExtractor:
    return TestCaseExtractor(LlmConfigModel(model="ollama_chat/devstral", api_key=SecretStr("x" * 12)), complete=fake, timeout_seconds=7)


def test_text_blocks_carry_their_line_numbers() -> None:
    blocks = split_blocks(TEXT, TestDataFormat.TEXT)

    assert [(b.id, b.ref) for b in blocks] == [("B1", "line 1"), ("B2", "lines 3-7"), ("B3", "line 9")]
    assert blocks[1].text.startswith("TC-101")


def test_case_is_read_with_its_source_and_element_hints() -> None:
    fake = FakeComplete({"cases": [GOOD_CASE], "not_test_cases": ["B1", "B3"]})

    data_set = _extractor(fake).extract(TEXT, TestDataFormat.TEXT)

    [case] = data_set.cases
    assert (case.id, case.title, case.source_ref) == ("TC-101", "Valid user can log in", "lines 3-7")
    open_step, type_step, click_step, _ = case.steps
    assert open_step.target == "https://www.saucedemo.com/"
    assert (type_step.target, type_step.target_hint, type_step.value) == (None, "Username field", "standard_user")
    assert (click_step.target_hint, click_step.needs_locator) == ("Login button", False)
    assert data_set.source_format == TestDataFormat.TEXT
    # The prompt names every block, and the schema only allows those block ids.
    assert "[B2] (lines 3-7)" in fake.calls[0]["text"]
    assert fake.calls[0]["schema"]["properties"]["cases"]["items"]["properties"]["source_blocks"]["items"]["enum"] == ["B1", "B2", "B3"]
    assert fake.calls[0]["timeout"] == 7


def test_values_that_are_not_in_the_source_are_dropped_and_reported() -> None:
    invented = dict(GOOD_CASE, steps=[
        _step("Open the site", "open", url="https://invented.example"),
        _step("Click Login", "click", locator="#login-btn", target_hint="Login button"),
        _step("Check title", "verify_text", target_hint="title", expected="Products"),
    ])
    fake = FakeComplete({"cases": [invented], "not_test_cases": []})

    data_set = _extractor(fake).extract(TEXT, TestDataFormat.TEXT)

    open_step, click_step, verify_step = data_set.cases[0].steps
    assert open_step.target is None
    assert click_step.target is None and click_step.target_hint == "Login button"
    assert verify_step.expected == "Products"  # it is in the source
    assert any("https://invented.example" in w for w in data_set.warnings)
    assert any("#login-btn" in w for w in data_set.warnings)


def test_invented_case_id_is_replaced_and_duplicates_are_merged() -> None:
    no_id = dict(GOOD_CASE, id="TC-999")
    fake = FakeComplete({"cases": [no_id, dict(no_id)], "not_test_cases": []})

    data_set = _extractor(fake).extract(TEXT, TestDataFormat.TEXT)

    assert [case.id for case in data_set.cases] == ["TD-001"]


def test_a_part_that_fails_is_reported_and_the_rest_is_kept() -> None:
    long_text = TEXT + "\n" + "\n\n".join(f"Paragraph {n}: " + "x" * 1500 for n in range(8))
    chunks = chunk_blocks(split_blocks(long_text, TestDataFormat.TEXT))
    assert len(chunks) >= 2 and all(sum(len(b.text) for b in c) <= MAX_CHUNK_CHARS for c in chunks)
    replies: list[Any] = [{"cases": [GOOD_CASE], "not_test_cases": []}] + [ValueError("the model call failed")] * (len(chunks) - 1)

    data_set = _extractor(FakeComplete(*replies)).extract(long_text, TestDataFormat.TEXT)

    assert [case.id for case in data_set.cases] == ["TC-101"]
    assert data_set.warnings[0].startswith("part 2 of the source could not be read")


def test_several_locators_for_one_element_are_kept_in_order_and_grounded() -> None:
    source = (
        "# Login - locators\n\n"
        "TC-201: Login button\n"
        "1. Open https://www.saucedemo.com/\n"
        "2. Click the submit button (id=login-button, css=#login-button, xpath=//button[@id='login-button'])\n"
    )
    case = {
        "source_blocks": ["B2"],
        "id": "TC-201",
        "title": "Login button",
        "preconditions": [],
        "tags": [],
        "steps": [
            _step("Open https://www.saucedemo.com/", "open", url="https://www.saucedemo.com/"),
            _step("Click the submit button", "click", locator="id=login-button",
                  alternate_locators=["css=#login-button", "xpath=//button[@id='login-button']"]),
        ],
    }

    data_set = _extractor(FakeComplete({"cases": [case], "not_test_cases": []})).extract(source, TestDataFormat.TEXT)

    click = data_set.cases[0].steps[1]
    assert click.target == "id=login-button"
    assert click.alternate_locators == ["css=#login-button", "xpath=//button[@id='login-button']"]


def test_an_alternate_locator_not_in_the_source_is_dropped_and_reported() -> None:
    source = (
        "# Login - locators\n\n"
        "TC-202: Login button\n"
        "1. Click the submit button (id=login-button, css=#login-button)\n"
    )
    case = {
        "source_blocks": ["B2"], "id": "TC-202", "title": "Login button", "preconditions": [], "tags": [],
        "steps": [
            _step("Click the submit button", "click", locator="id=login-button",
                  alternate_locators=["css=#login-button", "css=.invented"]),
        ],
    }

    data_set = _extractor(FakeComplete({"cases": [case], "not_test_cases": []})).extract(source, TestDataFormat.TEXT)

    assert data_set.cases[0].steps[0].alternate_locators == ["css=#login-button"]
    assert any("css=.invented" in warning for warning in data_set.warnings)


def test_a_locator_from_another_section_of_the_source_is_kept() -> None:
    source = (
        "PROJECT: Demo\n"
        "BASE URL: https://demo.example/login\n"
        "\n"
        "Scenario: Valid login\n"
        "1. Type Admin in the Username field\n"
        "2. Click the Login button\n"
        "Expected: the dashboard is shown\n"
        "\n"
        "SELECTORS\n"
        "Username field: name = username ; css = input[name='username']\n"
        "Dashboard title: css = .oxd-topbar-header-breadcrumb h6"
    )
    # The case cites only the scenario block; its url and locators are written in other sections of the source.
    case = {
        "source_blocks": ["B2"], "id": "TC-1", "title": "Valid login", "preconditions": [], "tags": [],
        "steps": [
            _step("Open the login page", "open", url="https://demo.example/login"),
            _step("Type Admin in the Username field", "type", target_hint="Username field",
                  locator="name=username", alternate_locators=["css=input[name='username']"], value="Admin"),
            _step("Click the Login button", "click", target_hint="Login button"),
            _step("the dashboard is shown", "verify_visible", target_hint="Dashboard title",
                  locator="css=.oxd-topbar-header-breadcrumb h6", expected="the dashboard is shown"),
        ],
    }

    data_set = _extractor(FakeComplete({"cases": [case], "not_test_cases": []})).extract(source, TestDataFormat.TEXT)

    steps = data_set.cases[0].steps
    assert steps[0].target == "https://demo.example/login"
    assert (steps[1].target, steps[1].alternate_locators) == ("name=username", ["css=input[name='username']"])
    assert steps[3].target == "css=.oxd-topbar-header-breadcrumb h6"
    assert not any("dropped" in warning for warning in data_set.warnings)


def test_nothing_readable_is_an_error() -> None:
    with pytest.raises(ValueError) as caught:
        _extractor(FakeComplete({"cases": [], "not_test_cases": ["B1", "B2", "B3"]})).extract(TEXT, TestDataFormat.TEXT)

    assert str(caught.value) == NO_CASES_FOUND


def test_json_export_records_become_blocks() -> None:
    export = {"suite": {"name": "Login", "tests": [{"name": "Valid login"}, {"name": "Locked user"}]}, "version": 3}

    blocks = split_blocks(json.dumps(export), TestDataFormat.JSON)

    assert [(b.id, b.ref) for b in blocks] == [("B1", "item 1"), ("B2", "item 2")]
    assert json.loads(blocks[1].text) == {"name": "Locked user"}


def test_case_without_a_check_for_an_expected_result_is_flagged() -> None:
    no_check = dict(GOOD_CASE, steps=GOOD_CASE["steps"][:3])  # the source says "Expected: ..." but no verify step
    fake = FakeComplete({"cases": [no_check], "not_test_cases": []})

    data_set = _extractor(fake).extract(TEXT, TestDataFormat.TEXT)

    assert "case TC-101: the source states an expected result but the case has no verify step" in data_set.warnings
