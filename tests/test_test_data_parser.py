import json

import pytest

from app.core.exceptions import ValidationError
from app.models.testDataModel import TestDataFormat
from app.utils.TestDataParser import MAX_TEST_DATA_BYTES, parse_test_data

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


def test_json_list_and_object_shapes() -> None:
    as_list = parse_test_data(json.dumps(CASES).encode(), "cases.json")
    as_object = parse_test_data(json.dumps({"test_cases": CASES}).encode(), "cases.JSON")

    assert as_list == as_object
    assert as_list.source_format == TestDataFormat.JSON
    first = as_list.cases[0]
    assert [s.action for s in first.steps] == ["open", "type", "type", "click", "verify_text"]
    assert first.test_data == {"password": "secret", "retries": "3"}
    assert first.tags == ["smoke", "login"]


def test_missing_locator_is_flagged_not_guessed() -> None:
    result = parse_test_data(json.dumps(CASES).encode(), "cases.json")
    second = result.cases[1]

    assert second.id == "TC-2"
    assert second.steps[0].needs_locator is True
    assert second.steps[0].target is None
    assert second.steps[1].action == "verify_visible"
    assert result.warnings == ["case TC-2 step 1: click needs a locator"]


def test_csv_rows_grouped_by_case_with_bom() -> None:
    csv_text = (
        "﻿Case_ID,Title,Action,Target,Value,Expected,Tags\n"
        "C1,Login,open,https://x.test,,,smoke;login\n"
        "C1,Login,type,id=user,bob,,\n"
        "C2,Logout,click,,,,\n"
    )

    result = parse_test_data(csv_text.encode("utf-8"), "data.csv")

    assert result.source_format == TestDataFormat.CSV
    assert [(c.id, len(c.steps)) for c in result.cases] == [("C1", 2), ("C2", 1)]
    assert result.cases[0].tags == ["smoke", "login"]
    assert result.cases[1].steps[0].needs_locator is True


@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (b"[]", "x.xlsx", ".json, .csv, .txt or .md"),
        (b"{not json", "x.json", "line 1"),
        (b'{"cases": []}', "x.json", "list of test cases"),
        (b"[]", "x.json", "no test cases"),
        (json.dumps([{"steps": [{"action": "click"}]}]).encode(), "x.json", "'title' is required"),
        (json.dumps([{"title": "t", "steps": []}]).encode(), "x.json", "'steps' is required"),
        (json.dumps([{"title": "t", "steps": [{"target": "id=x"}]}]).encode(), "x.json", "'action' is required"),
        (json.dumps([{"id": "A", "title": "t", "steps": [{"action": "open"}]}] * 2).encode(), "x.json", "used more than once"),
        (b"title,action\nx,click\n", "x.csv", "case_id"),
        (b"\xff\xfe\x00", "x.json", "UTF-8"),
    ],
)
def test_bad_input_is_explained(content: bytes, filename: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_test_data(content, filename)


def test_limits() -> None:
    with pytest.raises(ValidationError, match="larger than"):
        parse_test_data(b"x" * (MAX_TEST_DATA_BYTES + 1), "x.json")
    too_many_steps = [{"title": "t", "steps": [{"action": "click", "target": "id=x"}] * 201}]
    with pytest.raises(ValidationError, match="more than 200 steps"):
        parse_test_data(json.dumps(too_many_steps).encode(), "x.json")
    too_many_cases = [{"id": str(i), "title": "t", "steps": [{"action": "open", "target": "u"}]} for i in range(501)]
    with pytest.raises(ValidationError, match="more than 500"):
        parse_test_data(json.dumps(too_many_cases).encode(), "x.json")
