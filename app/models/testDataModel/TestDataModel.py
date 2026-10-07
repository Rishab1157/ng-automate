"""Test data: what the user wants tested. The test generator turns it into framework tests.

Locators come from the test data (V1). A step that needs a locator but has none is flagged,
never guessed.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TestDataFormat(str, Enum):
    JSON = "json"
    CSV = "csv"


class TestStepModel(BaseModel):
    action: str  # e.g. "open", "click", "type", "select", "verify_text", "wait"
    target: str | None = None  # locator (XPath, CSS, id=...) or URL for "open"
    value: str | None = None  # text to type / option to select
    expected: str | None = None  # expected text/state for verify steps
    needs_locator: bool = False  # set by the parser when the action needs a target and none was given


class TestCaseSpecModel(BaseModel):
    id: str
    title: str
    description: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    steps: list[TestStepModel]
    test_data: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class TestDataSetModel(BaseModel):
    source_format: TestDataFormat
    cases: list[TestCaseSpecModel]
    # Problems that did not stop parsing, e.g. "case LOGIN-2 step 3: click needs a locator".
    warnings: list[str] = Field(default_factory=list)
