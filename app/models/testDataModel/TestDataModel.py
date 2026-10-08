"""Test data: what the user wants tested. The test generator turns it into framework tests.

A data source is either our own JSON/CSV format (read at once) or free-form JSON / plain text that an LLM turns into
test cases in the background. A locator is never invented: a step either carries one from the source, names its
element in words (`target_hint`, resolved later from the project's page objects or the live page), or is flagged.
"""

from enum import Enum

from pydantic import BaseModel, Field


class TestDataFormat(str, Enum):
    __test__ = False  # not a pytest test class

    JSON = "json"
    CSV = "csv"
    TEXT = "text"


class TestDataStatus(str, Enum):
    """Our own JSON/CSV format is ready at once; free-form JSON and text are read by an LLM first."""

    __test__ = False

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class TestStepModel(BaseModel):
    __test__ = False

    action: str  # e.g. "open", "click", "type", "select", "verify_text", "wait"
    target: str | None = None  # locator (XPath, CSS, id=...) or URL for "open", only as the source gives it
    # The element in words when the source gives no locator, e.g. "Login button": found later, never guessed.
    target_hint: str | None = None
    value: str | None = None  # text to type / option to select
    expected: str | None = None  # expected text/state for verify steps
    text: str | None = None  # the step as the source wrote it
    needs_locator: bool = False  # the action needs an element and the source names none


class TestCaseSpecModel(BaseModel):
    __test__ = False

    id: str
    title: str
    description: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    steps: list[TestStepModel]
    test_data: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Where the case came from in a text source, e.g. "lines 3-9".
    source_ref: str | None = None


class TestDataSetModel(BaseModel):
    __test__ = False

    source_format: TestDataFormat
    cases: list[TestCaseSpecModel]
    # Problems that did not stop parsing, e.g. "case LOGIN-2 step 3: click needs a locator".
    warnings: list[str] = Field(default_factory=list)
