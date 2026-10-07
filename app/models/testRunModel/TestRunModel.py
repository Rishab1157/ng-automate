from enum import Enum

from pydantic import BaseModel, Field


class TestCaseStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TestCaseResultModel(BaseModel):
    """One test case from a report file (JUnit/Surefire/TestNG/pytest/Playwright XML)."""

    name: str
    suite: str | None = None
    status: TestCaseStatus
    duration_seconds: float | None = None
    # Failure/error message and stack trace, truncated.
    message: str | None = None
    details: str | None = None


class FailureKind(str, Enum):
    """Who handles a run's outcome. Healable kinds go to the healer; the rest are reported to the user."""

    PASSED = "passed"
    BUILD_FAILURE = "build_failure"  # compile/syntax errors -> healer
    DEPENDENCY_FAILURE = "dependency_failure"  # unresolved/missing packages, version conflicts -> healer
    ENVIRONMENT_FAILURE = "environment_failure"  # driver/browser/JDK/tool missing or mismatched -> healer
    LOCATOR_FAILURE = "locator_failure"  # element not found, locator timeouts -> user (test data)
    TEST_FAILURE = "test_failure"  # assertions, application behaviour -> user
    NO_TESTS = "no_tests"  # the command ran but executed no tests
    UNKNOWN = "unknown"


HEALABLE_KINDS = frozenset({FailureKind.BUILD_FAILURE, FailureKind.DEPENDENCY_FAILURE, FailureKind.ENVIRONMENT_FAILURE})


class FailureClassificationModel(BaseModel):
    kind: FailureKind
    healable: bool
    reason: str
    # The log lines or test messages that decided the kind.
    evidence: list[str] = Field(default_factory=list)


class TestCommandModel(BaseModel):
    """How to run a project's tests inside the sandbox and where the reports land."""

    tool: str  # e.g. "maven", "gradle", "pytest", "playwright"
    command: str  # shell command, run from working_dir
    working_dir: str = "."  # repo-relative
    report_globs: list[str] = Field(default_factory=list)  # repo-relative glob patterns


class TestRunResultModel(BaseModel):
    command: str
    exit_code: int
    duration_seconds: float
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    cases: list[TestCaseResultModel] = Field(default_factory=list)
    report_files: list[str] = Field(default_factory=list)
    # Last part of stdout+stderr, truncated.
    output_tail: str = ""
    classification: FailureClassificationModel
