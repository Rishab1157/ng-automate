"""Decide who handles a test run's outcome: the healer (technical failures) or the user (everything else).

Build output and test messages come from the project under test: they are untrusted and can be huge. Only the
tail of the output is scanned, and every pattern is a plain alternation without nested quantifiers (no
catastrophic backtracking).
"""

import re

from app.models.testRunModel import (
    HEALABLE_KINDS,
    FailureClassificationModel,
    FailureKind,
    TestCaseResultModel,
    TestCaseStatus,
)

MAX_SCANNED_CHARS = 200_000
MAX_EVIDENCE = 5
MAX_EVIDENCE_CHARS = 300

# Output patterns, checked in this order: the first kind with a matching line wins.
OUTPUT_PATTERNS: list[tuple[FailureKind, re.Pattern[str]]] = [
    (FailureKind.DEPENDENCY_FAILURE, re.compile(
        r"Could not resolve dependencies|Could not find artifact|Could not transfer artifact"
        r"|Could not resolve all (files|dependencies) for configuration|ModuleNotFoundError|No module named"
        r"|No matching distribution found|Cannot find module|npm ERR! 404|ERESOLVE|ERR_PNPM_",
        re.IGNORECASE,
    )),
    (FailureKind.BUILD_FAILURE, re.compile(
        r"COMPILATION ERROR|cannot find symbol|Compilation failed|error: ';' expected|SyntaxError"
        r"|IndentationError|error TS\d+|Execution failed for task ':[\w:]*compile"
        r"|ImportError while importing test module|package [\w.]+ does not exist"
        # BDD steps without step definitions (wrong glue path, or new feature files): missing code, not a test result.
        r"|UndefinedStep|step\(s\) are undefined|step is undefined|StepDefinitionNotFoundError|You can implement missing steps",
        re.IGNORECASE,
    )),
    (FailureKind.ENVIRONMENT_FAILURE, re.compile(
        r"SessionNotCreatedException|This version of ChromeDriver only supports|cannot find Chrome binary"
        r"|chromedriver.*(not found|unexpectedly exited)|Executable doesn't exist|npx playwright install"
        r"|JAVA_HOME|invalid target release|UnsupportedClassVersionError|: command not found|mvn: not found"
        r"|gradle: not found|DevToolsActivePort|Unable to obtain driver",
        re.IGNORECASE,
    )),
]

# Test failure messages that mean "the element was not found": a locator or test data problem.
LOCATOR_PATTERN = re.compile(
    r"NoSuchElementException|no such element|Unable to locate element|ElementNotInteractableException"
    r"|StaleElementReferenceException|ElementClickInterceptedException|waiting for locator|locator\.\w+: Timeout"
    r"|Timed out retrying after|waiting for (visibility|presence|element)|located by By\."
    r"|strict mode violation",
    re.IGNORECASE,
)

NO_TESTS_PATTERN = re.compile(
    r"No tests to run|Tests run: 0,|collected 0 items|no tests ran|No tests found|0 passing",
    re.IGNORECASE,
)

_REASONS = {
    FailureKind.PASSED: "All tests passed.",
    FailureKind.NO_TESTS: "The test command ran but no tests were executed.",
    FailureKind.DEPENDENCY_FAILURE: "A dependency is missing or could not be resolved.",
    FailureKind.BUILD_FAILURE: "The project does not compile.",
    FailureKind.ENVIRONMENT_FAILURE: "A browser, driver, JDK or tool is missing or does not match.",
    FailureKind.LOCATOR_FAILURE: "Tests could not find page elements: check the locators in the test data.",
    FailureKind.TEST_FAILURE: "Tests ran and failed: assertions or the application's behaviour.",
    FailureKind.UNKNOWN: "The test command failed for an unknown reason.",
}


def classify_failure(exit_code: int, output: str, cases: list[TestCaseResultModel]) -> FailureClassificationModel:
    tail = output[-MAX_SCANNED_CHARS:]
    failing = [case for case in cases if case.status in (TestCaseStatus.FAILED, TestCaseStatus.ERROR)]
    ran = [case for case in cases if case.status != TestCaseStatus.SKIPPED]

    if exit_code == 0 and not failing:
        if ran and not NO_TESTS_PATTERN.search(tail):
            return _result(FailureKind.PASSED)
        return _result(FailureKind.NO_TESTS, _matching_lines(tail, NO_TESTS_PATTERN))

    # Build/dependency/environment problems decide before test results: they explain the failures.
    for kind, pattern in OUTPUT_PATTERNS:
        lines = _matching_lines(tail, pattern)
        if lines:
            return _result(kind, lines)

    if failing:
        messages = [f"{case.name}: {case.message or case.details or ''}" for case in failing]
        locator = [m for m in messages if LOCATOR_PATTERN.search(m)]
        if len(locator) * 2 > len(messages):
            return _result(FailureKind.LOCATOR_FAILURE, locator)
        return _result(FailureKind.TEST_FAILURE, messages)

    if NO_TESTS_PATTERN.search(tail):
        return _result(FailureKind.NO_TESTS, _matching_lines(tail, NO_TESTS_PATTERN))
    return _result(FailureKind.UNKNOWN, tail.strip().splitlines()[-3:])


def _result(kind: FailureKind, evidence: list[str] | None = None) -> FailureClassificationModel:
    return FailureClassificationModel(
        kind=kind,
        healable=kind in HEALABLE_KINDS,
        reason=_REASONS[kind],
        evidence=[_clip(line.strip()) for line in (evidence or [])[:MAX_EVIDENCE]],
    )


def _matching_lines(text: str, pattern: re.Pattern[str]) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if pattern.search(line):
            lines.append(line)
            if len(lines) == MAX_EVIDENCE:
                break
    return lines


def _clip(text: str) -> str:
    return text if len(text) <= MAX_EVIDENCE_CHARS else text[: MAX_EVIDENCE_CHARS - 1] + "…"
