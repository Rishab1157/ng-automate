"""Decide who handles a test run's outcome: the healer (technical failures, and locators that do not match the page
unless HEAL_LOCATOR_FAILURES is off) or the user (everything else).

Build output and test messages come from the project under test: they are untrusted and can be huge. Only the
tail of the output is scanned, and every pattern is a plain alternation without nested quantifiers (no
catastrophic backtracking).
"""

import re

from app.config import settings
from app.models.testRunModel import (
    HEALABLE_KINDS,
    FailureClassificationModel,
    FailureKind,
    TestCaseResultModel,
    TestCaseStatus,
)

MAX_SCANNED_CHARS = 200_000
MAX_EVIDENCE = 5
MAX_EVIDENCE_CHARS = 600
# Stack frames kept with an error thrown by the test code: where it was thrown.
MAX_STACK_FRAMES = 3

# The build file itself no longer loads (Maven, Gradle, npm, Playwright's tsconfig.json): nothing can even compile.
_BUILD_FILE_ERRORS = (
    r"Malformed POM|Non-parseable POM|problems were encountered while processing the POMs"
    r"|Could not compile (?:build|settings) file|EJSONPARSE|Failed to parse JSON|Failed to load tsconfig file"
)
BUILD_FILE_BROKEN = re.compile(_BUILD_FILE_ERRORS, re.IGNORECASE)

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
        # A build file that no longer loads (a broken edit): Maven, Gradle, npm.
        rf"|{_BUILD_FILE_ERRORS}"
        # BDD steps without step definitions (wrong glue path, or new feature files): missing code, not a test result.
        r"|UndefinedStep|step\(s\) are undefined|step is undefined|StepDefinitionNotFoundError|You can implement missing steps",
        re.IGNORECASE,
    )),
    (FailureKind.ENVIRONMENT_FAILURE, re.compile(
        r"SessionNotCreatedException|This version of ChromeDriver only supports|cannot find Chrome binary"
        r"|chromedriver.*(not found|unexpectedly exited)|Executable doesn't exist|npx playwright install"
        r"|JAVA_HOME|invalid target release|UnsupportedClassVersionError|: command not found|mvn: not found"
        r"|gradle: not found|DevToolsActivePort|Unable to obtain driver"
        # The sandbox could not reach the network (a driver/browser download, a package index): the tests never
        # ran, so this is the environment's problem, not a failed check.
        r"|UnknownHostException|NoRouteToHostException|ConnectException|SocketTimeoutException|UnknownHost"
        r"|Temporary failure in name resolution|Name or service not known|getaddrinfo|Network is unreachable"
        r"|Connection (?:refused|reset|timed out)|Read timed out|Failed to download|Could not download"
        r"|WebDriverManagerException|Selenium Manager",
        re.IGNORECASE,
    )),
]

# A compiler error that names its file and line: javac/Maven "File.java:[23,1]", javac/Gradle "File.java:23:", kotlinc
# "File.kt:23:5", tsc "file.ts(23,5)" or "file.ts:23:5", Python 'File "x.py", line 23'.
COMPILER_DIAGNOSTIC = re.compile(
    r"[\w./\\-]+\.(?:java|kt|kts|groovy|scala|ts|tsx|js|jsx|mjs|py|cs)(?::\[\d+,\d+\]|:\d+|\(\d+,\d+\)|\", line \d+)"
)
# A stack frame. JVM: "	at io.cucumber.core.runtime.TestCaseResultObserver.assert(TestCaseResultObserver.java:69)" or,
# with its module, "at java.base/java.lang.reflect.Method.invoke(Method.java:580)". JavaScript, with a space before the
# location or no function at all: "    at loadTsConfig (/workspace/project/node_modules/.../index.js:753:11)",
# "    at /workspace/project/tests/login.spec.ts:12:5".
STACK_FRAME = re.compile(r"\s*at\s+(?:[\w$.<>/@-]+\(|.+ \(.+:\d+:\d+\)\s*$|\S+:\d+:\d+\s*$)")

# A failing test's message that means the environment (network, driver download, toolchain) stopped the run before
# the test could check anything. Checked on the failing cases too: the build output may not repeat the message.
ENVIRONMENT_MESSAGE_PATTERN = re.compile(
    r"UnknownHostException|NoRouteToHostException|ConnectException|SocketTimeoutException|UnknownHost"
    r"|Temporary failure in name resolution|Name or service not known|getaddrinfo|Network is unreachable"
    r"|Connection (?:refused|reset|timed out)|Read timed out|Failed to download|Could not download"
    r"|WebDriverManagerException|Selenium Manager|Unable to obtain driver|SessionNotCreatedException",
    re.IGNORECASE,
)

# Test failure messages that mean "the element was not found": a locator or test data problem.
LOCATOR_PATTERN = re.compile(
    r"NoSuchElementException|no such element|Unable to locate element|ElementNotInteractableException"
    r"|StaleElementReferenceException|ElementClickInterceptedException|waiting for locator|locator\.\w+: Timeout"
    r"|Timed out retrying after|waiting for (visibility|presence|element)|located by By\."
    r"|strict mode violation",
    re.IGNORECASE,
)

# The first line of a failing test's error that names a programming error in the test code itself (not a check on the
# application): a wrong key or argument, null, a bad cast, a missing method or class.
CODE_ERROR_PATTERN = re.compile(
    r"\b(?:IllegalArgumentException|IllegalStateException|NullPointerException|ClassCastException"
    r"|NumberFormatException|(?:ArrayIndex|StringIndex|Index)OutOfBoundsException|UnsupportedOperationException"
    r"|NoSuchMethodError|NoSuchFieldError|NoClassDefFoundError|ClassNotFoundException|ExceptionInInitializerError"
    r"|AttributeError|NameError|UnboundLocalError|TypeError|KeyError|IndexError|ReferenceError)\b"
    r"|is not a function|Cannot read propert(?:y|ies) of (?:undefined|null)|fixture '[^']{1,100}' not found",
)

# Stack frames of the runtimes and libraries tests run on. The first frames outside them show where in the project a
# test failed (Java/Kotlin package frames; Python and JavaScript frames are told apart by their file paths).
LIBRARY_FRAME_PATTERN = re.compile(
    r"^at (?:java|javax|jdk|sun|com\.sun|kotlin|org\.openqa|org\.testng|org\.junit|junit|io\.cucumber|org\.apache"
    r"|org\.gradle|org\.assertj|org\.hamcrest|io\.restassured|org\.springframework)\."
    r"|site-packages|dist-packages|node_modules|<frozen |internal/"
)

NO_TESTS_PATTERN = re.compile(
    r"No tests to run|Tests run: 0,|collected 0 items|no tests ran|No tests found|0 passing",
    re.IGNORECASE,
)

# NG Automate's own notes in the output (TestRunner's: the setup it ran first, a timeout) start with this. They are not
# the project's output, so no pattern reads them as its failure: the setup note names its command, e.g. "npx playwright
# install chromium", which is no sign of a missing browser.
OWN_NOTE_PREFIX = "[NG Automate]"

# A failure message that only names the test: its file, line and column, then its title, e.g. "login.spec.ts:12:7 logs
# in". Playwright's JUnit report writes it up to 1.58 (from 1.59 on, the message is the error, unless the test failed
# without one). What went wrong is then in the details.
LOCATION_AND_TITLE = re.compile(r"([^:\n]+:\d+:\d+) (.+)")
# The call log under a Playwright error is not the error: it says "waiting for locator(...)" for failed checks too.
CALL_LOG_LINE = re.compile(r"Call log:|- waiting for ")

_REASONS = {
    FailureKind.PASSED: "All tests passed.",
    FailureKind.NO_TESTS: "The test command ran but no tests were executed.",
    FailureKind.DEPENDENCY_FAILURE: "A dependency is missing or could not be resolved.",
    FailureKind.BUILD_FAILURE: "The project does not compile.",
    FailureKind.ENVIRONMENT_FAILURE: "A browser, driver, JDK or tool is missing or does not match.",
    FailureKind.LOCATOR_FAILURE: "Tests could not find page elements: a locator does not match the page.",
    FailureKind.TEST_CODE_FAILURE: "Tests failed with errors in the test code itself (for example a wrong key or a null value).",
    FailureKind.TEST_FAILURE: "Tests ran and failed: assertions or the application's behaviour.",
    FailureKind.UNKNOWN: "The test command failed for an unknown reason.",
}


def classify_failure(exit_code: int, output: str, cases: list[TestCaseResultModel]) -> FailureClassificationModel:
    tail = output[-MAX_SCANNED_CHARS:]
    failing = [
        _with_error_message(case) for case in cases if case.status in (TestCaseStatus.FAILED, TestCaseStatus.ERROR)
    ]
    ran = [case for case in cases if case.status != TestCaseStatus.SKIPPED]

    if exit_code == 0 and not failing:
        if ran and not NO_TESTS_PATTERN.search(tail):
            return _result(FailureKind.PASSED)
        return _result(FailureKind.NO_TESTS, _matching_lines(tail, NO_TESTS_PATTERN))

    # Build/dependency/environment problems decide before test results: they explain the failures.
    for kind, pattern in OUTPUT_PATTERNS:
        lines = _matching_lines(tail, pattern)
        if lines:
            if kind == FailureKind.BUILD_FAILURE:
                # The compiler's own errors (file, line, message), not a "COMPILATION ERROR" header that every build
                # failure shares: they tell one build failure from the next (no-progress check, memory, the UI).
                lines = _compiler_errors(tail) or lines
            return _result(kind, lines)

    if failing:
        # The environment can stop every test in a setup method (driver download, network) without the build output
        # naming it: the failing cases' own messages still do. Those failures are technical, not failed checks; but
        # only when they are most of the failures (one test hitting a network error must not hide real failed checks).
        environment = [
            _failure_line(case)
            for case in failing
            if (case.message or case.details) and ENVIRONMENT_MESSAGE_PATTERN.search(f"{case.message or ''}\n{case.details or ''}")
        ]
        if len(environment) * 2 > len(failing):
            return _result(FailureKind.ENVIRONMENT_FAILURE, environment)

        locator: list[str] = []
        code: list[str] = []
        for case in failing:
            error = _code_error(case)
            if LOCATOR_PATTERN.search(f"{case.name}: {case.message or case.details or ''}"):
                locator.append(_failure_line(case))
            elif error:
                code.append(f"{_test_label(case)}: {error}")
        # Mostly technical problems in the tests (not checks that failed): the larger kind decides.
        if (len(locator) + len(code)) * 2 > len(failing):
            if len(code) > len(locator):
                return _result(FailureKind.TEST_CODE_FAILURE, code)
            return _result(FailureKind.LOCATOR_FAILURE, locator)
        return _result(FailureKind.TEST_FAILURE, [_failure_line(case) for case in failing])

    if NO_TESTS_PATTERN.search(tail):
        return _result(FailureKind.NO_TESTS, _matching_lines(tail, NO_TESTS_PATTERN))
    return _result(FailureKind.UNKNOWN, tail.strip().splitlines()[-3:])


def _with_error_message(case: TestCaseResultModel) -> TestCaseResultModel:
    """The failing test with its error as its message. When the message only names the test (LOCATION_AND_TITLE; the
    title is the end of the test's name, which also holds the names of its describe blocks), the error is the first
    line of the details that is not blank, not the header that repeats the test's location, and not its call log."""
    named = LOCATION_AND_TITLE.fullmatch(case.message or "")
    if named is None or not case.name.endswith(named.group(2)):
        return case
    location = named.group(1)
    lines = (line.strip() for line in (case.details or "").splitlines())
    error = next((line for line in lines if line and location not in line and not CALL_LOG_LINE.match(line)), None)
    return case.model_copy(update={"message": error}) if error else case


def _code_error(case: TestCaseResultModel) -> str | None:
    """The error line when a failing test threw a programming error (checked on the first line of the message and of
    the stack trace, where the exception's type is), else None. With where in the project it was thrown, e.g.
    "(at framework.Config.locator(Config.java:31) < at framework.BaseTest.byLocator(...))"."""
    for text in (case.message, case.details):
        first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
        if first and CODE_ERROR_PATTERN.search(first) and "AssertionError" not in first:
            return _with_frames(first, case)
    return None


def _failure_line(case: TestCaseResultModel) -> str:
    """One failing test as the healer reads it: which test (with its class or module), the first line of its message
    (the rest is often the driver's boilerplate), and where in the project it failed."""
    first = next((line.strip() for line in (case.message or case.details or "").splitlines() if line.strip()), "")
    return f"{_test_label(case)}: {_with_frames(first, case)}"


def _test_label(case: TestCaseResultModel) -> str:
    """e.g. "com.demo.tests.CartTest.removeProduct" (the class or module tells which file the test is in)."""
    if case.suite and not case.name.startswith(case.suite):
        return f"{case.suite}.{case.name}"
    return case.name


def _with_frames(line: str, case: TestCaseResultModel) -> str:
    frames = _project_frames(case)
    return f"{line} ({' < '.join(frames)})" if frames else line


def _project_frames(case: TestCaseResultModel) -> list[str]:
    """The first frames of the project's own code, always with the test's own frame (the line in the test file),
    which a deep helper chain would otherwise push out."""
    frames = [
        frame for frame in (raw.strip() for raw in (case.details or "").splitlines()[1:])
        if frame.startswith(("at ", 'File "')) and not LIBRARY_FRAME_PATTERN.search(frame)
    ]
    test = case.name.split("[", 1)[0]  # pytest parameters: test_login[admin]
    own = next((frame for frame in frames if f".{test}(" in frame or frame.endswith(f"in {test}")), None)
    chosen = frames[:MAX_STACK_FRAMES]
    if own is not None and own not in chosen:
        chosen = [*frames[: MAX_STACK_FRAMES - 1], own]
    return chosen


def _result(kind: FailureKind, evidence: list[str] | None = None) -> FailureClassificationModel:
    return FailureClassificationModel(
        kind=kind,
        healable=kind in HEALABLE_KINDS or (kind == FailureKind.LOCATOR_FAILURE and settings.HEAL_LOCATOR_FAILURES),
        reason=_REASONS[kind],
        evidence=[_clip(line.strip()) for line in (evidence or [])[:MAX_EVIDENCE]],
    )


def _compiler_errors(text: str) -> list[str]:
    """The first MAX_EVIDENCE compiler errors that name file and line, each once (Maven repeats them). A stack frame
    ("at io.cucumber...(Runner.java:69)") names a file and line too, but it is not a compiler error."""
    lines: list[str] = []
    for line in text.splitlines():
        if COMPILER_DIAGNOSTIC.search(line) and not STACK_FRAME.match(line) and line not in lines:
            lines.append(line)
            if len(lines) == MAX_EVIDENCE:
                break
    return lines


def _matching_lines(text: str, pattern: re.Pattern[str]) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if pattern.search(line) and not line.startswith(OWN_NOTE_PREFIX):
            lines.append(line)
            if len(lines) == MAX_EVIDENCE:
                break
    return lines


def _clip(text: str) -> str:
    return text if len(text) <= MAX_EVIDENCE_CHARS else text[: MAX_EVIDENCE_CHARS - 1] + "…"
