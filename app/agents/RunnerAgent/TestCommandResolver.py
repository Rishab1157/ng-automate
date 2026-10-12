"""How to run a project's tests inside the sandbox, decided from its fact sheet (plain code, no LLM).

Each command writes machine-readable reports where TestReportParser can find them. The optional test selector
comes from users (a test class, file or Cucumber tag expression): it is validated and shell-quoted.
"""

import re
import shlex
from collections.abc import Callable
from pathlib import Path

from app.core.exceptions import ErrorCode, ValidationError
from app.models.analyzerModel import FactSheetModel
from app.models.testRunModel import TestCommandModel

RESULTS_DIR = "ngauto-results"
_SELECTOR_PATTERN = re.compile(r"^(?!-)[A-Za-z0-9_.:/@#*,\- ]{1,200}$")
MAX_TEST_FILES = 200
_TEST_FILE_PATH = re.compile(r"^(?![-/])[A-Za-z0-9_.@/\- ]{1,300}$")
# Files a test runner picks up: Surefire's default class names, Cucumber features, pytest, JavaScript specs, Robot.
_JVM_TEST_NAME = re.compile(r"(^Test\w*|\w*(Test|Tests|TestCase))\.(java|kt)$")
_OTHER_TEST_NAME = re.compile(r"\.feature$|^test_.*\.py$|_test\.py$|\.(spec|test|cy)\.(ts|tsx|js|jsx|mjs|cjs)$|\.robot$")
_JVM_TEST_DIRS = {"test", "tests", "it", "integrationTest"}
# What makes a JVM class runnable as tests: JUnit 4/5 and TestNG test methods, Cucumber and JUnit runner classes.
_JVM_TEST_MARKERS = re.compile(
    r"@(Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate|CucumberOptions|Suite|RunWith|Cucumber)\b"
    r"|extends\s+(AbstractTestNGCucumberTests|TestCase)\b"
)
_ABSTRACT_CLASS = re.compile(r"\babstract\s+class\b")

INVALID_TEST_SELECTOR = "Invalid test selector: use letters, digits and _ . : / @ # * , - only"
INVALID_TEST_FILE = "Invalid test file path: {path}"
TOO_MANY_TEST_FILES = f"Choose at most {MAX_TEST_FILES} test files"
SELECTOR_AND_FILES = "Give either a test selector or test files, not both"
NOT_A_TEST_FILE = "{path} is not a test file this project's test runner can run"
UNSUPPORTED_UFT = "UFT/QTP projects need Windows and a UFT licence: they cannot run in the Linux sandbox"
UNSUPPORTED_STACK = "Could not tell how to run this project's tests from its build files"

# Surefire/Failsafe write TEST-*.xml for JUnit AND TestNG; for TestNG they also write testng-results.xml with the
# same tests, so reading both would count every test twice.
MAVEN_REPORTS = [
    "target/surefire-reports/TEST-*.xml",
    "target/failsafe-reports/TEST-*.xml",
]
GRADLE_REPORTS = ["**/build/test-results/**/TEST-*.xml"]
JUNIT_RESULT = f"{RESULTS_DIR}/junit.xml"
# Browsers show on the sandbox's display (the live view) when there is one. The shell drops the flag when DISPLAY is
# unset (the display could not start), so the tests then run headless instead of failing.
HEADED = "${DISPLAY:+--headed}"
DEFAULT_WDIO_CONFIG = "wdio.conf.js"


class UnsupportedStackError(ValidationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, error_code=ErrorCode.INVALID_INPUT)


def resolve_test_command(
    fact_sheet: FactSheetModel, test_selector: str | None = None, test_files: list[str] | None = None
) -> TestCommandModel:
    """The command that runs the project's tests: all of them, the selector's, or just the given test files."""
    selector = check_test_selector(test_selector)
    files = check_test_files(test_files)
    if selector and files:
        raise ValidationError(SELECTOR_AND_FILES)
    build_tool = _first(fact_sheet.build_tool.value)
    frameworks = _values(fact_sheet.test_frameworks.value)
    tools = _values(fact_sheet.automation_tools.value)
    bdd = _first(fact_sheet.bdd_tool.value)
    markers = {marker.rsplit("/", 1)[-1] for marker in fact_sheet.marker_files}

    if "UFT" in tools:
        raise UnsupportedStackError(UNSUPPORTED_UFT)
    if build_tool == "Maven":
        return _maven(selector, bdd == "Cucumber", files)
    if build_tool == "Gradle":
        return _gradle(selector, "gradlew" in markers, files)
    paths = " ".join(_quoted(path) for path in files) if files else (_quoted(selector) if selector else "")
    if "pytest" in frameworks:
        return _pytest(selector, headed="pytest-playwright" in frameworks, paths=paths if files else None)
    if "Robot Framework" in frameworks:
        return TestCommandModel(
            tool="robot",
            command=f"python -m robot --xunit {RESULTS_DIR}/xunit.xml {paths or '.'}",
            report_globs=[f"{RESULTS_DIR}/xunit.xml"],
        )
    if "Playwright Test" in frameworks:
        return TestCommandModel(
            tool="playwright",
            command=f"PLAYWRIGHT_JUNIT_OUTPUT_NAME={JUNIT_RESULT} npx playwright test --reporter=junit {HEADED}"
            + (f" {paths}" if paths else ""),
            report_globs=[JUNIT_RESULT],
        )
    if "Cypress" in frameworks:
        specs = ",".join(files) if files else selector
        spec = f" --spec {_quoted(specs)}" if specs else ""
        return TestCommandModel(
            tool="cypress",
            command=f"npx cypress run {HEADED} --reporter junit "
            f"--reporter-options mochaFile={RESULTS_DIR}/junit-[hash].xml{spec}",
            report_globs=[f"{RESULTS_DIR}/junit-*.xml"],
        )
    if "WebdriverIO" in frameworks:
        # Before Jest/Mocha: a WebdriverIO project also lists the runner framework it uses (often Mocha).
        spec = "".join(f" --spec {_quoted(item)}" for item in (files or ([selector] if selector else [])))
        return TestCommandModel(tool="webdriverio", command=f"npx wdio run {_quoted(_wdio_config(fact_sheet.marker_files))}{spec}")
    for framework, command in (("Jest", "npx jest --ci"), ("Mocha", "npx mocha")):
        if framework in frameworks:
            return TestCommandModel(tool=framework.lower(), command=command + (f" {paths}" if paths else ""))
    if build_tool in ("npm", "Yarn", "pnpm"):
        if files:
            raise ValidationError(NOT_A_TEST_FILE.format(path=files[0]))
        return TestCommandModel(tool=build_tool.lower(), command=f"{build_tool.lower()} test")
    raise UnsupportedStackError(UNSUPPORTED_STACK)


def _maven(selector: str | None, cucumber: bool, files: list[str] | None = None) -> TestCommandModel:
    command = "mvn -B -ntp test"
    if files:
        features = [path for path in files if path.endswith(".feature")]
        classes = [_jvm_class(path) for path in files if not path.endswith(".feature")]
        if features and not cucumber:
            raise ValidationError(NOT_A_TEST_FILE.format(path=features[0]))
        if classes:
            # In a multi-module build, modules without the chosen classes must not fail the build.
            command += f" {_quoted('-Dtest=' + ','.join(classes))} -Dsurefire.failIfNoSpecifiedTests=false"
        if features:
            # Cucumber's own property: its runner class runs just these features.
            command += f" {_quoted('-Dcucumber.features=' + ','.join(features))}"
    elif selector:
        option = "cucumber.filter.tags" if cucumber and selector.startswith("@") else "test"
        command += f" {_quoted(f'-D{option}={selector}')}"
    return TestCommandModel(tool="maven", command=command, report_globs=MAVEN_REPORTS)


def _gradle(selector: str | None, has_wrapper: bool, files: list[str] | None = None) -> TestCommandModel:
    command = f"{'./gradlew' if has_wrapper else 'gradle'} test --no-daemon"
    if files:
        for path in files:
            if path.endswith(".feature"):
                raise ValidationError(NOT_A_TEST_FILE.format(path=path))
            command += f" --tests {_quoted(_jvm_class(path))}"
    elif selector:
        command += f" --tests {_quoted(selector)}"
    return TestCommandModel(tool="gradle", command=command, report_globs=GRADLE_REPORTS)


def _jvm_class(path: str) -> str:
    """src/test/java/com/acme/LoginTest.java -> com.acme.LoginTest (just the class name outside a java/kotlin root)."""
    if not _JVM_TEST_NAME.search(path.rsplit("/", 1)[-1]):
        raise ValidationError(NOT_A_TEST_FILE.format(path=path))
    parts = path.rsplit(".", 1)[0].split("/")
    roots = [index for index, part in enumerate(parts[:-1]) if part in ("java", "kotlin")]
    return ".".join(parts[roots[-1] + 1:]) if roots else parts[-1]


def is_test_file(path: str, read: Callable[[], str | None] | None = None) -> bool:
    """A file a test runner picks up (project-relative, "/"-separated): what users can choose to run.

    With `read` (the file's text, read only when needed), a Java/Kotlin class counts only when it has tests or is a
    test runner and is not abstract: a base class such as BaseTest is named like a test but runs nothing.
    """
    parts = path.split("/")
    name = parts[-1]
    if _JVM_TEST_NAME.search(name):
        if not any(part in _JVM_TEST_DIRS for part in parts[:-1]):
            return False
        if read is None:
            return True
        text = read()
        return text is not None and bool(_JVM_TEST_MARKERS.search(text)) and not _ABSTRACT_CLASS.search(text)
    return bool(_OTHER_TEST_NAME.search(name))


def read_small_text(path: Path, limit: int = 512 * 1024) -> str | None:
    """A file's text for is_test_file (None when it is missing, too large or not text)."""
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def check_test_files(paths: list[str] | None) -> list[str] | None:
    """The chosen test files, "/"-separated and without duplicates, None when empty; ValidationError when unsafe."""
    if not paths:
        return None
    if len(paths) > MAX_TEST_FILES:
        raise ValidationError(TOO_MANY_TEST_FILES)
    files: list[str] = []
    for raw in paths:
        path = raw.replace("\\", "/").strip()
        if not _TEST_FILE_PATH.match(path) or ".." in path.split("/") or "//" in path:
            raise ValidationError(INVALID_TEST_FILE.format(path=raw[:300]))
        if path not in files:
            files.append(path)
    return files


def _wdio_config(marker_files: list[str]) -> str:
    """The project's WebdriverIO config (repo-relative), e.g. wdio.conf.ts or config/wdio.android.conf.js.

    The plain wdio.conf.* first, then the shallowest one.
    """
    def name(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    configs = [path for path in marker_files if name(path).startswith("wdio.") and ".conf." in name(path)]
    configs.sort(key=lambda path: (not name(path).startswith("wdio.conf."), path.count("/"), path))
    return configs[0] if configs else DEFAULT_WDIO_CONFIG


def _pytest(selector: str | None, headed: bool = False, paths: str | None = None) -> TestCommandModel:
    command = f"python -m pytest -q --junitxml={JUNIT_RESULT}"
    if headed:
        # Only with the pytest-playwright plugin: without it pytest rejects the unknown option.
        command += f" {HEADED}"
    if paths:
        command += f" {paths}"
    elif selector:
        # A path selects files; anything else is a -k expression.
        command += f" {_quoted(selector)}" if "/" in selector or selector.endswith(".py") else f" -k {_quoted(selector)}"
    return TestCommandModel(tool="pytest", command=command, report_globs=[JUNIT_RESULT])


def check_test_selector(selector: str | None) -> str | None:
    """The selector trimmed, None when empty; raises ValidationError when it has characters that are not allowed."""
    if selector is None or not selector.strip():
        return None
    selector = selector.strip()
    if not _SELECTOR_PATTERN.match(selector):
        raise ValidationError(INVALID_TEST_SELECTOR)
    return selector


def _quoted(value: str) -> str:
    return shlex.quote(value)


def _values(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _first(value: str | list[str] | None) -> str | None:
    values = _values(value)
    return values[0] if values else None
