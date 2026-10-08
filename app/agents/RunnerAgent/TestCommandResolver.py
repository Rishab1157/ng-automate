"""How to run a project's tests inside the sandbox, decided from its fact sheet (plain code, no LLM).

Each command writes machine-readable reports where TestReportParser can find them. The optional test selector
comes from users (a test class, file or Cucumber tag expression): it is validated and shell-quoted.
"""

import re
import shlex

from app.core.exceptions import ErrorCode, ValidationError
from app.models.analyzerModel import FactSheetModel
from app.models.testRunModel import TestCommandModel

RESULTS_DIR = "ngauto-results"
_SELECTOR_PATTERN = re.compile(r"^(?!-)[A-Za-z0-9_.:/@#*,\- ]{1,200}$")

INVALID_TEST_SELECTOR = "Invalid test selector: use letters, digits and _ . : / @ # * , - only"
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


def resolve_test_command(fact_sheet: FactSheetModel, test_selector: str | None = None) -> TestCommandModel:
    selector = check_test_selector(test_selector)
    build_tool = _first(fact_sheet.build_tool.value)
    frameworks = _values(fact_sheet.test_frameworks.value)
    tools = _values(fact_sheet.automation_tools.value)
    bdd = _first(fact_sheet.bdd_tool.value)
    markers = {marker.rsplit("/", 1)[-1] for marker in fact_sheet.marker_files}

    if "UFT" in tools:
        raise UnsupportedStackError(UNSUPPORTED_UFT)
    if build_tool == "Maven":
        return _maven(selector, bdd == "Cucumber")
    if build_tool == "Gradle":
        return _gradle(selector, "gradlew" in markers)
    if "pytest" in frameworks:
        return _pytest(selector, headed="pytest-playwright" in frameworks)
    if "Robot Framework" in frameworks:
        return TestCommandModel(
            tool="robot",
            command=f"python -m robot --xunit {RESULTS_DIR}/xunit.xml {_quoted(selector) if selector else '.'}",
            report_globs=[f"{RESULTS_DIR}/xunit.xml"],
        )
    if "Playwright Test" in frameworks:
        return TestCommandModel(
            tool="playwright",
            command=f"PLAYWRIGHT_JUNIT_OUTPUT_NAME={JUNIT_RESULT} npx playwright test --reporter=junit {HEADED}"
            + (f" {_quoted(selector)}" if selector else ""),
            report_globs=[JUNIT_RESULT],
        )
    if "Cypress" in frameworks:
        spec = f" --spec {_quoted(selector)}" if selector else ""
        return TestCommandModel(
            tool="cypress",
            command=f"npx cypress run {HEADED} --reporter junit "
            f"--reporter-options mochaFile={RESULTS_DIR}/junit-[hash].xml{spec}",
            report_globs=[f"{RESULTS_DIR}/junit-*.xml"],
        )
    if "WebdriverIO" in frameworks:
        # Before Jest/Mocha: a WebdriverIO project also lists the runner framework it uses (often Mocha).
        spec = f" --spec {_quoted(selector)}" if selector else ""
        return TestCommandModel(tool="webdriverio", command=f"npx wdio run {_quoted(_wdio_config(fact_sheet.marker_files))}{spec}")
    for framework, command in (("Jest", "npx jest --ci"), ("Mocha", "npx mocha")):
        if framework in frameworks:
            return TestCommandModel(tool=framework.lower(), command=command + (f" {_quoted(selector)}" if selector else ""))
    if build_tool in ("npm", "Yarn", "pnpm"):
        return TestCommandModel(tool=build_tool.lower(), command=f"{build_tool.lower()} test")
    raise UnsupportedStackError(UNSUPPORTED_STACK)


def _maven(selector: str | None, cucumber: bool) -> TestCommandModel:
    command = "mvn -B -ntp test"
    if selector:
        option = "cucumber.filter.tags" if cucumber and selector.startswith("@") else "test"
        command += f" {_quoted(f'-D{option}={selector}')}"
    return TestCommandModel(tool="maven", command=command, report_globs=MAVEN_REPORTS)


def _gradle(selector: str | None, has_wrapper: bool) -> TestCommandModel:
    command = f"{'./gradlew' if has_wrapper else 'gradle'} test --no-daemon"
    if selector:
        command += f" --tests {_quoted(selector)}"
    return TestCommandModel(tool="gradle", command=command, report_globs=GRADLE_REPORTS)


def _wdio_config(marker_files: list[str]) -> str:
    """The project's WebdriverIO config (repo-relative), e.g. wdio.conf.ts or config/wdio.android.conf.js.

    The plain wdio.conf.* first, then the shallowest one.
    """
    def name(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    configs = [path for path in marker_files if name(path).startswith("wdio.") and ".conf." in name(path)]
    configs.sort(key=lambda path: (not name(path).startswith("wdio.conf."), path.count("/"), path))
    return configs[0] if configs else DEFAULT_WDIO_CONFIG


def _pytest(selector: str | None, headed: bool = False) -> TestCommandModel:
    command = f"python -m pytest -q --junitxml={JUNIT_RESULT}"
    if headed:
        # Only with the pytest-playwright plugin: without it pytest rejects the unknown option.
        command += f" {HEADED}"
    if selector:
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
