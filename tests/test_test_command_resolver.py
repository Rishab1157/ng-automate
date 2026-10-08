import pytest

from app.agents.RunnerAgent.TestCommandResolver import UnsupportedStackError, resolve_test_command
from app.core.exceptions import ValidationError
from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, confidence=Confidence.HIGH if value else Confidence.LOW)


def _sheet(build=None, frameworks=None, tools=None, bdd=None, markers=()) -> FactSheetModel:
    return FactSheetModel(
        total_files=1, language_files={}, primary_language=_fact(None), build_tool=_fact(build),
        test_frameworks=_fact(frameworks), automation_tools=_fact(tools), bdd_tool=_fact(bdd),
        marker_files=list(markers), test_dirs=[], feature_file_count=0, top_level_tree=[],
    )


def test_maven() -> None:
    command = resolve_test_command(_sheet("Maven", ["TestNG"], ["Selenium"]))

    assert command.tool == "maven"
    assert command.command == "mvn -B -ntp test"
    assert command.report_globs == ["target/surefire-reports/TEST-*.xml", "target/failsafe-reports/TEST-*.xml"]


def test_maven_cucumber_tag_and_test_class() -> None:
    cucumber = _sheet("Maven", ["TestNG"], bdd="Cucumber")

    assert resolve_test_command(cucumber, "@smoke").command == "mvn -B -ntp test -Dcucumber.filter.tags=@smoke"
    assert resolve_test_command(cucumber, "LoginTest").command == "mvn -B -ntp test -Dtest=LoginTest"


def test_tag_with_spaces_is_quoted() -> None:
    command = resolve_test_command(_sheet("Maven", bdd="Cucumber"), "@smoke and @login")

    assert command.command == "mvn -B -ntp test '-Dcucumber.filter.tags=@smoke and @login'"


@pytest.mark.parametrize(("markers", "expected"), [(["gradlew", "build.gradle"], "./gradlew"), (["build.gradle"], "gradle")])
def test_gradle_wrapper_or_not(markers: list[str], expected: str) -> None:
    command = resolve_test_command(_sheet("Gradle", ["JUnit 5"], markers=markers))

    assert command.command.startswith(f"{expected} test --no-daemon")


def test_pytest_path_and_keyword() -> None:
    sheet = _sheet("pip", ["pytest"], ["Playwright"])

    assert resolve_test_command(sheet).command == "python -m pytest -q --junitxml=ngauto-results/junit.xml"
    assert resolve_test_command(sheet, "tests/test_login.py").command.endswith(" tests/test_login.py")
    assert resolve_test_command(sheet, "login").command.endswith(" -k login")


def test_playwright_and_cypress() -> None:
    playwright = resolve_test_command(_sheet("npm", ["Playwright Test"], ["Playwright"]))
    cypress = resolve_test_command(_sheet("npm", ["Cypress"], ["Cypress"]))

    assert playwright.command.startswith("PLAYWRIGHT_JUNIT_OUTPUT_NAME=ngauto-results/junit.xml npx playwright test")
    assert cypress.report_globs == ["ngauto-results/junit-*.xml"]


@pytest.mark.parametrize("selector", ["; rm -rf /", "$(id)", "`id`", "-Dmaven.repo.local=/x", "a|b", "x" * 201])
def test_unsafe_selectors_are_rejected(selector: str) -> None:
    with pytest.raises(ValidationError):
        resolve_test_command(_sheet("Maven", ["TestNG"]), selector)


def test_uft_and_unknown_stacks_are_unsupported() -> None:
    with pytest.raises(UnsupportedStackError, match="UFT"):
        resolve_test_command(_sheet(None, tools=["UFT"]))
    with pytest.raises(UnsupportedStackError):
        resolve_test_command(_sheet())


def test_first_build_tool_of_a_list_wins() -> None:
    assert resolve_test_command(_sheet(["Maven", "npm"], ["TestNG"])).tool == "maven"


HEADED = "${DISPLAY:+--headed}"


def test_browser_runners_are_headed_when_the_sandbox_has_a_display() -> None:
    playwright = resolve_test_command(_sheet("npm", ["Playwright Test"], ["Playwright"]))
    cypress = resolve_test_command(_sheet("npm", ["Cypress"], ["Cypress"]))
    plugin = resolve_test_command(_sheet("pip", ["pytest", "pytest-playwright"], ["Playwright"]))
    plain_pytest = resolve_test_command(_sheet("pip", ["pytest"], ["Selenium"]))

    assert HEADED in playwright.command and HEADED in cypress.command and HEADED in plugin.command
    # Without the pytest-playwright plugin, pytest would reject --headed.
    assert HEADED not in plain_pytest.command


def test_headed_flag_disappears_without_a_display() -> None:
    import subprocess

    command = resolve_test_command(_sheet("npm", ["Playwright Test"], ["Playwright"])).command
    flag_part = command.split(" npx ")[1]
    shown = subprocess.run(
        ["bash", "-c", f"echo {flag_part}"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
    ).stdout
    with_display = subprocess.run(
        ["bash", "-c", f"echo {flag_part}"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "DISPLAY": ":99"}
    ).stdout

    assert "--headed" not in shown and "--headed" in with_display


@pytest.mark.parametrize(
    ("markers", "config"),
    [([], "wdio.conf.js"), (["wdio.conf.ts"], "wdio.conf.ts"), (["config/wdio.android.conf.js"], "config/wdio.android.conf.js"),
     (["wdio.android.conf.ts", "wdio.conf.ts"], "wdio.conf.ts")],
)
def test_webdriverio_runs_with_wdio_even_when_mocha_is_listed(markers: list[str], config: str) -> None:
    command = resolve_test_command(_sheet("npm", ["WebdriverIO", "Mocha"], ["WebdriverIO"], markers=markers))

    assert command.tool == "webdriverio"
    assert command.command == f"npx wdio run {config}"
