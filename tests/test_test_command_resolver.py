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


def test_chosen_java_test_files_become_classes() -> None:
    command = resolve_test_command(_sheet("Maven", ["TestNG"]), test_files=[
        "src/test/java/com/acme/LoginTest.java", "module-b/src/test/java/com/acme/cart/CartTests.java",
    ])

    assert command.command == (
        "mvn -B -ntp test -Dtest=com.acme.LoginTest,com.acme.cart.CartTests -Dsurefire.failIfNoSpecifiedTests=false"
    )


def test_chosen_features_run_through_cucumber_with_the_runner_class() -> None:
    command = resolve_test_command(_sheet("Maven", ["JUnit 4"], bdd="Cucumber"), test_files=[
        "src/test/resources/features/login.feature", "src/test/resources/features/my cart.feature",
        "src/test/java/com/acme/RunCucumberTest.java",
    ])

    assert command.command == (
        "mvn -B -ntp test -Dtest=com.acme.RunCucumberTest -Dsurefire.failIfNoSpecifiedTests=false "
        "'-Dcucumber.features=src/test/resources/features/login.feature,src/test/resources/features/my cart.feature'"
    )


def test_chosen_files_for_other_runners() -> None:
    files = ["tests/test_login.py", "tests/ui/test_cart.py"]
    assert resolve_test_command(_sheet("pip", ["pytest"]), test_files=files).command == (
        "python -m pytest -q --junitxml=ngauto-results/junit.xml tests/test_login.py tests/ui/test_cart.py"
    )
    assert resolve_test_command(_sheet("Gradle", ["JUnit 5"]), test_files=["src/test/kotlin/a/LoginTest.kt"]).command == (
        "gradle test --no-daemon --tests a.LoginTest"
    )
    specs = ["cypress/e2e/login.cy.ts", "cypress/e2e/cart.cy.ts"]
    assert "--spec cypress/e2e/login.cy.ts,cypress/e2e/cart.cy.ts" in resolve_test_command(
        _sheet("npm", ["Cypress"]), test_files=specs).command
    assert resolve_test_command(_sheet("npm", ["WebdriverIO", "Mocha"]), test_files=["test/a.spec.js", "test/b.spec.js"]
                                ).command.endswith("--spec test/a.spec.js --spec test/b.spec.js")


@pytest.mark.parametrize("files", [
    ["../etc/passwd"], ["/abs/LoginTest.java"], ["src/test/java/a;rm -rf/LoginTest.java"], ["-Dx=1"],
    ["src//LoginTest.java"], [f"t/test_{n}.py" for n in range(201)],
])
def test_unsafe_test_files_are_rejected(files: list[str]) -> None:
    with pytest.raises(ValidationError):
        resolve_test_command(_sheet("pip", ["pytest"]), test_files=files)


def test_files_that_the_runner_cannot_run_are_rejected() -> None:
    # A page object (a class without tests) is refused when the run is created, from its content (is_test_file);
    # here only what is not a class at all.
    with pytest.raises(ValidationError):
        resolve_test_command(_sheet("Maven", ["TestNG"]), test_files=["src/test/resources/config.properties"])
    with pytest.raises(ValidationError):
        resolve_test_command(_sheet("Maven", ["TestNG"]), test_files=["src/test/resources/login.feature"])  # no Cucumber
    with pytest.raises(ValidationError):
        resolve_test_command(_sheet("Maven", ["TestNG"]), "LoginTest", ["src/test/java/LoginTest.java"])  # both


@pytest.mark.parametrize(("path", "expected"), [
    ("src/test/java/com/acme/LoginTest.java", True),
    ("src/test/java/com/acme/TestLogin.java", True),
    ("src/test/java/com/acme/pages/LoginPage.java", False),
    ("src/main/java/com/acme/LoginTest.java", False),  # not in a test folder
    ("src/test/resources/features/login.feature", True),
    ("tests/test_login.py", True),
    ("tests/conftest.py", False),
    ("e2e/login.spec.ts", True),
    ("cypress/e2e/login.cy.js", True),
    ("suites/login.robot", True),
    ("package.json", False),
])
def test_is_test_file(path: str, expected: bool) -> None:
    from app.agents.RunnerAgent.TestCommandResolver import is_test_file

    assert is_test_file(path) is expected


@pytest.mark.parametrize(("content", "expected"), [
    ("public class BaseTest {\n  @BeforeMethod public void setUp() {}\n}\n", False),  # a base class: runs nothing
    ("public abstract class AbstractLoginTest {\n  @Test public void a() {}\n}\n", False),
    ("public class LoginTest extends BaseTest {\n  @Test public void valid() {}\n}\n", True),
    ("@RunWith(Cucumber.class)\n@CucumberOptions(features = \"src/test/resources\")\npublic class RunCucumberTest {}\n", True),
    ("public class RunnerTest extends AbstractTestNGCucumberTests {}\n", True),
    (None, False),  # unreadable
])
def test_jvm_test_classes_need_tests_inside(content: str | None, expected: bool) -> None:
    from app.agents.RunnerAgent.TestCommandResolver import is_test_file

    assert is_test_file("src/test/java/framework/SomeTest.java", lambda: content) is expected
    assert is_test_file("tests/test_login.py", lambda: None) is True  # only JVM classes are read


def test_a_runner_with_any_name_is_a_test_file_when_its_content_says_so() -> None:
    # A real run: the user told the generator to use Cucumber BDD and it wrote "Runner.java"; the API refused to run
    # it ("not a test file") although a chosen class runs by its name (-Dtest=com.demo.ngauto.Runner).
    from app.agents.RunnerAgent.TestCommandResolver import is_test_file

    runner = "@CucumberOptions(features = \"src/test/resources/features\")\npublic class Runner extends AbstractTestNGCucumberTests {}\n"
    page = "public class LoginPage {\n  public void login() {}\n}\n"

    assert is_test_file("src/test/java/com/demo/ngauto/Runner.java", lambda: runner) is True
    assert is_test_file("src/test/java/com/demo/pages/LoginPage.java", lambda: page) is False
    assert is_test_file("src/test/java/com/demo/ngauto/Runner.java") is False  # nothing to read: the name decides


@pytest.mark.parametrize(("frameworks", "build", "markers", "setup"), [
    (["pytest"], "pip", ["requirements.txt", "conftest.py"], "python -m pip install -q -r requirements.txt"),
    (["pytest"], "pip", ["requirements-test.txt", "requirements.txt"],
     "python -m pip install -q -r requirements-test.txt -r requirements.txt"),
    (["pytest"], "pip", ["pyproject.toml"], "python -m pip install -q -e ."),
    (["pytest"], "pip", ["sub/requirements.txt"], None),  # only what the project root declares
    (["Playwright Test"], "npm", ["package.json", "package-lock.json"], "npm ci && npx playwright install chromium"),
    (["Cypress"], "Yarn", ["package.json", "yarn.lock"], "npx --yes yarn install --frozen-lockfile"),
    (["Jest"], "npm", ["package.json"], "npm install"),
    (["TestNG"], "Maven", ["pom.xml"], None),  # Maven resolves its own dependencies
])
def test_declared_dependencies_are_installed_first(frameworks: list[str], build: str, markers: list[str], setup: str | None) -> None:
    assert resolve_test_command(_sheet(build, frameworks, markers=markers)).setup == setup
