import time
from pathlib import Path

import pytest

from app.agents.RunnerAgent.TestRunner import SETUP_NOTE
from app.config import settings
from app.models.testRunModel import FailureKind, TestCaseResultModel, TestCaseStatus
from app.utils.FailureClassifier import OWN_NOTE_PREFIX, classify_failure
from app.utils.TestReportParser import parse_report_file


def _case(status: TestCaseStatus, message: str | None = None) -> TestCaseResultModel:
    return TestCaseResultModel(name="t", status=status, message=message)


PASS = _case(TestCaseStatus.PASSED)


def test_passed() -> None:
    result = classify_failure(0, "Tests run: 3, Failures: 0", [PASS, PASS])

    assert result.kind == FailureKind.PASSED
    assert not result.healable


@pytest.mark.parametrize("output", ["collected 0 items", "Tests run: 0, Failures: 0", ""])
def test_no_tests(output: str) -> None:
    assert classify_failure(0, output, []).kind == FailureKind.NO_TESTS


@pytest.mark.parametrize(
    ("output", "kind"),
    [
        ("[ERROR] Failed to execute goal on project shop: Could not resolve dependencies for project com.demo:shop", FailureKind.DEPENDENCY_FAILURE),
        ("E   ModuleNotFoundError: No module named 'playwright'", FailureKind.DEPENDENCY_FAILURE),
        ("Error: Cannot find module '@playwright/test'", FailureKind.DEPENDENCY_FAILURE),
        ("[ERROR] COMPILATION ERROR :\n[ERROR] /src/LoginTest.java:[12,5] cannot find symbol", FailureKind.BUILD_FAILURE),
        ("ImportError while importing test module '/app/tests/test_x.py'", FailureKind.BUILD_FAILURE),
        ("Execution failed for task ':compileTestJava'.", FailureKind.BUILD_FAILURE),
        ("SessionNotCreatedException: session not created: This version of ChromeDriver only supports Chrome version 114", FailureKind.ENVIRONMENT_FAILURE),
        ("browserType.launch: Executable doesn't exist at /ms-playwright/chromium", FailureKind.ENVIRONMENT_FAILURE),
        ("[ERROR] Fatal error compiling: invalid target release: 21", FailureKind.ENVIRONMENT_FAILURE),
    ],
)
def test_technical_failures_go_to_the_healer(output: str, kind: FailureKind) -> None:
    result = classify_failure(1, output, [])

    assert result.kind == kind
    assert result.healable
    assert result.evidence


def test_dependency_beats_build_when_both_appear() -> None:
    output = "cannot find symbol\nCould not resolve dependencies for project x"

    assert classify_failure(1, output, []).kind == FailureKind.DEPENDENCY_FAILURE


LOCATOR_CASES = [
    _case(TestCaseStatus.ERROR, "org.openqa.selenium.NoSuchElementException: no such element: Unable to locate element: {\"id\":\"cart\"}"),
    _case(TestCaseStatus.FAILED, "TimeoutException: waiting for visibility of element located by By.id: login"),
    _case(TestCaseStatus.FAILED, "expected [Dashboard] but found [Login]"),
]


def test_mostly_locator_failures_go_to_the_healer() -> None:
    result = classify_failure(1, "Tests run: 3, Failures: 3", LOCATOR_CASES)

    assert result.kind == FailureKind.LOCATOR_FAILURE
    assert result.healable  # e.g. a locator the generator assumed: the healer finds the real one on the live page


def test_locator_failures_are_for_the_user_when_locator_healing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "HEAL_LOCATOR_FAILURES", False)

    result = classify_failure(1, "Tests run: 3, Failures: 3", LOCATOR_CASES)

    assert result.kind == FailureKind.LOCATOR_FAILURE
    assert not result.healable


def test_assertion_failures_are_test_failures() -> None:
    cases = [_case(TestCaseStatus.FAILED, "expected [Dashboard] but found [Login]"), PASS]

    result = classify_failure(1, "Tests run: 2, Failures: 1", cases)

    assert result.kind == FailureKind.TEST_FAILURE
    assert not result.healable


def test_unknown_failure() -> None:
    assert classify_failure(137, "Killed", []).kind == FailureKind.UNKNOWN


def test_huge_hostile_output_stays_fast() -> None:
    output = ("a" * 10_000 + "\n") * 500 + "x" * 1_000_000

    started = time.perf_counter()
    classify_failure(1, output, [])

    assert time.perf_counter() - started < 2



def test_network_failure_in_the_output_is_an_environment_problem_for_the_healer() -> None:
    output = (
        "io.github.bonigarcia.wdm.config.WebDriverManagerException: java.net.UnknownHostException: "
        "googlechromelabs.github.io\n[INFO] BUILD FAILURE"
    )

    result = classify_failure(1, output, [])

    assert (result.kind, result.healable) == (FailureKind.ENVIRONMENT_FAILURE, True)
    assert result.evidence


def test_network_failure_only_in_a_setup_case_goes_to_the_healer() -> None:
    # A real run: every test failed in @BeforeMethod with an UnknownHostException; the build output stayed clean.
    cases = [
        _thrown("setUp", "java.net.UnknownHostException: googlechromelabs.github.io",
                "io.github.bonigarcia.wdm.config.WebDriverManagerException: java.net.UnknownHostException: "
                "googlechromelabs.github.io\n\tat framework.BaseTest.setUp(BaseTest.java:25)"),
        _case(TestCaseStatus.SKIPPED, "java.net.UnknownHostException: googlechromelabs.github.io"),
    ]

    result = classify_failure(1, "Tests run: 3, Failures: 1, Skipped: 2", cases)

    assert (result.kind, result.healable) == (FailureKind.ENVIRONMENT_FAILURE, True)
    assert result.evidence == [
        # Where it happened in the project: the setup method that downloads the driver.
        "setUp: java.net.UnknownHostException: googlechromelabs.github.io (at framework.BaseTest.setUp(BaseTest.java:25))",
    ]



def test_undefined_cucumber_steps_are_a_build_problem_for_the_healer() -> None:
    output = (
        "[ERROR]   TestRunner>AbstractTestNGCucumberTests.runScenario:35 » UndefinedStep The step "
        "'the user opens \"https://www.saucedemo.com\"' and 4 other step(s) are undefined.\n[INFO] BUILD FAILURE"
    )
    cases = [TestCaseResultModel(name="runScenario", status=TestCaseStatus.FAILED,
                                 message="The step 'the user opens' and 4 other step(s) are undefined.")]

    result = classify_failure(1, output, cases)

    assert (result.kind, result.healable) == (FailureKind.BUILD_FAILURE, True)


def _thrown(name: str, message: str, details: str) -> TestCaseResultModel:
    return TestCaseResultModel(name=name, status=TestCaseStatus.FAILED, message=message, details=details)


def test_errors_thrown_by_the_test_code_go_to_the_healer() -> None:
    # As in a real run: generated tests passed a locator value where the project's helper expects a key.
    cases = [
        _thrown("testValidLogin", "Locator key not found: user-name",
                "java.lang.IllegalArgumentException: Locator key not found: user-name\n\tat framework.Config.locator(Config.java:31)"),
        _thrown("testEmptyPassword", "Locator key not found: login-button",
                "java.lang.IllegalArgumentException: Locator key not found: login-button\n\tat framework.Config.locator(Config.java:31)"),
        _case(TestCaseStatus.FAILED, "java.lang.AssertionError: expected [Products] but found [Swag Labs]"),
    ]

    result = classify_failure(1, "Tests run: 3, Failures: 3", cases)

    assert result.kind == FailureKind.TEST_CODE_FAILURE
    assert result.healable
    # With where it was thrown: the healer reads that code (here: the helper that reads locators.properties).
    assert result.evidence == [
        "testValidLogin: java.lang.IllegalArgumentException: Locator key not found: user-name"
        " (at framework.Config.locator(Config.java:31))",
        "testEmptyPassword: java.lang.IllegalArgumentException: Locator key not found: login-button"
        " (at framework.Config.locator(Config.java:31))",
    ]


def test_python_and_javascript_code_errors_are_recognised() -> None:
    cases = [
        _case(TestCaseStatus.FAILED, "AttributeError: 'NoneType' object has no attribute 'click'"),
        _case(TestCaseStatus.FAILED, "TypeError: page.clik is not a function"),
    ]

    assert classify_failure(1, "2 failed", cases).kind == FailureKind.TEST_CODE_FAILURE


def test_failed_checks_stay_with_the_user_even_with_a_code_error_among_them() -> None:
    cases = [
        _case(TestCaseStatus.FAILED, "java.lang.AssertionError: expected [1] but found [2]"),
        _case(TestCaseStatus.FAILED, "AssertionError: assert 'Products' == 'Swag Labs'"),
        _thrown("t3", "boom", "java.lang.NullPointerException: boom"),
    ]

    result = classify_failure(1, "Tests run: 3, Failures: 3", cases)

    assert result.kind == FailureKind.TEST_FAILURE
    assert not result.healable


def test_one_network_error_does_not_hide_failed_checks() -> None:
    cases = [
        _case(TestCaseStatus.FAILED, "java.lang.AssertionError: expected [Products] but found [Swag Labs]"),
        _case(TestCaseStatus.FAILED, "java.lang.AssertionError: expected [1] but found [0]"),
        _thrown("t3", "Connection refused", "java.net.ConnectException: Connection refused"),
    ]

    result = classify_failure(1, "Tests run: 3, Failures: 3", cases)

    assert result.kind == FailureKind.TEST_FAILURE
    assert not result.healable


def test_a_locator_failure_says_which_test_and_where_without_the_drivers_boilerplate() -> None:
    # As in a real run: Selenium's message carries build/system/driver info and a capabilities dump.
    case = TestCaseResultModel(
        name="removeProductFromCart", suite="com.demo.tests.ngauto.CartAndCheckoutTest", status=TestCaseStatus.FAILED,
        message="Expected condition failed: waiting for visibility of element located by By.cssSelector: "
                "[data-test='shopping-cart-badge'] (tried for 10 second(s) with 500 milliseconds interval)\n"
                "Build info: version: '4.27.0', revision: 'd6e718d134'\nSystem info: os.name: 'Linux'",
        details="org.openqa.selenium.TimeoutException: \nExpected condition failed: waiting for visibility\n"
                "Capabilities {acceptInsecureCerts: false, browserName: chrome}\n"
                "\tat org.openqa.selenium.support.ui.WebDriverWait.timeoutException(WebDriverWait.java:84)\n"
                "\tat org.openqa.selenium.support.ui.FluentWait.until(FluentWait.java:228)\n"
                "\tat com.demo.pages.BasePage.find(BasePage.java:24)\n"
                "\tat com.demo.pages.InventoryPage.cartCount(InventoryPage.java:55)\n"
                "\tat com.demo.tests.ngauto.CartAndCheckoutTest.removeProductFromCart(CartAndCheckoutTest.java:20)\n"
                "\tat java.base/jdk.internal.reflect.DirectMethodHandleAccessor.invoke(DirectMethodHandleAccessor.java:103)\n",
    )

    result = classify_failure(1, "Tests run: 12, Failures: 1", [case, *[PASS] * 11])

    assert result.kind == FailureKind.LOCATOR_FAILURE
    assert result.evidence == [
        "com.demo.tests.ngauto.CartAndCheckoutTest.removeProductFromCart: Expected condition failed: waiting for "
        "visibility of element located by By.cssSelector: [data-test='shopping-cart-badge'] (tried for 10 second(s) "
        "with 500 milliseconds interval) (at com.demo.pages.BasePage.find(BasePage.java:24) < at "
        "com.demo.pages.InventoryPage.cartCount(InventoryPage.java:55) < at "
        "com.demo.tests.ngauto.CartAndCheckoutTest.removeProductFromCart(CartAndCheckoutTest.java:20))",
    ]


def test_python_frames_point_at_the_projects_files_not_site_packages() -> None:
    case = TestCaseResultModel(
        name="test_cart", suite="tests.ngauto.test_cart", status=TestCaseStatus.FAILED,
        message="AttributeError: 'InventoryPage' object has no attribute 'open_cart'",
        details="Traceback (most recent call last):\n"
                '  File "/workspace/project/tests/ngauto/test_cart.py", line 9, in test_cart\n'
                '  File "/home/openhands/.local/lib/python3.13/site-packages/_pytest/python.py", line 159, in call\n',
    )

    result = classify_failure(1, "1 failed", [case])

    assert result.kind == FailureKind.TEST_CODE_FAILURE
    assert result.evidence == [
        "tests.ngauto.test_cart.test_cart: AttributeError: 'InventoryPage' object has no attribute 'open_cart' "
        '(File "/workspace/project/tests/ngauto/test_cart.py", line 9, in test_cart)',
    ]


def test_the_tests_own_frame_is_kept_behind_a_deep_helper_chain() -> None:
    # As in a real run: the test -> page object -> base page helpers -> wait (4 project frames).
    case = TestCaseResultModel(
        name="removeProductFromCart", suite="com.demo.tests.ngauto.CartAndCheckoutTest", status=TestCaseStatus.FAILED,
        message="Expected condition failed: waiting for visibility of element located by By.cssSelector: [data-test='badge']",
        details="org.openqa.selenium.TimeoutException: \n"
                "\tat org.openqa.selenium.support.ui.FluentWait.until(FluentWait.java:228)\n"
                "\tat com.demo.pages.BasePage.find(BasePage.java:23)\n"
                "\tat com.demo.pages.BasePage.textOf(BasePage.java:45)\n"
                "\tat com.demo.pages.InventoryPage.cartCount(InventoryPage.java:49)\n"
                "\tat com.demo.tests.ngauto.CartAndCheckoutTest.removeProductFromCart(CartAndCheckoutTest.java:20)\n",
    )

    [line] = classify_failure(1, "Tests run: 2, Failures: 1", [case, PASS]).evidence

    assert line.endswith(
        "(at com.demo.pages.BasePage.find(BasePage.java:23) < at com.demo.pages.BasePage.textOf(BasePage.java:45) < "
        "at com.demo.tests.ngauto.CartAndCheckoutTest.removeProductFromCart(CartAndCheckoutTest.java:20))"
    )


@pytest.mark.parametrize("output", [
    "[ERROR] [ERROR] Some problems were encountered while processing the POMs:\n"
    "[ERROR] Malformed POM /workspace/project/pom.xml: Unrecognised tag: 'dependency' (position: START_TAG seen ...",
    "[ERROR] Non-parseable POM /workspace/project/pom.xml: end tag name </dependencies> must match start tag",
    "* What went wrong:\nCould not compile build file '/workspace/project/build.gradle'.",
    "npm ERR! code EJSONPARSE\nnpm ERR! JSON.parse Failed to parse json",
    # Playwright 1.50 to 1.64 (at least) word it the same.
    "Error: Failed to load tsconfig file at /workspace/project/tsconfig.json:\nJSON5: invalid character ']' at 12:1",
])
def test_a_build_file_that_no_longer_parses_is_a_build_failure(output: str) -> None:
    result = classify_failure(1, output, [])

    assert (result.kind, result.healable) == (FailureKind.BUILD_FAILURE, True)


# Real JUnit reports and console output of fixtures/playwright_junit/demo.spec.ts, run in the agents' sandbox image
# with NG Automate's command (PLAYWRIGHT_JUNIT_OUTPUT_NAME=ngauto-results/junit.xml npx playwright test --reporter=junit).
# Up to Playwright 1.58 a failure's message is only the test's location and title ("demo.spec.ts:8:5 clicks a missing
# element") and the error is in the details; from 1.59 on, the message is the error.
PLAYWRIGHT_RUNS = Path(__file__).parent / "fixtures" / "playwright_junit"
PLAYWRIGHT_VERSIONS = ["1.50.1", "1.58.2", "1.64.1"]
PLAYWRIGHT_SETUP = SETUP_NOTE.format(command="npm ci && npx playwright install chromium")


def _playwright_output(name: str) -> str:
    return (PLAYWRIGHT_RUNS / name).read_text(encoding="utf-8")


def _playwright_case(version: str, name: str) -> TestCaseResultModel:
    return next(case for case in parse_report_file(PLAYWRIGHT_RUNS / f"{version}-junit.xml") if case.name == name)


@pytest.mark.parametrize("version", PLAYWRIGHT_VERSIONS)
def test_a_playwright_locator_timeout_goes_to_the_healer_with_its_error(version: str) -> None:
    case = _playwright_case(version, "clicks a missing element")

    result = classify_failure(1, _playwright_output(f"{version}-output.txt"), [case])

    assert (result.kind, result.healable) == (FailureKind.LOCATOR_FAILURE, True)
    # The healer reads only these lines for a locator failure: they hold the error, not just the test's name.
    [line] = result.evidence
    assert "locator.click: Timeout 2000ms exceeded." in line
    assert line.endswith("(at /workspace/project/tests/demo.spec.ts:10:41)")


@pytest.mark.parametrize("version", PLAYWRIGHT_VERSIONS)
def test_a_playwright_type_error_in_the_test_code_goes_to_the_healer(version: str) -> None:
    case = _playwright_case(version, "checkout › calls a method that does not exist")

    result = classify_failure(1, _playwright_output(f"{version}-output.txt"), [case])

    assert (result.kind, result.healable) == (FailureKind.TEST_CODE_FAILURE, True)
    assert "page.typeInto is not a function" in result.evidence[0]


@pytest.mark.parametrize("version", PLAYWRIGHT_VERSIONS)
def test_a_failed_playwright_check_stays_with_the_user_though_its_call_log_waits_for_a_locator(version: str) -> None:
    case = _playwright_case(version, "checkout › checks the wrong text")
    assert "waiting for locator('#title')" in (case.details or "")

    result = classify_failure(1, _playwright_output(f"{version}-output.txt"), [case])

    assert (result.kind, result.healable) == (FailureKind.TEST_FAILURE, False)
    assert "toHaveText" in result.evidence[0]


@pytest.mark.parametrize("version", PLAYWRIGHT_VERSIONS)
def test_a_playwright_run_after_ng_automates_setup_note_is_decided_by_its_tests(version: str) -> None:
    # As TestRunner builds it: NG Automate's note on the setup it ran, the setup's output, then the tests' output.
    tests_output = _playwright_output(f"{version}-output.txt")
    output = f"{PLAYWRIGHT_SETUP}added 3 packages in 2s\n{tests_output}"
    cases = parse_report_file(PLAYWRIGHT_RUNS / f"{version}-junit.xml")

    result = classify_failure(1, output, cases)

    # Two of the three failures are technical (a locator, the test code): the healer's job, not an environment one.
    assert (result.kind, result.healable) == (FailureKind.LOCATOR_FAILURE, True)
    assert "locator.click: Timeout 2000ms exceeded." in result.evidence[0]


def test_ng_automates_own_setup_note_is_never_read_as_the_projects_failure() -> None:
    # A real run's output: the setup note names "npx playwright install chromium", which made this run an environment
    # failure with the note as its only evidence; the real error is a tsconfig.json that does not parse.
    output = _playwright_output("tsconfig-error-output-tail.txt")
    assert output.startswith(PLAYWRIGHT_SETUP.strip())

    result = classify_failure(1, output, [])

    # A project file that no longer loads, for the healer (not a missing browser). The evidence is Playwright's error,
    # not the stack frames under it (node_modules/.../index.js:753:11 names a file and line, like a compiler error).
    assert (result.kind, result.healable) == (FailureKind.BUILD_FAILURE, True)
    assert result.evidence == ["Error: Failed to load tsconfig file at /workspace/project/tsconfig.json:"]


def test_a_missing_browser_is_still_found_after_ng_automates_setup_note() -> None:
    # A real run without the browser: Playwright's own output says "npx playwright install" too. Only our note is
    # skipped; the project's own lines still make it an environment failure.
    output = f"{PLAYWRIGHT_SETUP}added 3 packages in 2s\n{_playwright_output('1.58.2-no-browser-output.txt')}"

    result = classify_failure(1, output, [])

    assert (result.kind, result.healable) == (FailureKind.ENVIRONMENT_FAILURE, True)
    assert result.evidence[0].startswith("Error: browserType.launch: Executable doesn't exist at ")
    assert any("npx playwright install" in line for line in result.evidence)
    assert not any(line.startswith(OWN_NOTE_PREFIX) for line in result.evidence)


def test_a_message_that_starts_with_another_place_is_read_as_before() -> None:
    # Only a message that names the test itself (its location, then its own title) is replaced by the details' error.
    # This one starts with a file, line and column too (a type checker's error), but it is the error itself.
    error = "cart.ts:12:5 - error TS2339: Property 'total' does not exist on type 'Cart'."
    case = TestCaseResultModel(
        name="shows the cart total", suite="cart.test.ts", status=TestCaseStatus.FAILED, message=error,
        details=f"  ● shows the cart total\n\n    {error}",
    )

    result = classify_failure(1, "1 failed", [case])

    assert result.evidence == [f"cart.test.ts.shows the cart total: {error}"]
