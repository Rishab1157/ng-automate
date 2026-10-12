import time

import pytest

from app.config import settings
from app.models.testRunModel import FailureKind, TestCaseResultModel, TestCaseStatus
from app.utils.FailureClassifier import classify_failure


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
])
def test_a_build_file_that_no_longer_parses_is_a_build_failure(output: str) -> None:
    result = classify_failure(1, output, [])

    assert (result.kind, result.healable) == (FailureKind.BUILD_FAILURE, True)
