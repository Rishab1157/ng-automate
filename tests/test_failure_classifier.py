import time

import pytest

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


def test_mostly_locator_failures_are_for_the_user() -> None:
    cases = [
        _case(TestCaseStatus.ERROR, "org.openqa.selenium.NoSuchElementException: no such element: Unable to locate element: {\"id\":\"cart\"}"),
        _case(TestCaseStatus.FAILED, "TimeoutException: waiting for visibility of element located by By.id: login"),
        _case(TestCaseStatus.FAILED, "expected [Dashboard] but found [Login]"),
    ]

    result = classify_failure(1, "Tests run: 3, Failures: 3", cases)

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



def test_undefined_cucumber_steps_are_a_build_problem_for_the_healer() -> None:
    output = (
        "[ERROR]   TestRunner>AbstractTestNGCucumberTests.runScenario:35 » UndefinedStep The step "
        "'the user opens \"https://www.saucedemo.com\"' and 4 other step(s) are undefined.\n[INFO] BUILD FAILURE"
    )
    cases = [TestCaseResultModel(name="runScenario", status=TestCaseStatus.FAILED,
                                 message="The step 'the user opens' and 4 other step(s) are undefined.")]

    result = classify_failure(1, output, cases)

    assert (result.kind, result.healable) == (FailureKind.BUILD_FAILURE, True)
