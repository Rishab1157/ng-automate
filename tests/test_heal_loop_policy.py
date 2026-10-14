"""HealLoopPolicy: when the healer gets another try, when the test phase stops, and how it is reported."""

import pytest

from app.agents.HealerAgent.HealLoopPolicy import (
    build_test_report,
    describe_test_report,
    failure_signature,
    got_better,
    got_worse,
    same_failure_streak,
    stop_reason,
)
from app.models.healerModel import FileChangeModel, HealOutcomeModel
from app.models.runModel import TestAttemptModel, TestStopReason
from app.models.testRunModel import HEALABLE_KINDS, FailureClassificationModel, FailureKind, TestRunResultModel

SAME_FAILURE_LIMIT = 3
DEPENDENCY_EVIDENCE = ["Could not find artifact org.testng:testng:jar:99.0.0"]


def _result(kind: FailureKind, evidence: list[str] | None = None, passed: int = 0, failed: int = 0) -> TestRunResultModel:
    return TestRunResultModel(
        command="mvn -B -ntp test", exit_code=0 if kind == FailureKind.PASSED else 1, duration_seconds=10.0,
        total=passed + failed, passed=passed, failed=failed, errors=0, skipped=0,
        classification=FailureClassificationModel(
            kind=kind, healable=kind in HEALABLE_KINDS, reason="reason", evidence=evidence or []
        ),
    )


def _fix(*paths: str) -> HealOutcomeModel:
    return HealOutcomeModel(summary="fixed", changes=[FileChangeModel(path=p, change="modified") for p in paths])


def _attempt(number: int, result: TestRunResultModel, heal: HealOutcomeModel | None = None,
             state: str | None = None) -> TestAttemptModel:
    return TestAttemptModel(number=number, result=result, heal=heal, project_state=state)


DEPENDENCY = _result(FailureKind.DEPENDENCY_FAILURE, DEPENDENCY_EVIDENCE)
BUILD = _result(FailureKind.BUILD_FAILURE, ["LoginPage.java:[12,8] cannot find symbol"])


def test_passing_run_stops() -> None:
    assert stop_reason([_attempt(1, _result(FailureKind.PASSED, passed=3))], SAME_FAILURE_LIMIT) == TestStopReason.PASSED


@pytest.mark.parametrize("kind", [FailureKind.TEST_FAILURE, FailureKind.LOCATOR_FAILURE, FailureKind.NO_TESTS, FailureKind.UNKNOWN])
def test_failures_for_the_user_are_not_healed(kind: FailureKind) -> None:
    assert stop_reason([_attempt(1, _result(kind))], SAME_FAILURE_LIMIT) == TestStopReason.NOT_HEALABLE


def test_technical_failure_is_healed() -> None:
    assert stop_reason([_attempt(1, DEPENDENCY)], SAME_FAILURE_LIMIT) is None


def test_one_failed_fix_is_not_the_end_the_healer_tries_another_way() -> None:
    attempts = [_attempt(1, DEPENDENCY, _fix("pom.xml")), _attempt(2, DEPENDENCY)]

    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


def test_the_same_failure_too_many_runs_in_a_row_is_no_progress() -> None:
    attempts = [_attempt(1, DEPENDENCY, _fix("pom.xml")), _attempt(2, DEPENDENCY, _fix("pom.xml")), _attempt(3, DEPENDENCY)]

    assert same_failure_streak(attempts) == 3
    assert stop_reason(attempts, SAME_FAILURE_LIMIT) == TestStopReason.NO_PROGRESS


def test_a_new_failure_after_a_fix_is_progress() -> None:
    attempts = [_attempt(1, DEPENDENCY, _fix("pom.xml")), _attempt(2, DEPENDENCY, _fix("pom.xml")), _attempt(3, BUILD)]

    assert same_failure_streak(attempts) == 1
    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


def test_edits_that_bring_back_files_that_already_failed_the_same_way_are_no_progress() -> None:
    # A -> B -> A: the streak never reaches the limit, but the healer is going in circles.
    attempts = [
        _attempt(1, DEPENDENCY, _fix("pom.xml"), state="files-A"),
        _attempt(2, BUILD, _fix("pom.xml"), state="files-B"),
        _attempt(3, DEPENDENCY, state="files-A"),
    ]

    assert stop_reason(attempts, SAME_FAILURE_LIMIT) == TestStopReason.NO_PROGRESS


def test_a_heal_that_changed_no_file_never_counts_as_going_back() -> None:
    # It may have changed the environment (installed a driver, a JDK): the same files are no proof of a loop.
    installed_a_driver = HealOutcomeModel(summary="Installed chromedriver 129")
    attempts = [
        _attempt(1, DEPENDENCY, _fix("pom.xml"), state="files-A"),
        _attempt(2, BUILD, installed_a_driver, state="files-B"),
        _attempt(3, DEPENDENCY, state="files-A"),
    ]

    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


def test_the_same_files_with_a_different_failure_are_not_a_loop() -> None:
    attempts = [
        _attempt(1, DEPENDENCY, _fix("pom.xml"), state="files-A"),
        _attempt(2, BUILD, _fix("pom.xml"), state="files-B"),
        _attempt(3, _result(FailureKind.ENVIRONMENT_FAILURE, ["no display"]), state="files-A"),
    ]

    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


def test_numbers_do_not_make_a_failure_new() -> None:
    first = _result(FailureKind.BUILD_FAILURE, ["Total time: 3.2 s, LoginPage.java:[12,8] cannot find symbol"])
    second = _result(FailureKind.BUILD_FAILURE, ["Total time: 4.9 s, LoginPage.java:[13,8] cannot find symbol"])

    assert failure_signature(first) == failure_signature(second)


def test_there_is_no_fix_count_limit() -> None:
    kinds = [FailureKind.DEPENDENCY_FAILURE, FailureKind.BUILD_FAILURE, FailureKind.ENVIRONMENT_FAILURE] * 3
    attempts = [_attempt(n + 1, _result(kind, [f"error {n}"]), _fix(f"file{n}")) for n, kind in enumerate(kinds)]
    attempts.append(_attempt(len(attempts) + 1, _result(FailureKind.BUILD_FAILURE, ["another error"])))

    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


CODE_ERROR = _result(FailureKind.TEST_CODE_FAILURE, ["CartTest.remove: NullPointerException"], passed=11, failed=1)


def test_a_heal_that_made_it_worse_stays_and_the_healer_goes_on_from_there() -> None:
    # Nothing is ever rolled back (the user's rule): the next heal works on the worse failure.
    attempts = [_attempt(1, CODE_ERROR, _fix("pom.xml")), _attempt(2, BUILD)]

    assert got_worse(CODE_ERROR, BUILD)
    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


@pytest.mark.parametrize(
    ("before", "after", "worse"),
    [
        (CODE_ERROR, BUILD, True),  # the tests no longer run
        (CODE_ERROR, _result(FailureKind.TEST_CODE_FAILURE, ["x"], passed=10, failed=2), True),  # fewer pass
        (CODE_ERROR, _result(FailureKind.TEST_FAILURE, passed=11, failed=1), False),  # as many pass
        (DEPENDENCY, BUILD, False),  # no tests ran before either
        (BUILD, CODE_ERROR, False),  # better: the tests run now
        (CODE_ERROR, _result(FailureKind.PASSED, passed=12), False),
    ],
)
def test_worse_means_fewer_tests_pass_or_none_run(before: TestRunResultModel, after: TestRunResultModel, worse: bool) -> None:
    assert got_worse(before, after) is worse


def _build(*errors: str) -> TestRunResultModel:
    return _result(FailureKind.BUILD_FAILURE, list(errors))


BASE_ERROR = "[ERROR] /workspace/project/src/test/java/com/demo/base/BaseTest.java:[22,48] ';' expected"
TEST_ERROR = "[ERROR] /workspace/project/src/test/java/com/demo/ngauto/LoginTest.java:[20,2] reached end of file while parsing"


def test_a_build_with_fewer_errors_and_no_new_one_is_better() -> None:
    # As in a real run: the first heal fixed the base class, one generated test still did not compile. The same
    # "COMPILATION ERROR" header made that look like no progress, and the fix was thrown away.
    before, after = _build(BASE_ERROR, TEST_ERROR), _build(TEST_ERROR.replace("[20,2]", "[21,2]"))  # its line moved
    attempts = [_attempt(1, before, _fix("src/test/java/com/demo/base/BaseTest.java")), _attempt(2, after)]

    assert got_better(before, after) and not got_worse(before, after)
    assert same_failure_streak(attempts) == 1


def test_the_same_failure_with_more_tests_passing_is_progress_not_a_streak() -> None:
    # A real run: Cucumber scenarios went from 0 to 1 passing with the same failure, and the loop stopped as "no
    # progress".
    none_pass = _result(FailureKind.TEST_CODE_FAILURE, ["Steps.login: NullPointerException"], failed=3)
    one_passes = _result(FailureKind.TEST_CODE_FAILURE, ["Steps.login: NullPointerException"], passed=1, failed=2)
    attempts = [_attempt(1, none_pass, _fix("Steps.java")), _attempt(2, none_pass, _fix("Steps.java")),
                _attempt(3, one_passes)]

    assert same_failure_streak(attempts) == 1
    assert stop_reason(attempts, SAME_FAILURE_LIMIT) is None


def test_a_build_that_only_gained_errors_is_worse() -> None:
    assert got_worse(_build(TEST_ERROR), _build(TEST_ERROR, BASE_ERROR))
    # Other errors instead of the old ones: neither better nor worse.
    assert not got_worse(_build(TEST_ERROR), _build(BASE_ERROR)) and not got_better(_build(TEST_ERROR), _build(BASE_ERROR))


def test_report_lists_every_change_also_when_the_tests_got_worse() -> None:
    attempts = [
        _attempt(1, CODE_ERROR, _fix("pom.xml")),
        _attempt(2, BUILD, _fix("src/test/java/Junk.java")),
        _attempt(3, BUILD),
    ]

    report = build_test_report(attempts, TestStopReason.NO_PROGRESS)

    assert report.changed_files == ["pom.xml", "src/test/java/Junk.java"]


def test_no_attempts_is_a_bug() -> None:
    with pytest.raises(ValueError):
        stop_reason([], SAME_FAILURE_LIMIT)


def test_report_counts_the_last_run_and_every_kept_change() -> None:
    attempts = [
        _attempt(1, DEPENDENCY, _fix("pom.xml")),
        _attempt(2, _result(FailureKind.BUILD_FAILURE, ["x"]), _fix("pom.xml", "src/test/java/Base.java")),
        _attempt(3, _result(FailureKind.PASSED, passed=4)),
    ]

    report = build_test_report(attempts, TestStopReason.PASSED)

    assert (report.outcome, report.runs, report.heals, report.total, report.passed) == (FailureKind.PASSED, 3, 2, 4, 4)
    assert report.changed_files == ["pom.xml", "src/test/java/Base.java"]
    assert describe_test_report(report) == "All 4 tests passed after 2 fixes by the healer"


@pytest.mark.parametrize(
    ("attempts", "reason", "detail", "expected"),
    [
        ([_attempt(1, _result(FailureKind.PASSED, passed=2))], TestStopReason.PASSED, None, "All 2 tests passed"),
        (
            [_attempt(1, _result(FailureKind.TEST_FAILURE, passed=3, failed=1))],
            TestStopReason.NOT_HEALABLE,
            None,
            "Tests finished: 3 passed, 1 failed · some tests failed their checks: review the failures",
        ),
        (
            [_attempt(1, DEPENDENCY, _fix("pom.xml")), _attempt(2, DEPENDENCY, _fix("pom.xml")), _attempt(3, DEPENDENCY)],
            TestStopReason.NO_PROGRESS,
            None,
            "Stopped: no progress, the same dependency problem keeps coming back after the healer's fixes",
        ),
        (
            [_attempt(1, _result(FailureKind.ENVIRONMENT_FAILURE))],
            TestStopReason.BLOCKED,
            "the sandbox cannot reach the internet",
            "Stopped: the healer is blocked: the sandbox cannot reach the internet",
        ),
        (
            [_attempt(1, _result(FailureKind.BUILD_FAILURE))],
            TestStopReason.TIMED_OUT,
            "Stopped: the healer used all of its 60 minutes of healing time",
            "Stopped: the healer used all of its 60 minutes of healing time",
        ),
        (
            [_attempt(1, _result(FailureKind.BUILD_FAILURE))],
            TestStopReason.HEALER_FAILED,
            "The healer stopped: the model could not be reached, try again later",
            "The healer stopped: the model could not be reached, try again later",
        ),
    ],
)
def test_report_lines(attempts: list[TestAttemptModel], reason: TestStopReason, detail: str | None, expected: str) -> None:
    assert describe_test_report(build_test_report(attempts, reason, detail)) == expected
