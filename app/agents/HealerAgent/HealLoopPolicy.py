"""When the master lets the healer try again, and when the test phase stops. Plain code, no LLM.

The healer is agent-driven: it decides how to fix a failure. This code only decides when to stop. After a test run:
- the tests passed -> stop;
- the failure is not the healer's job (failed checks, no tests; locators when HEAL_LOCATOR_FAILURES is off) -> stop,
  for the user;
- no progress: the same failure came back HEALER_SAME_FAILURE_LIMIT runs in a row, or the healer's edits brought
  the project back to files that were already tested and failed the same way -> stop;
- otherwise the healer gets another turn, told what its earlier fixes did.
A run that is worse than the run before the heal (got_worse) never gets healed: the master undoes the heal's changes
and runs the tests again, so one bad heal cannot pile up on top of a good one.
After a heal (decided by the master, before the tests run again): a proven blocker, the healing time budget used up,
or the healer itself failing -> stop.
"""

import re

from app.config import settings
from app.models.healMemoryModel import HealMemoryResult
from app.models.runModel import TestAttemptModel, TestReportModel, TestStopReason
from app.models.testRunModel import FailureKind, TestRunResultModel

# Numbers change between runs (times, line numbers of generated code, ports): ignore them when comparing failures.
_NUMBERS = re.compile(r"\d+")


def stop_reason(
    attempts: list[TestAttemptModel], same_failure_limit: int = settings.HEALER_SAME_FAILURE_LIMIT
) -> TestStopReason | None:
    """Why the loop stops after the last run, or None to let the healer fix it."""
    if not attempts:
        raise ValueError("No test run yet")
    last = attempts[-1].result
    if last.classification.kind == FailureKind.PASSED:
        return TestStopReason.PASSED
    if not last.classification.healable:
        return TestStopReason.NOT_HEALABLE
    if same_failure_streak(attempts) >= same_failure_limit or _back_to_a_tested_state(attempts):
        return TestStopReason.NO_PROGRESS
    return None


def same_failure_streak(attempts: list[TestAttemptModel]) -> int:
    """How many runs in a row, up to the last one, failed the same way. A run of changes that were undone afterwards
    does not count either way: the files it tested are gone."""
    counted = [attempt for index, attempt in enumerate(attempts) if index == 0 or not _was_undone(attempts[index - 1])]
    last = failure_signature(counted[-1].result)
    streak = 0
    for attempt in reversed(counted):
        if failure_signature(attempt.result) != last:
            break
        streak += 1
    return streak


def _was_undone(attempt: TestAttemptModel) -> bool:
    return attempt.heal is not None and attempt.heal.undone


def _back_to_a_tested_state(attempts: list[TestAttemptModel]) -> bool:
    """The healer edited files, and the result is files that were already tested and failed the same way.

    A heal that changed no file (for example one that installed a driver) never counts: the environment may differ.
    """
    if len(attempts) < 3:
        return False
    last, fix = attempts[-1], attempts[-2].heal
    if last.project_state is None or fix is None or not fix.changes:
        return False
    signature = failure_signature(last.result)
    return any(
        earlier.project_state == last.project_state and failure_signature(earlier.result) == signature
        for earlier in attempts[:-2]
    )


def judge_heal(attempt: TestAttemptModel, next_attempt: TestAttemptModel | None) -> HealMemoryResult:
    """What `attempt`'s heal achieved, by the test run after it (None: the loop stopped before running again)."""
    if next_attempt is None:
        return HealMemoryResult.BLOCKED if attempt.heal is not None and attempt.heal.blocker else HealMemoryResult.UNFINISHED
    if next_attempt.result.classification.kind == FailureKind.PASSED:
        return HealMemoryResult.FIXED
    if got_worse(attempt.result, next_attempt.result):
        return HealMemoryResult.WORSE
    if failure_signature(next_attempt.result) == failure_signature(attempt.result):
        return HealMemoryResult.NOT_FIXED
    return HealMemoryResult.CHANGED


def got_worse(before: TestRunResultModel, after: TestRunResultModel) -> bool:
    """`after` is worse than `before`: the tests no longer run (where they did), or fewer of them pass."""
    return _rank(after) < _rank(before)


def _rank(result: TestRunResultModel) -> tuple[bool, bool, int]:
    """How far the tests got, to compare runs: passing beats everything, then running at all, then passing tests."""
    return result.classification.kind == FailureKind.PASSED, result.total > 0, result.passed


def failure_signature(result: TestRunResultModel) -> tuple[str, ...]:
    """What makes two failures "the same": the kind and the lines that decided it, without numbers."""
    return (result.classification.kind.value, *(_NUMBERS.sub("#", line) for line in result.classification.evidence))


def heal_count(attempts: list[TestAttemptModel]) -> int:
    return sum(1 for attempt in attempts if attempt.heal is not None)


def heal_changes_helped(attempts: list[TestAttemptModel]) -> bool:
    """Whether the healer's changes may go into the project: the tests ran again after the last heal and are better
    than before the first one: they pass, more of them pass, or they run at all where they did not (a build fixed).
    Changes that were never checked by a run, or that changed nothing measurable (e.g. debug prints and pauses), or
    after which fewer tests passed, would only clutter or damage the user's project."""
    if not attempts or attempts[-1].heal is not None:
        return False  # the loop stopped right after a heal: nothing checked its changes
    return _rank(attempts[-1].result) > _rank(attempts[0].result)


CHANGES_NOT_KEPT = "the healer's changes were not kept: the tests did not get better with them"

_KIND_TEXT: dict[FailureKind, str] = {
    FailureKind.BUILD_FAILURE: "build error",
    FailureKind.DEPENDENCY_FAILURE: "dependency problem",
    FailureKind.ENVIRONMENT_FAILURE: "environment problem",
    FailureKind.LOCATOR_FAILURE: "elements were not found on the page: check the locators and the test data",
    FailureKind.TEST_FAILURE: "some tests failed their checks: review the failures",
    FailureKind.TEST_CODE_FAILURE: "the test code itself failed with errors",
    FailureKind.NO_TESTS: "no tests were run: check the test selector and the test setup",
    FailureKind.UNKNOWN: "the tests failed for a reason NG Automate could not classify",
}


def describe_test_report(report: TestReportModel) -> str:
    """One line for the timeline and the run summary, e.g. "All 12 tests passed after 1 fix by the healer"."""
    text = _describe_outcome(report)
    return f"{text} · {CHANGES_NOT_KEPT}" if report.changes_kept is False else text


def _describe_outcome(report: TestReportModel) -> str:
    fixes = f"{report.heals} fix" if report.heals == 1 else f"{report.heals} fixes"
    kind = _KIND_TEXT.get(report.outcome, report.outcome.value)
    if report.stop_reason == TestStopReason.PASSED:
        tests = f"All {report.total} tests passed" if report.total else "The tests passed"
        return f"{tests} after {fixes} by the healer" if report.heals else tests
    if report.stop_reason == TestStopReason.NOT_HEALABLE:
        found = f"{report.passed} passed, {report.failed + report.errors} failed · " if report.total else ""
        return f"Tests finished: {found}{kind}"
    if report.stop_reason == TestStopReason.NO_PROGRESS:
        return f"Stopped: no progress, the same {kind} keeps coming back after the healer's fixes"
    if report.stop_reason == TestStopReason.BLOCKED:
        return f"Stopped: the healer is blocked: {report.detail}" if report.detail else "Stopped: the healer is blocked"
    if report.stop_reason == TestStopReason.MAX_ATTEMPTS:
        return f"Stopped: the healer could not fix the {kind} in {fixes}"
    return report.detail or "Stopped: the healer could not continue"


def build_test_report(
    attempts: list[TestAttemptModel], reason: TestStopReason, detail: str | None = None
) -> TestReportModel:
    last = attempts[-1].result
    changed = sorted({
        change.path for attempt in attempts if attempt.heal and not attempt.heal.undone for change in attempt.heal.changes
    })
    return TestReportModel(
        changes_kept=heal_changes_helped(attempts) if changed else None,
        outcome=last.classification.kind,
        stop_reason=reason,
        detail=detail,
        runs=len(attempts),
        heals=heal_count(attempts),
        total=last.total,
        passed=last.passed,
        failed=last.failed,
        errors=last.errors,
        skipped=last.skipped,
        changed_files=changed,
    )
