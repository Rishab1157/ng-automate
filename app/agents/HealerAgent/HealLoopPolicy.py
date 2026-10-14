"""When the master lets the healer try again, and when the test phase stops. Plain code, no LLM.

The healer is agent-driven: it decides how to fix a failure. This code only decides when to stop. After a test run:
- the tests passed -> stop;
- the failure is not the healer's job (failed checks, no tests; locators when HEAL_LOCATOR_FAILURES is off) -> stop,
  for the user;
- no progress: the same failure came back HEALER_SAME_FAILURE_LIMIT runs in a row, or the healer's edits brought
  the project back to files that were already tested and failed the same way -> stop;
- otherwise the healer gets another turn, told what its earlier fixes did.
A heal's changes are never rolled back (the user's rule: the project's own framework may need them), also when the
tests got worse with them: the next heal works from there, and the heal memory remembers that heal as `worse`.
After a heal (decided by the master, before the tests run again): a proven blocker, the healing time budget used up,
or the healer itself failing -> stop.
"""

import re

from app.config import settings
from app.models.healMemoryModel import HealMemoryResult
from app.models.runModel import TestAttemptModel, TestReportModel, TestStopReason
from app.models.testRunModel import FailureKind, TestRunResultModel
from app.utils.FailureClassifier import BUILD_FILE_BROKEN

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
    """How many runs in a row, up to the last one, failed the same way without getting better. A run that got better
    than the one before it (more tests pass, fewer build errors) starts the streak again: the same failure, but
    progress."""
    last = failure_signature(attempts[-1].result)
    streak = 0
    for index in range(len(attempts) - 1, -1, -1):
        if failure_signature(attempts[index].result) != last:
            break
        streak += 1
        if index > 0 and got_better(attempts[index - 1].result, attempts[index].result):
            break
    return streak


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
    """`after` is worse than `before`: the tests no longer run (where they did), fewer of them pass, or the build only
    gained errors."""
    if _rank(after) != _rank(before):
        return _rank(after) < _rank(before)
    errors_before, errors_after = _build_errors(before), _build_errors(after)
    return errors_before is not None and errors_after is not None and errors_after > errors_before


def got_better(before: TestRunResultModel, after: TestRunResultModel) -> bool:
    """`after` is better than `before`: the tests pass, more of them pass, they run at all where they did not, or the
    build lost some of its errors and gained none (a build with several errors is fixed one heal at a time)."""
    if _rank(after) != _rank(before):
        return _rank(after) > _rank(before)
    errors_before, errors_after = _build_errors(before), _build_errors(after)
    return errors_before is not None and errors_after is not None and errors_after < errors_before


def _rank(result: TestRunResultModel) -> tuple[bool, bool, int, bool]:
    """How far the tests got, to compare runs: passing beats everything, then running at all, then passing tests, and
    among builds that fail, a build file that still loads (code that does not compile can be fixed one file at a time;
    a build file the tool cannot read stops everything)."""
    build_file_broken = result.classification.kind == FailureKind.BUILD_FAILURE and any(
        BUILD_FILE_BROKEN.search(line) for line in result.classification.evidence
    )
    return result.classification.kind == FailureKind.PASSED, result.total > 0, result.passed, not build_file_broken


def _build_errors(result: TestRunResultModel) -> frozenset[str] | None:
    """A build failure's compiler errors without numbers (line numbers move as code changes); None for other runs."""
    if result.classification.kind != FailureKind.BUILD_FAILURE or not result.classification.evidence:
        return None
    return frozenset(_NUMBERS.sub("#", line) for line in result.classification.evidence)


def failure_signature(result: TestRunResultModel) -> tuple[str, ...]:
    """What makes two failures "the same": the kind and the lines that decided it, without numbers."""
    return (result.classification.kind.value, *(_NUMBERS.sub("#", line) for line in result.classification.evidence))


def heal_count(attempts: list[TestAttemptModel]) -> int:
    return sum(1 for attempt in attempts if attempt.heal is not None)


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
    changed = sorted({change.path for attempt in attempts if attempt.heal for change in attempt.heal.changes})
    return TestReportModel(
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
