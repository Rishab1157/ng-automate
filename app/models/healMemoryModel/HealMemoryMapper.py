import re
import uuid
from datetime import UTC, datetime
from typing import Any

from app.models.runModel import TestAttemptModel
from app.models.testRunModel import FailureKind, TestRunResultModel

from .HealMemoryDbModel import HealMemoryDbModel
from .HealMemoryModel import SUCCESSFUL_RESULTS, HealMemoryModel, HealMemoryResult

MAX_PROBLEM_CHARS = 4000
MAX_OUTPUT_ERROR_LINES = 8
MAX_LINE_CHARS = 300
MAX_SUMMARY_CHARS = 2000
MAX_DIFF_CHARS = 4000
MAX_FILES = 50
# Point ids come from the run and the attempt: saving the same heal again (a resumed run) replaces it.
_POINT_NAMESPACE = uuid.UUID("5b0f8a8e-4c1d-4f7e-9a51-2a1f3c9d7e60")
# Lines of build or test output that describe a failure (any tool, any language).
_ERROR_LINE = re.compile(
    r"error|exception|fail|not found|cannot|could not|unable|missing|denied|refused|no such", re.IGNORECASE
)


def memory_point_id(run_id: str, attempt: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{run_id}:{attempt}"))


def describe_problem(result: TestRunResultModel) -> str:
    """The text a failure is remembered and found by: its kind and the lines that decided it (the classifier's
    evidence; without any, the first error lines of the output, where the cause usually is).

    Measured with qwen3-embedding:0.6b on real failures, this separates "the same problem" from "a different one"
    best: the kind's generic reason and more output lines made different problems look alike.
    """
    classification = result.classification
    lines = [classification.kind.value]
    seen: set[str] = set()
    for line in classification.evidence:
        line = _clip(line.strip())
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    if len(lines) == 1:
        for line in result.output_tail.splitlines():
            line = _clip(line.strip())
            if len(lines) > MAX_OUTPUT_ERROR_LINES:
                break
            if line and line not in seen and _ERROR_LINE.search(line):
                seen.add(line)
                lines.append(line)
    return "\n".join(lines)[:MAX_PROBLEM_CHARS]


def describe_run(result: TestRunResultModel) -> str:
    """One line about a test run, e.g. "the tests passed (4/4)" or "build_failure: The code does not compile (...)"."""
    classification = result.classification
    if classification.kind == FailureKind.PASSED:
        return f"the tests passed ({result.passed}/{result.total})" if result.total else "the tests passed"
    line = f"{classification.kind.value}: {classification.reason}"
    if classification.evidence:
        line += f" ({classification.evidence[0]})"
    return _clip(line)


class HealMemoryMapper:
    @staticmethod
    def to_db_model(
        *,
        org_id: str,
        project_id: str,
        run_id: str,
        attempt: TestAttemptModel,
        next_attempt: TestAttemptModel | None,
        result: HealMemoryResult,
        stack: str,
        test_command: str,
        embedding_model: str,
    ) -> HealMemoryDbModel:
        """The memory of `attempt`'s heal; `next_attempt` is the test run after it (None when the loop stopped)."""
        heal = attempt.heal
        if heal is None:
            raise ValueError(f"Attempt {attempt.number} has no heal to remember")
        failure = attempt.result.classification
        diff = "\n".join(change.diff for change in heal.changes if change.diff)
        return HealMemoryDbModel(
            org_id=org_id,
            project_id=project_id,
            run_id=run_id,
            attempt=attempt.number,
            created_at=datetime.now(UTC),
            embedding_model=embedding_model,
            stack=stack,
            test_command=test_command,
            failure_kind=failure.kind.value,
            failure_reason=failure.reason,
            failure_evidence=list(failure.evidence),
            problem=describe_problem(attempt.result),
            fix_summary=heal.summary[:MAX_SUMMARY_CHARS],
            changed_files=[change.path for change in heal.changes][:MAX_FILES],
            reverted_files=heal.reverted_files[:MAX_FILES],
            diff=diff if len(diff) <= MAX_DIFF_CHARS else diff[: MAX_DIFF_CHARS - 1] + "…",
            blocker=heal.blocker,
            result=result.value,
            succeeded=result in SUCCESSFUL_RESULTS,
            next_run=describe_run(next_attempt.result) if next_attempt is not None else None,
        )

    @staticmethod
    def to_model(point_id: str, payload: dict[str, Any], score: float | None = None) -> HealMemoryModel:
        data = HealMemoryDbModel.model_validate(payload).model_dump(exclude={"embedding_model"})
        return HealMemoryModel(id=point_id, score=score, **data)


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"
