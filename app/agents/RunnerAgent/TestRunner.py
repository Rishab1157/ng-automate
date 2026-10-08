"""Run a project's tests inside a writable sandbox and turn the outcome into a TestRunResultModel.

Plain code, no LLM: the framework's own command runs in the container, the report files it writes land in the
run's project copy on the host (it is mounted read-write), and the failure classifier decides who handles the
result (the healer or the user).
"""

import logging
import shutil
import time
from pathlib import Path, PurePosixPath

from openhands.sdk.workspace import BaseWorkspace

from app.models.testRunModel import TestCaseStatus, TestCommandModel, TestRunResultModel
from app.services.sandboxService.SandboxService import SANDBOX_PROJECT_DIR
from app.utils.FailureClassifier import classify_failure
from app.utils.TestReportParser import parse_reports

logger = logging.getLogger(__name__)

MAX_OUTPUT_TAIL_CHARS = 20_000
TIMED_OUT_NOTE = "\n[NG Automate] The test command was stopped after {seconds} seconds."


class TestRunner:
    def __init__(self, timeout_seconds: int = 1800) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, workspace: BaseWorkspace, project_dir: Path, command: TestCommandModel) -> TestRunResultModel:
        """Blocking: call from a worker thread."""
        _remove_old_reports(project_dir, command.report_globs)
        cwd = str(PurePosixPath(SANDBOX_PROJECT_DIR) / command.working_dir)
        started = time.monotonic()
        result = workspace.execute_command(command.command, cwd=cwd, timeout=self.timeout_seconds)
        duration = time.monotonic() - started

        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.timeout_occurred:
            output += TIMED_OUT_NOTE.format(seconds=self.timeout_seconds)
        exit_code = result.exit_code if not result.timeout_occurred else 124
        cases, report_files = parse_reports(project_dir, command.report_globs)
        classification = classify_failure(exit_code, output, cases)
        logger.info(
            "Tests ran with %s in %.0fs: exit %d, %d case(s), %s", command.tool, duration, exit_code, len(cases),
            classification.kind.value,
        )

        def count(status: TestCaseStatus) -> int:
            return sum(1 for case in cases if case.status == status)

        return TestRunResultModel(
            command=command.command,
            exit_code=exit_code,
            duration_seconds=round(duration, 1),
            total=len(cases),
            passed=count(TestCaseStatus.PASSED),
            failed=count(TestCaseStatus.FAILED),
            errors=count(TestCaseStatus.ERROR),
            skipped=count(TestCaseStatus.SKIPPED),
            cases=cases,
            report_files=report_files,
            output_tail=output[-MAX_OUTPUT_TAIL_CHARS:],
            classification=classification,
        )


def _remove_old_reports(project_dir: Path, report_globs: list[str]) -> None:
    """Reports from an earlier run (or committed by mistake) must not be read as this run's results."""
    root = project_dir.resolve()
    for pattern in report_globs:
        for match in root.glob(pattern):
            if match.is_file() and match.resolve().is_relative_to(root):
                match.unlink(missing_ok=True)
    stale_dir = root / "ngauto-results"
    if stale_dir.is_dir():
        shutil.rmtree(stale_dir, ignore_errors=True)
