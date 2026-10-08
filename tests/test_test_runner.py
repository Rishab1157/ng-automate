"""TestRunner: runs the command in the sandbox, reads the reports from the host copy, classifies the outcome.

The workspace is a fake that "runs" the command by writing report files into the project folder, the way a
real test framework writes them through the read-write mount.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.agents.RunnerAgent.TestRunner import MAX_OUTPUT_TAIL_CHARS, TestRunner
from app.models.testRunModel import FailureKind, TestCommandModel

MAVEN = TestCommandModel(
    tool="maven",
    command="mvn -B -ntp test",
    report_globs=["target/surefire-reports/TEST-*.xml"],
)

PASSING_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="LoginTest" tests="2" failures="0" errors="0" skipped="0">
  <testcase name="validLogin" classname="tests.LoginTest" time="1.2"/>
  <testcase name="logout" classname="tests.LoginTest" time="0.4"/>
</testsuite>
"""

FAILING_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="LoginTest" tests="1" failures="1" errors="0" skipped="0">
  <testcase name="validLogin" classname="tests.LoginTest" time="1.2">
    <failure message="expected [Dashboard] but found [Login]">java.lang.AssertionError</failure>
  </testcase>
</testsuite>
"""


class FakeWorkspace:
    def __init__(self, project_dir: Path, *, exit_code: int = 0, stdout: str = "", stderr: str = "",
                 timed_out: bool = False, report: str | None = None) -> None:
        self.project_dir = project_dir
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.report = report
        self.calls: list[dict[str, Any]] = []

    def execute_command(self, command: str, cwd: str, timeout: float) -> Any:
        self.calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        if self.report is not None:
            reports = self.project_dir / "target" / "surefire-reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / "TEST-tests.LoginTest.xml").write_text(self.report, encoding="utf-8")
        return SimpleNamespace(
            stdout=self.stdout, stderr=self.stderr, exit_code=self.exit_code, timeout_occurred=self.timed_out
        )


def test_passing_run_counts_cases_and_runs_in_the_sandbox_project_dir(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path, stdout="[INFO] BUILD SUCCESS", report=PASSING_REPORT)

    result = TestRunner(timeout_seconds=600).run(workspace, tmp_path, MAVEN)

    assert workspace.calls == [{"command": "mvn -B -ntp test", "cwd": "/workspace/project", "timeout": 600}]
    assert (result.total, result.passed, result.failed, result.errors, result.skipped) == (2, 2, 0, 0, 0)
    assert result.classification.kind == FailureKind.PASSED
    assert result.report_files == ["target/surefire-reports/TEST-tests.LoginTest.xml"]


def test_working_dir_is_joined_to_the_sandbox_project_dir(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path, report=PASSING_REPORT)
    command = MAVEN.model_copy(update={"working_dir": "ui-tests"})

    TestRunner().run(workspace, tmp_path, command)

    assert workspace.calls[0]["cwd"] == "/workspace/project/ui-tests"


def test_failed_assertion_is_a_test_failure_for_the_user(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path, exit_code=1, stdout="[ERROR] Tests run: 1, Failures: 1", report=FAILING_REPORT)

    result = TestRunner().run(workspace, tmp_path, MAVEN)

    assert result.failed == 1
    assert result.classification.kind == FailureKind.TEST_FAILURE
    assert not result.classification.healable


def test_unresolved_dependency_is_healable(tmp_path: Path) -> None:
    output = "[ERROR] Failed to execute goal on project demo: Could not resolve dependencies for project x:y:jar:1.0"
    workspace = FakeWorkspace(tmp_path, exit_code=1, stdout=output)

    result = TestRunner().run(workspace, tmp_path, MAVEN)

    assert result.total == 0
    assert result.classification.kind == FailureKind.DEPENDENCY_FAILURE
    assert result.classification.healable


def test_old_reports_are_removed_before_the_run(tmp_path: Path) -> None:
    stale = tmp_path / "target" / "surefire-reports"
    stale.mkdir(parents=True)
    (stale / "TEST-old.xml").write_text(PASSING_REPORT, encoding="utf-8")
    (tmp_path / "ngauto-results").mkdir()
    (tmp_path / "ngauto-results" / "junit.xml").write_text(PASSING_REPORT, encoding="utf-8")
    workspace = FakeWorkspace(tmp_path, exit_code=1, stdout="[ERROR] COMPILATION ERROR : cannot find symbol")

    result = TestRunner().run(workspace, tmp_path, MAVEN)

    assert result.total == 0  # the old passing report was not read
    assert not (stale / "TEST-old.xml").exists()
    assert not (tmp_path / "ngauto-results").exists()


def test_timeout_sets_exit_124_and_notes_it_in_the_output(tmp_path: Path) -> None:
    workspace = FakeWorkspace(tmp_path, exit_code=-1, stdout="Running tests...", timed_out=True)

    result = TestRunner(timeout_seconds=5).run(workspace, tmp_path, MAVEN)

    assert result.exit_code == 124
    assert "stopped after 5 seconds" in result.output_tail


def test_output_tail_keeps_the_end_of_long_output(tmp_path: Path) -> None:
    output = "x" * MAX_OUTPUT_TAIL_CHARS + "THE END"
    workspace = FakeWorkspace(tmp_path, exit_code=1, stdout=output)

    result = TestRunner().run(workspace, tmp_path, MAVEN)

    assert len(result.output_tail) == MAX_OUTPUT_TAIL_CHARS
    assert result.output_tail.endswith("THE END")
