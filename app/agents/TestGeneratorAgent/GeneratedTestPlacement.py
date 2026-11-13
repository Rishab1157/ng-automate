"""Where generated tests go, the locators the generator had to assume, and the new test files the project's plain
test command would skip. Plain code, no LLM.

Generated tests live in a folder (or package) named `ngauto`, so they are easy to find next to the project's own.
The user then picks which test files to run (a test run's test_files). Every case the test data gives becomes a test,
whatever it holds: steps are detail the agent works out from the case's words, the user's files and the application
when they are missing, and a locator it finds or assumes and marks (UNVERIFIED_MARKER).
"""

from functools import partial
from pathlib import Path

from app.agents.RunnerAgent.TestCommandResolver import is_test_file, read_small_text
from app.models.healerModel import FileChangeModel, GuardViolationModel
from app.models.testGeneratorModel import UnverifiedLocatorModel

GENERATED_DIR = "ngauto"
# Put by the generator (and the healer) on every locator that is not in the test data or the project: a person
# should check it.
UNVERIFIED_MARKER = "NGAUTO-UNVERIFIED"
NOT_RUN_BY_DEFAULT = "not_run_by_default"


def unverified_locators(project_root: Path, files: list[FileChangeModel], limit: int = 200) -> list[UnverifiedLocatorModel]:
    """The marked (assumed) locators in the files the generator wrote: file, line and the code line."""
    found: list[UnverifiedLocatorModel] = []
    for change in files:
        if change.change == "deleted":
            continue
        try:
            lines = (project_root / change.path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            if UNVERIFIED_MARKER not in line:
                continue
            code = line.split(UNVERIFIED_MARKER, 1)[0].rstrip(" \t/#*").strip()
            if not code and index + 1 < len(lines):
                code = lines[index + 1].strip()  # the comment sits on its own line, above the locator
            found.append(UnverifiedLocatorModel(path=change.path, line=index + 1, code=code[:300]))
            if len(found) >= limit:
                return found
    return found


def tests_not_run_by_default(project_root: Path, files: list[FileChangeModel], tool: str) -> list[GuardViolationModel]:
    """New files that hold tests (by their content) but are named so that the project's plain test command skips them,
    to review: chosen in a test run they still run (by name). Only Maven picks its test classes by name (Surefire's
    Test*, *Test, *Tests, *TestCase, what is_test_file goes by without the content); Gradle finds them by their
    content, and for the other runners a file's name already decides whether it is a test file at all."""
    if tool != "maven":
        return []
    return [
        GuardViolationModel(file=change.path, rule=NOT_RUN_BY_DEFAULT, line=change.path)
        for change in files
        if change.change == "added"
        and is_test_file(change.path, partial(read_small_text, project_root / change.path))
        and not is_test_file(change.path)
    ]
