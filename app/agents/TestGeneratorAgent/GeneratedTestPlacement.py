"""Where generated tests go, and how to run only them afterwards. Plain code, no LLM.

Every generated test lives in a folder (or package) named `ngauto`; generated Cucumber features carry the `@ngauto`
tag. The master then runs just those tests with a test selector built here from the files the generator wrote.
"""

import re
from pathlib import Path, PurePosixPath

from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import FileChangeModel
from app.models.testDataModel import TestCaseSpecModel
from app.models.testGeneratorModel import UnverifiedLocatorModel

GENERATED_DIR = "ngauto"
GENERATED_TAG = "@ngauto"
NO_STEPS = "the case has no steps"
# Put by the generator (and the healer) on every locator that is not in the test data or the project: a person
# should check it.
UNVERIFIED_MARKER = "NGAUTO-UNVERIFIED"

_FEATURE_LINE = re.compile(r"^(\s*)Feature:", re.MULTILINE)
_TEST_FILE = re.compile(
    r"(Test|Tests)\.(java|kt)$|^test_.*\.py$|_test\.py$|\.(spec|test|cy)\.(ts|tsx|js|jsx|mjs|cjs)$|\.robot$"
)


def placement_rule(fact_sheet: FactSheetModel) -> str:
    """The instruction that makes the generated tests selectable, for this project's stack."""
    stack = _stack(fact_sheet)
    if stack == "cucumber":
        return (
            f"Write the scenarios in a new .feature file in the project's features folder, with the tag {GENERATED_TAG} "
            "on the line above 'Feature:'. Add any missing step definitions to the existing step-definition classes "
            "(or a new class next to them)."
        )
    if stack in ("maven", "gradle"):
        return (
            f"Put each new test class in a package named '{GENERATED_DIR}' under the project's existing test package "
            "(e.g. src/test/java/com/acme/ngauto/LoginTest.java); class names end with 'Test'."
        )
    if stack == "pytest":
        return f"Put the new test files in a folder named '{GENERATED_DIR}' inside the tests folder, named test_<case>.py."
    if stack == "robot":
        return f"Put the new .robot files in a folder named '{GENERATED_DIR}' next to the existing suites."
    return (
        f"Put the new spec files in a folder named '{GENERATED_DIR}' inside the folder that holds the existing specs, "
        "using the same file naming and language (.ts or .js) as the project."
    )


def runnable_cases(cases: list[TestCaseSpecModel]) -> tuple[list[TestCaseSpecModel], list[tuple[str, str]]]:
    """Cases the generator can write, and (case id, reason) for the ones it must leave out.

    A step without a locator does not leave its case out: the generator finds one or assumes one and marks it
    (UNVERIFIED_MARKER). Only a case without any step cannot become a test.
    """
    runnable: list[TestCaseSpecModel] = []
    skipped: list[tuple[str, str]] = []
    for case in cases:
        if case.steps:
            runnable.append(case)
        else:
            skipped.append((case.id, NO_STEPS))
    return runnable, skipped


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


def tag_untagged_features(project_root: Path, files: list[FileChangeModel]) -> list[str]:
    """Add the @ngauto tag to new .feature files that lack it; returns the files it changed."""
    tagged: list[str] = []
    for change in files:
        if change.change != "added" or not change.path.endswith(".feature"):
            continue
        path = project_root / change.path
        text = path.read_text(encoding="utf-8")
        if GENERATED_TAG in text or not _FEATURE_LINE.search(text):
            continue
        path.write_text(_FEATURE_LINE.sub(rf"\g<1>{GENERATED_TAG}\n\g<1>Feature:", text, count=1), encoding="utf-8")
        tagged.append(change.path)
    return tagged


def generated_selector(fact_sheet: FactSheetModel, files: list[FileChangeModel]) -> str | None:
    """A test selector that runs only the generated tests, or None when they cannot be told apart."""
    added = [PurePosixPath(change.path) for change in files if change.change == "added"]
    stack = _stack(fact_sheet)
    if stack == "cucumber":
        return GENERATED_TAG if any(path.suffix == ".feature" for path in added) else None
    tests = [path for path in added if _TEST_FILE.search(path.name)]
    folders = {_generated_folder(path) for path in tests}
    if not tests or None in folders:
        return None
    if stack == "maven":
        return f"**/{GENERATED_DIR}/*"
    if stack == "gradle":
        return f"*.{GENERATED_DIR}.*"
    if len(folders) != 1:
        return None
    folder = str(folders.pop())
    return f"{folder}/**" if "Cypress" in _values(fact_sheet.test_frameworks.value) else folder


def _generated_folder(path: PurePosixPath) -> PurePosixPath | None:
    for index, part in enumerate(path.parts[:-1]):
        if part == GENERATED_DIR:
            return PurePosixPath(*path.parts[: index + 1])
    return None


def _stack(fact_sheet: FactSheetModel) -> str:
    build_tool = _first(fact_sheet.build_tool.value)
    frameworks = _values(fact_sheet.test_frameworks.value)
    if build_tool == "Maven" and _first(fact_sheet.bdd_tool.value) == "Cucumber":
        return "cucumber"
    if build_tool == "Maven":
        return "maven"
    if build_tool == "Gradle":
        return "gradle"
    if "pytest" in frameworks:
        return "pytest"
    if "Robot Framework" in frameworks:
        return "robot"
    return "javascript"


def _values(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _first(value: str | list[str] | None) -> str | None:
    values = _values(value)
    return values[0] if values else None
