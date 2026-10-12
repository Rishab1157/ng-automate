"""Where generated tests go, which test cases can become tests, and the locators the generator had to assume.
Plain code, no LLM.

Generated tests live in a folder (or package) named `ngauto`, so they are easy to find next to the project's own.
The user then picks which test files to run (a test run's test_files).
"""

from pathlib import Path

from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import FileChangeModel
from app.models.testDataModel import TestCaseSpecModel
from app.models.testGeneratorModel import UnverifiedLocatorModel

GENERATED_DIR = "ngauto"
NO_STEPS = "the case has no steps"
# Put by the generator (and the healer) on every locator that is not in the test data or the project: a person
# should check it.
UNVERIFIED_MARKER = "NGAUTO-UNVERIFIED"


def placement_rule(fact_sheet: FactSheetModel) -> str:
    """Where the generated tests go, for this project's stack."""
    stack = _stack(fact_sheet)
    if stack == "cucumber":
        return (
            "Write the scenarios in a new .feature file in the project's features folder. Add any missing step "
            "definitions to the existing step-definition classes (or a new class next to them)."
        )
    if stack in ("maven", "gradle"):
        return (
            f"Put each new test class in a package named '{GENERATED_DIR}' under the project's existing test package: "
            f"find a file that already holds the project's tests and put yours next to it in a '{GENERATED_DIR}' "
            f"subfolder, with the same package name plus '.{GENERATED_DIR}' (so the class can still import the "
            "project's base class and helpers). Class names end with 'Test'. A project with no tests yet: use its "
            f"root test package instead (page objects in com.acme.pages mean the package com.acme.{GENERATED_DIR}, in "
            f"src/test/java/com/acme/{GENERATED_DIR})."
        )
    if stack == "pytest":
        return (
            f"Put the new test files in a folder named '{GENERATED_DIR}' inside the tests folder (a 'tests' folder at "
            "the project root when there is none), named test_<case>.py."
        )
    if stack == "robot":
        return (
            f"Put the new .robot files in a folder named '{GENERATED_DIR}' next to the existing suites (in a 'tests' "
            "folder at the project root when there are none)."
        )
    return (
        f"Put the new spec files in a folder named '{GENERATED_DIR}' inside the folder that holds the existing specs "
        "(the test folder the runner's config names, when there are none), using the same file naming and language "
        "(.ts or .js) as the project."
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
