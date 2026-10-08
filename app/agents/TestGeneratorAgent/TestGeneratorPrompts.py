"""The test generator's task message, kept compact for a small local model."""

from app.models.analyzerModel import AnalyzerFindingsModel, FactModel, FactSheetModel
from app.models.testDataModel import TestCaseSpecModel, TestStepModel
from app.models.testRunModel import TestCommandModel

MAX_IMPORTANT_PATHS = 12

_GENERATE_PROMPT = """You are adding new automated tests to the test-automation project in /workspace/project.

Project: {stack}
Architecture: {architecture}
Important places:
{important_paths}
The tests run with: {command}

Write one automated test for each of these test cases (from the user's test data):
{cases}

Rules:
- First read a few existing tests and the page objects / step definitions / helpers they use. Follow the same
  structure, style and naming, and reuse what exists (base classes, page objects, drivers, config, waits).
- {placement}
- Use exactly the URLs, locators and values given in the steps. Never invent a locator or a URL.
- A step with element="..." names its element in words. Find its real locator: first in the project's page
  objects; if it is not there, run `ngauto-find-elements <page URL>` in the terminal, which lists the page's
  elements with locators checked on the live page, and use one of those. If you cannot find it, leave that case
  out and say so in your summary.
- Every "verify" step must become a real assertion. One test per case, named after the case id and title.
- Only ADD code. Never change or remove existing lines (tests, locators, page-object methods, step definitions):
  other tests use them. If you need a different locator or a new action, add a new field, method or step.
- Every step in a feature file you write must have a step definition: reuse existing steps or add new ones.
- Keep working until all files are written; do not stop to explain your plan.
- Change files only with the file_editor tool (create, view, str_replace). Do not edit files with shell commands,
  do not create backup copies, and do not run git.
- You may compile to check the code builds (e.g. `mvn -q -DskipTests test-compile`), but do not run the whole suite.

When you are done, reply with a short summary: which files you created or changed."""


MOBILE_RULE = (
    "\n- This is a mobile (Appium) project: reuse its driver and capabilities setup and its screen objects. Locators "
    "on app screens are accessibility id, resource-id or XPath; 'tap' means a click on the element."
)


def build_generate_prompt(
    cases: list[TestCaseSpecModel],
    fact_sheet: FactSheetModel,
    findings: AnalyzerFindingsModel | None,
    command: TestCommandModel,
    placement: str,
) -> str:
    if "Appium" in _values(fact_sheet.automation_tools):
        placement += MOBILE_RULE
    return _GENERATE_PROMPT.format(
        stack=_stack(fact_sheet),
        architecture=_text(findings.architecture_pattern) if findings else "unknown",
        important_paths=_important_paths(findings),
        command=command.command,
        cases="\n".join(_case(case) for case in cases),
        placement=placement,
    )


def _case(case: TestCaseSpecModel) -> str:
    tags = f" [tags: {', '.join(case.tags)}]" if case.tags else ""
    lines = [f"Case {case.id}: {case.title}{tags}"]
    if case.description:
        lines.append(f"  About: {case.description}")
    for precondition in case.preconditions:
        lines.append(f"  Given: {precondition}")
    lines.extend(f"  {number}. {_step(step)}" for number, step in enumerate(case.steps, start=1))
    if case.test_data:
        lines.append("  Data: " + ", ".join(f"{key}={value}" for key, value in case.test_data.items()))
    return "\n".join(lines)


def _step(step: TestStepModel) -> str:
    parts = [step.action]
    if step.target:
        parts.append(f"target={step.target}")
    elif step.target_hint:
        parts.append(f'element="{step.target_hint}"')
    if step.value is not None:
        parts.append(f'value="{step.value}"')
    if step.expected is not None:
        parts.append(f'expected="{step.expected}"')
    return " ".join(parts)


def _important_paths(findings: AnalyzerFindingsModel | None) -> str:
    if not findings or not findings.important_paths:
        return "- (none recorded)"
    return "\n".join(f"- {item.path}: {item.role}" for item in findings.important_paths[:MAX_IMPORTANT_PATHS])


def _stack(fact_sheet: FactSheetModel) -> str:
    parts = [
        _text(fact)
        for fact in (fact_sheet.primary_language, fact_sheet.build_tool, fact_sheet.test_frameworks,
                     fact_sheet.automation_tools, fact_sheet.bdd_tool)
    ]
    return " · ".join(part for part in parts if part) or "unknown"


def _values(fact: FactModel) -> list[str]:
    if fact.value is None:
        return []
    return [fact.value] if isinstance(fact.value, str) else list(fact.value)


def _text(fact: FactModel) -> str:
    if fact.value is None:
        return ""
    return fact.value if isinstance(fact.value, str) else ", ".join(fact.value)
