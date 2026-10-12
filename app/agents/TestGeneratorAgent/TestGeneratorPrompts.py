"""The test generator's task message, kept compact for a small local model."""

from app.agents.TestGeneratorAgent.GeneratedTestPlacement import UNVERIFIED_MARKER
from app.models.analyzerModel import AnalyzerFindingsModel, FactModel, FactSheetModel
from app.models.testDataModel import TestCaseSpecModel, TestStepModel
from app.models.testRunModel import TestCommandModel
from app.utils.AgentPromptRules import PLAIN_TOOL_VALUES_RULE

MAX_IMPORTANT_PATHS = 12

_GENERATE_PROMPT = """You are adding new automated tests to the test-automation project in /workspace/project.

Project: {stack}
Architecture: {architecture}
Important places:
{important_paths}
The tests run with: {command}

Write one automated test for each of these test cases (from the user's test data):{batch_note}
{cases}

Rules:
- First read a few existing tests and the page objects / step definitions / helpers they use. Follow the same
  structure, style and naming, and reuse what exists (base classes, page objects, drivers, config, waits).
- {placement}
- Use exactly the URLs, locators and values given in the steps.
- A step without a locator (or with element="..." naming its element in words) still gets its test. Find the
  element's locator: first in the project's page objects (reuse it); otherwise `ngauto-find-elements <page URL>`
  in the terminal lists the page's elements with locators checked on the live page. If you still cannot find
  it, write your best guess. Every locator that is not in the test data and not already in the project gets the
  comment {marker} at the end of its line (// in Java/JS/TS, # in Python, Robot and feature files), e.g.
  `By checkout = By.id("checkout"); // {marker}: assumed locator, not in the test data`. Never leave a case out.
- If the project keeps its locators in a file and looks them up by key (for example locators.properties read by a
  helper), pass an existing KEY from that file (the key whose value is the step's locator) or add a new key=value
  line to the file. Never pass a locator value where the code expects a key.
- Assert with the project's test framework (for example TestNG Assert, JUnit Assertions, pytest assert, expect),
  not with a language-level assert keyword that can be switched off.
- Preconditions ("Given:") are part of the test: automate them as setup steps before the case's steps (for
  example log in, or add an item to the cart before testing the payment), reusing the project's existing helpers,
  page objects and steps. Never assume the application is already in that state; no person will prepare it.
- Every "verify" step must become a real assertion. One test per case, named after the case id and title.
- Only ADD code. Never change or remove existing lines (tests, locators, page-object methods, step definitions):
  other tests use them. If you need a different locator or a new action, add a new field, method or step.
- Every step in a feature file you write must have a step definition: reuse existing steps or add new ones.
- Keep working until all files are written; do not stop to explain your plan.
- {plain_values}
- Change files only with the file_editor tool (create, view, str_replace). Do not edit files with shell commands,
  do not create backup copies, and do not run git.
- You may compile to check the code builds (e.g. `mvn -q -DskipTests test-compile`), but do not run the whole suite.

When you are done, reply with a short summary: which files you created or changed."""

_BATCH_NOTE = "\n(These are cases 1-{last} of {total}; the next ones follow once these are written.)"

_NEXT_BATCH_PROMPT = """Now write the tests for the next test cases ({first}-{last} of {total}), with the same rules as before.
Reuse what you already found and wrote: add new tests next to the ones you wrote (new files, or new tests in the
files you created).
{cases}

When these are written, reply with a short summary: which files you created or changed."""


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
    total_cases: int | None = None,
) -> str:
    """The first message: the project, the rules and the first cases (`total_cases` when more follow)."""
    if "Appium" in _values(fact_sheet.automation_tools):
        placement += MOBILE_RULE
    total = total_cases or len(cases)
    return _GENERATE_PROMPT.format(
        stack=_stack(fact_sheet),
        architecture=_text(findings.architecture_pattern) if findings else "unknown",
        important_paths=_important_paths(findings),
        command=command.command,
        batch_note=_BATCH_NOTE.format(last=len(cases), total=total) if total > len(cases) else "",
        cases="\n".join(_case(case) for case in cases),
        placement=placement,
        marker=UNVERIFIED_MARKER,
        plain_values=PLAIN_TOOL_VALUES_RULE,
    )


def build_next_batch_prompt(cases: list[TestCaseSpecModel], first: int, total: int) -> str:
    """A later message in the same conversation: the next cases (`first` is the 1-based number of the first one)."""
    return _NEXT_BATCH_PROMPT.format(
        first=first, last=first + len(cases) - 1, total=total, cases="\n".join(_case(case) for case in cases)
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
