"""The test generator's task message, kept compact for a small local model."""

from collections.abc import Sequence

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
{briefing}
Write one automated test for each of these test cases (from the user's test data):{batch_note}
{cases}

Rules:
- First read a few existing tests and the page objects / step definitions / helpers they use. Follow the same
  structure, style and naming, and reuse what exists (base classes, page objects, drivers, config, waits).
- Write code that compiles. First read the base test class and one existing test, and use only what they provide
  (their fields and methods). Reuse the project's own helpers for waits, navigation, typing and clicking instead of
  building your own (e.g. use the wait the base class already holds; do not construct a new one, and never reference a
  constant or member that does not exist). If your test extends a class or uses a type from the project, copy that
  class's exact package/import line into your file; every type you use must be imported (if the compiler says
  `cannot find symbol` for a type, find the file that declares or imports it and copy that import line in).
- Put your tests in new files. You may also change the project's existing files where your tests need it (for
  example add a page-object method or a step definition, make a helper usable from your tests, add a dependency to
  the build file); such changes are kept. Keep every existing test working: add to the existing code instead of
  rewriting it, never delete a project file, and never remove or weaken an existing test, step or assertion.
- {placement}
- Use exactly the URLs, locators and values given in the steps.
- A page is not ready the moment it loads (single-page apps render afterwards). Look up every element with the
  project's own lookup/wait helpers (the ones the base class or an existing test uses) instead of a bare
  driver.findElement / direct lookup, so the test waits for the element and can fall back to another locator.
- To add lines to a file, view it and use the file_editor `insert` command with `insert_line` (the line after which
  the new code goes) and `new_str`; it needs no unique text.
- When str_replace is rejected (`did not appear verbatim`, or `Multiple occurrences ... Please ensure it is
  unique`), never send the same call again: view the file and include the whole enclosing method or statement,
  with enough surrounding lines, so old_str matches exactly once (or use `insert`).
- A step without a locator (or with element="..." naming its element in words) still gets its test. Find the
  element's locator: first in the user's test data (mounted read-only at /workspace/test-data — it may list several
  locators for the same element), then in the project's page objects (reuse it); otherwise `ngauto-find-elements
  <page URL>` in the terminal lists the page's elements with locators checked on the live page. If you still cannot
  find it, write your best guess. Every locator that is not in the test data and not already in the project gets the
  comment {marker} at the end of its line (// in Java/JS/TS, # in Python, Robot and feature files), e.g.
  `By checkout = By.id("checkout"); // {marker}: assumed locator, not in the test data`. Never leave a case out.
- If the project keeps its locators in a file and looks them up by key (however this project does that), the string
  you pass is the project's KEY, not the element's label or its locator: pass a key the project already defines.
  When the project has no key for an element, use the step's locator directly in your test, or add a key the same
  way the project defines its other keys.
- When a step gives more than one locator for the same element, use the first one and keep the others as a fallback
  (for example a list of candidate locators tried in order, or the extra ones in a clear comment beside the locator).
- Assert with the project's test framework (for example TestNG Assert, JUnit Assertions, pytest assert, expect),
  not with a language-level assert keyword that can be switched off.
- Preconditions ("Given:") are part of the test: automate them as setup steps before the case's steps (for
  example log in, or add an item to the cart before testing the payment), reusing the project's existing helpers,
  page objects and steps. Never assume the application is already in that state; no person will prepare it.
- Every "verify" step must become a real assertion. One test per case, named after the case id and title.
  Write tests only for the cases listed here, with their own ids: never add tests for other cases.
- The `view` command shows each line with its number in front: never copy those numbers into a file.
- `create` never overwrites a file. To rewrite a test file you wrote yourself (for example when it is badly broken),
  delete it first with `rm <path>` in the terminal, then `create` it again with its whole corrected content. Never do
  this to the project's own files.
- If you need a different locator or a new action, add it as a new field or method (in your own file or in the
  project's page object); never change one that the existing tests use.
- Every step in a feature file you write must have a step definition: reuse the existing steps, or define the missing
  ones in a new step-definition class of your own.
- In a feature file, write each step in plain language a business reader understands, using the step's own words
  when the case gives them ("in words"). Never copy the action/target/locator line above into a step: locators and
  values belong in the step definitions or the page objects.
- Keep working until all files are written; do not stop to explain your plan.
- {plain_values}
- Change files only with the file_editor tool (create, view, str_replace). Create each new file with the file_editor's
  `create` command; if its folder does not exist yet, make the folder first with `mkdir -p`, but never create the file
  itself with a shell command or `touch` — the editor refuses to overwrite an existing file, so the file would stay
  empty. If a file you created needs a fix, edit that same file with str_replace; never write the test a second time or
  in another folder. Do not edit files with shell commands, do not create backup copies, and do not run git.
- You may compile to check the code builds (e.g. `mvn -q -DskipTests test-compile`), but do not run the whole suite.
  Treat this as required: run the compile step and fix every error it reports before you finish; never finish with a
  build that does not compile.

When you are done, reply with a short summary: which files you created or changed."""

_BATCH_NOTE = "\n(These are cases 1-{last} of {total}; the next ones follow once these are written.)"

_NEXT_BATCH_PROMPT = """Now write the tests for the next test cases ({first}-{last} of {total}), with the same rules as before.
Put them in NEW test files next to the ones you wrote, made with the file_editor `create` command (a new class per
case or for these cases together, with a name no file has yet). Do not edit the test files you already wrote. Reuse
what you already found: the same base class, page objects, helpers and imports.
{cases}

When these are written, reply with a short summary: which files you created or changed."""
_USER_INSTRUCTIONS = "\n\nThe user told you during this run (this comes before the rules above):\n{instructions}"


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
    briefing: str = "",
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
        briefing=_briefing_section(briefing),
        batch_note=_BATCH_NOTE.format(last=len(cases), total=total) if total > len(cases) else "",
        cases="\n".join(_case(case) for case in cases),
        placement=placement,
        marker=UNVERIFIED_MARKER,
        plain_values=PLAIN_TOOL_VALUES_RULE,
    )


def build_next_batch_prompt(
    cases: list[TestCaseSpecModel], first: int, total: int, instructions: Sequence[str] = ()
) -> str:
    """A later message in the same conversation: the next cases (`first` is the 1-based number of the first one).
    `instructions`: what the user told the agents during the run; repeated so "the same rules" never override it."""
    prompt = _NEXT_BATCH_PROMPT.format(
        first=first, last=first + len(cases) - 1, total=total, cases="\n".join(_case(case) for case in cases)
    )
    if instructions:
        prompt += _USER_INSTRUCTIONS.format(instructions="\n".join(f"- {text}" for text in instructions))
    return prompt


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
    if step.alternate_locators:
        parts.append("if-not-found=" + " | ".join(step.alternate_locators))
    if step.value is not None:
        parts.append(f'value="{step.value}"')
    if step.expected is not None:
        parts.append(f'expected="{step.expected}"')
    # The step in the test data's own words: what a feature file's step or a test's comment should say.
    words = f' (in words: "{step.text}")' if step.text else ""
    return " ".join(parts) + words


def _important_paths(findings: AnalyzerFindingsModel | None) -> str:
    if not findings or not findings.important_paths:
        return "- (none recorded)"
    return "\n".join(f"- {item.path}: {item.role}" for item in findings.important_paths[:MAX_IMPORTANT_PATHS])


BRIEFING_HEADER = (
    "The project's own tests and helpers, exactly as they are (reuse these exact classes and methods, and never call "
    "a helper or constant that does not exist):\n{briefing}"
)


def _briefing_section(briefing: str) -> str:
    """The inlined project API, indented under its header. Empty when the generator has nothing to show."""
    if not briefing.strip():
        return ""
    return "\n" + BRIEFING_HEADER.format(briefing=briefing)


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
