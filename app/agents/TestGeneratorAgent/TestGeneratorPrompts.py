"""The test generator's messages, kept compact for a small local model.

The rules are the same for every stack: they say how to work (look first, follow the project, reuse, check), never
how one framework does it. The project supplies the details: its facts, a sample of its own code, the check command
NG Automate chose for it and, from the second batch on, where the work stands. The user's own instructions come first.
"""

from collections.abc import Sequence

from app.agents.TestGeneratorAgent.GeneratedTestPlacement import UNVERIFIED_MARKER
from app.models.analyzerModel import AnalyzerFindingsModel, FactModel, FactSheetModel
from app.models.testDataModel import TestCaseSpecModel, TestStepModel
from app.models.testRunModel import TestCommandModel
from app.utils.AgentPromptRules import PLAIN_TOOL_VALUES_RULE, VISIBLE_BROWSER_RULE, user_instructions_block

MAX_IMPORTANT_PATHS = 12

_GENERATE_PROMPT = """You are adding new automated tests to the test-automation project in /workspace/project.

Project: {stack}
Architecture: {architecture}
Important places:
{important_paths}
The tests run with: {command}
{check_line}
{briefing}
Write one automated test for each of these test cases (from the user's test data):{batch_note}
{cases}
{user_block}
How to work (the project decides the details: look them up in it, do not assume them):
- Look before you write. Read two or three existing tests and the shared code they use: how a test is declared, how
  it gets its browser, page or client, how it waits, asserts and imports. The shared code is whatever this project has
  (page objects, fixtures, hooks, base classes, helpers, step definitions, keywords, config); the files above are only
  a sample, so search the project (grep, glob) for the rest.
- Same framework and style: write the new tests with the project's own test framework, language, runner and
  conventions, so its own test command runs them. If the project mixes styles, follow the one most of its tests use.
  Use another framework or style only when the user asks for it; then add only what that needs to run (a dependency,
  a runner, glue) and keep the existing tests running as before.
- Reuse first: before you write a helper, page-object method, fixture, step, keyword or setup, search the project and
  your own new files for one that already does it, and call it. Write what several of your tests need (for example
  logging in) once, in a shared file of your own next to your tests, organised the way this project organises such
  code. Never copy the same block into several files, and never define a step text or keyword name that already
  exists: duplicates break the run.
- One test per case, named after the case id (exactly as given, e.g. TD-001) and its title, in a form the runner
  collects. Write tests only for the cases listed here, and never a second test for a case that already has one.
- Put your tests in new files. {placement} You may change the project's own files where your tests need it (add a
  page-object method, a step definition, an export); such changes are kept, so keep every existing test working: add
  instead of rewriting, never delete a project file, never change a locator or method the existing tests use, and
  never remove or weaken a test, step or assertion.
- Imports: every name you use must exist and be imported from where it really is. Your file is not in the folder of
  the test you learn from, so never copy a relative import as it is: work it out from your file's own folder to the
  file it points to (from a/b/new/x, the file a/c/y is ../../c/y), and check that the file and the name exist.
- Dependencies and configuration: change a dependency, script, compiler or test-runner setting only when a test cannot
  work without it (or to show the browser on the screen, see below), never to make checking easier. Do not add build
  tools, compilers or type packages to check your code, and never write compiled copies of your files into the project.
- Steps: use exactly the URLs, values and elements the steps give. Write each locator in this framework's own syntax
  without changing what it points to (id=user becomes the framework's id locator, or the CSS #user). Open pages the way
  the existing tests do. Let lookups and checks wait the way the existing tests do (many frameworks wait by themselves);
  never add fixed sleeps. Assert with the framework's own assertions, as the existing tests do: every "verify" step
  becomes a real assertion.
- Preconditions ("Given:") are part of the test: automate them as setup before the case's steps, with the project's own
  helpers. Never assume the application is already in that state; nobody will prepare it.
- {screen_rule}
- A step without a locator (or with element="..." naming its element in words) still gets its test. Find the locator
  in the user's test data (read-only at /workspace/test-data; it may list several for one element), then in the
  project's page objects, then with `ngauto-find-elements <page URL>` in the terminal (it lists the page's elements with
  locators checked on the live page); otherwise write your best guess. Mark every locator that is in neither the test
  data nor the project with a comment saying {marker}, in that file's comment syntax, at the end of its line or on its
  own line above it (in a feature file only on its own line). Never leave a case out.
- If the project looks up locators by key, pass the project's KEY, not the label or the locator; with no key for an
  element, use the locator directly, or add a key the way the project adds its other keys.
- When a step gives several locators for one element, use the first and keep the others as a fallback (candidates
  tried in order, or a clear comment beside the locator).
- Feature files: write each step in plain language a business reader understands (the case's own words when given);
  locators and values belong in the step definitions or page objects. Every step must match a step definition: reuse a
  matching one, and put only the missing ones in a new step file of your own where the runner finds steps.
- Edit files only with the file_editor tool. A new file: make its folder with `mkdir -p`, then `create` it (never
  `touch` it first: `create` does not overwrite). To add lines, use str_replace with old_str = the line just above
  where they go (copied exactly) and new_str = that line followed by the new lines. If str_replace is rejected, never
  send the same call again: view the file and include enough surrounding lines to match exactly once. Never copy the
  line numbers `view` shows into a file. A badly broken file of your own: `rm` it, then `create` it again whole (never a
  project file). No edits with shell commands, no backup copies, no git.
- Your sandbox has internet access and sudo works: install what your tests need yourself (the project's own install
  command first) and keep downloads out of /workspace/project.
- {check_rule}
- Keep working until all files are written; do not stop to explain your plan.
- {plain_values}

When you are done, reply with a short summary: each file you created or changed (with the case ids it covers), the
existing code you reused, and the shared code you added."""

_BATCH_NOTE = "\n(These are cases 1-{last} of {total}; the next ones follow once these are written.)"

_NEXT_BATCH_PROMPT = """Now write the tests for the next test cases ({first}-{last} of {total}), with the same rules as before.
{user_block}{state}
First look at the files you wrote so far and reuse them: call the page objects, helpers, fixtures, steps and keywords
that you or the project already wrote instead of writing them again (you may add to your own shared files). Put these
tests in NEW test files next to the ones you wrote, with names no file has yet; do not edit the test files you already
wrote, and never write again a case that is already written.
{cases}

When these are written, reply with a short summary: each file you created or changed (with the case ids it covers),
the existing code you reused, and the shared code you added."""

_FIX_PROMPT = """The check `{check}` fails with the files you wrote. Fix the cause of these errors (do not rename, copy
or rewrite tests to get around them), then reply with a short summary:
```
{errors}
```
{state}
Usual causes: an import that is missing, or whose relative path is wrong for your file's folder (work it out from your
file's folder to the file it points to, and check that this file exists); a name that does not exist (a function,
method, field, fixture, step or keyword: search the project for what really exists and use it, or add the missing piece
to the project's shared code, keeping it working for the existing tests); a wrong type; or text that is not code, such
as the line numbers `view` shows. Never add a build tool, compiler or dependency to get around an error the check
reports. A badly broken file of your own is easiest to rewrite: `rm` it, then `create` it again whole."""

_CHECK_LINE ="NG Automate checks your files after each batch, without running the tests, with: {check}"
_NO_CHECK_LINE = "This project has no quick check that builds or loads the tests without running them."
_CHECK_RULE = (
    "Before you finish, run `{check}` and fix every error it reports; do not run the whole suite. It only proves that "
    "your files load: also make sure that every file and name you import or call really exists."
)
_NO_CHECK_RULE = (
    "This project has no quick check: before you finish, read each new file once more and make sure that every file "
    "and name you import or call really exists. Do not run the whole suite."
)


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
    instructions: Sequence[str] = (),
) -> str:
    """The first message: the project, the first cases, the user's instructions and the rules (`total_cases` when more
    cases follow)."""
    if "Appium" in _values(fact_sheet.automation_tools):
        placement += MOBILE_RULE
    total = total_cases or len(cases)
    return _GENERATE_PROMPT.format(
        stack=_stack(fact_sheet),
        architecture=_text(findings.architecture_pattern) if findings else "unknown",
        important_paths=_important_paths(findings),
        command=command.command,
        check_line=_CHECK_LINE.format(check=command.check) if command.check else _NO_CHECK_LINE,
        briefing=_briefing_section(briefing),
        batch_note=_BATCH_NOTE.format(last=len(cases), total=total) if total > len(cases) else "",
        cases="\n".join(_case(case) for case in cases),
        user_block=user_instructions_block(instructions),
        placement=placement,
        screen_rule=VISIBLE_BROWSER_RULE,
        marker=UNVERIFIED_MARKER,
        check_rule=_CHECK_RULE.format(check=command.check) if command.check else _NO_CHECK_RULE,
        plain_values=PLAIN_TOOL_VALUES_RULE,
    )


def build_next_batch_prompt(
    cases: list[TestCaseSpecModel], first: int, total: int, instructions: Sequence[str] = (), state: str = ""
) -> str:
    """A later message in the same conversation: the next cases (`first` is the 1-based number of the first one).
    `instructions`: the user's (repeated, so "the same rules" never override them); `state`: where the work stands."""
    return _NEXT_BATCH_PROMPT.format(
        first=first,
        last=first + len(cases) - 1,
        total=total,
        user_block=user_instructions_block(instructions),
        state=f"\n{state}\n" if state else "",
        cases="\n".join(_case(case) for case in cases),
    )


def build_fix_prompt(errors: str, check: str, state: str = "") -> str:
    """The check's errors handed back to the generator, with where the work stands."""
    return _FIX_PROMPT.format(check=check, errors=errors, state=f"{state}\n" if state else "")


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
    "A sample of the project's own tests and shared code, exactly as they are (search the project for the rest). Reuse "
    "these exact fixtures, functions, classes, steps and keywords, and never call one that does not exist:\n{briefing}"
)


def _briefing_section(briefing: str) -> str:
    """The inlined project code, under its header. Empty when the generator has nothing to show."""
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
