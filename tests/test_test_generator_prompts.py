"""The test generator's messages: the check NG Automate chose, the user's instructions before the rules, where the
work stands, the same rules for every stack, and nothing from another stack in a Playwright TypeScript project's
messages. Plain strings, no LLM."""

from collections.abc import Sequence

import pytest

from app.agents.RunnerAgent.TestCommandResolver import resolve_test_command
from app.agents.TestGeneratorAgent.TestGeneratorPrompts import (
    build_fix_prompt,
    build_generate_prompt,
    build_next_batch_prompt,
)
from app.models.analyzerModel import (
    AnalyzerFindingsModel,
    Confidence,
    FactModel,
    FactSheetModel,
    FactSource,
    ImportantPathModel,
)
from app.models.testDataModel import TestCaseSpecModel, TestStepModel
from app.models.testRunModel import TestCommandModel
from app.utils.AgentPromptRules import TERMINAL_RULE

RULES = "How to work"
USER_FIRST = "they come first"
NO_CHECK = "no quick check"
# What other stacks need (Maven, a Java compiler error, Selenium lookups, a Java glob, an editor command the generator
# no longer uses, a Java base class): a Playwright TypeScript project's messages never name them.
OTHER_STACKS = (
    "mvn", "cannot find symbol", "driver.findElement", "By.id", "**/*.java", "insert_line", "base test class",
)


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=["package.json"], confidence=Confidence.HIGH)


PLAYWRIGHT_TS = FactSheetModel(
    total_files=6, language_files={"TypeScript": 5}, primary_language=_fact("TypeScript"), build_tool=_fact("npm"),
    test_frameworks=_fact(["Playwright Test"]), automation_tools=_fact(["Playwright"]), bdd_tool=_fact(None),
    marker_files=["package.json", "package-lock.json", "playwright.config.ts"], test_dirs=["tests"],
    feature_file_count=0, top_level_tree=["package.json", "playwright.config.ts", "tests/"],
)
FINDINGS = AnalyzerFindingsModel(
    project_summary=_fact("Shop UI tests"),
    architecture_pattern=_fact("Page Object Model with Playwright fixtures"),
    test_command=_fact("npx playwright test"),
    reporting_tools=_fact(["Playwright HTML reporter"]),
    important_paths=[ImportantPathModel(path="tests/pages", role="page objects")],
)
# What NG Automate runs for this project: its check lists the tests without running them.
COMMAND = resolve_test_command(PLAYWRIGHT_TS)
NO_CHECK_COMMAND = TestCommandModel(tool="npm", command="npm test")
BRIEFING = """--- tests/pages/LoginPage.ts ---
import { Page } from '@playwright/test';

export class LoginPage {
  constructor(private readonly page: Page) {}

  async login(user: string, password: string) {
    await this.page.fill('#username', user);
    await this.page.fill('#password', password);
  }
}"""
LOGIN = TestCaseSpecModel(
    id="TD-001",
    title="Valid login",
    steps=[
        TestStepModel(action="open", target="https://shop.example/login"),
        TestStepModel(action="type", target="id=username", value="admin"),
        TestStepModel(action="verify_text", target="css=h1", expected="Dashboard"),
    ],
)
LOGOUT = TestCaseSpecModel(
    id="TD-002",
    title="Logout",
    preconditions=["The user is logged in"],
    steps=[TestStepModel(action="click", target_hint="Logout button", needs_locator=True)],
)
STATE = (
    "Where the work stands (worked out from the project's files: trust it over your memory):\n"
    "- Cases already written: TD-001 in tests/ngauto/td-001-valid-login.spec.ts."
)
ERRORS = (
    "Error: tests/ngauto/td-002-logout.spec.ts:3:10: Cannot find module '../pages/LogoutPage'\n"
    "  { code: 'MODULE_NOT_FOUND' }"
)


def _first(command: TestCommandModel = COMMAND, instructions: Sequence[str] = (), broken_before: bool = False) -> str:
    """The first message for the Playwright project: TD-001 now, one more case later."""
    return build_generate_prompt(
        [LOGIN], PLAYWRIGHT_TS, FINDINGS, command, 2, BRIEFING, instructions, broken_before=broken_before
    )


# ---------- the check ----------

def test_the_first_message_names_the_projects_check_in_its_check_line_and_its_rule() -> None:
    assert COMMAND.check  # NG Automate has a quick check for Playwright projects

    head, rules = _first().split(RULES, 1)

    assert COMMAND.check in head  # what NG Automate runs after each batch
    assert COMMAND.check in rules  # what the model runs before it finishes
    assert NO_CHECK not in head + rules


def test_without_a_quick_check_the_first_message_says_so_instead() -> None:
    head, rules = _first(NO_CHECK_COMMAND).split(RULES, 1)

    assert NO_CHECK in head and NO_CHECK in rules
    assert "`None`" not in rules and ": None" not in head


def test_a_project_that_did_not_build_before_is_not_the_generators_task() -> None:
    # A real run: the project did not build (jars it needs are not in git). Told to fix every error the check reports,
    # the generator cut its pom.xml down with sed and downloaded jars, then asked the user for jar URLs in its summary.
    head, rules = _first(broken_before=True).split(RULES, 1)
    rules = " ".join(rules.split())

    assert f"`{COMMAND.check}` failed before your work" in head
    assert "NG Automate checks your files after each batch" not in head  # no check runs after a batch
    assert "that problem is not your task" in rules
    assert "do not rewrite build files or download libraries to fix it" in rules
    assert "never remove existing dependencies or configuration" in rules
    assert "Write the tests with the code that exists (search the project for it)" in rules
    assert "In your summary, also say what blocks the build" in rules
    assert "fix every error it reports" not in rules
    assert "that problem is not your task" not in _first() + _first(NO_CHECK_COMMAND, broken_before=True)


# ---------- the user's instructions ----------

def test_the_users_instructions_come_first_before_the_rules() -> None:
    prompt = _first(instructions=["Write the new tests as BDD feature files", "  reuse the {loggedInPage} fixture  "])

    head, rules = prompt.split(RULES, 1)
    assert USER_FIRST in head
    assert "Write the new tests as BDD feature files" in head
    assert "reuse the {loggedInPage} fixture" in head  # its braces are kept as written
    assert "reuse the {loggedInPage} fixture  " not in head  # trimmed
    assert "Write the new tests as BDD feature files" not in rules


@pytest.mark.parametrize("instructions", [(), ("", "   ")])
def test_without_instructions_no_message_has_a_user_block(instructions: Sequence[str]) -> None:
    assert USER_FIRST not in _first(instructions=instructions)
    assert USER_FIRST not in build_next_batch_prompt([LOGOUT], 2, 2, instructions, STATE)


# ---------- later batches and build fixes ----------

def test_a_later_batch_carries_where_the_work_stands_and_the_users_instructions() -> None:
    prompt = build_next_batch_prompt([LOGOUT], 2, 2, ["reuse the login fixture"], STATE)

    assert STATE in prompt
    assert USER_FIRST in prompt and "reuse the login fixture" in prompt
    assert "TD-002" in prompt
    assert "https://shop.example/login" not in prompt  # the first batch's case is not asked for again


def test_the_fix_message_carries_the_errors_the_check_and_where_the_work_stands() -> None:
    assert COMMAND.check

    prompt = build_fix_prompt(ERRORS, COMMAND.check, STATE)

    assert ERRORS in prompt  # braces in the errors are kept as written
    assert COMMAND.check in prompt
    assert STATE in prompt


# ---------- generic: one fixed folder, the rest is the framework's ----------

# Per-stack placement and framework know-how the rules no longer give: where a framework's files go, how it works.
PER_STACK = (
    "src/test", "package named", "Class names end with", "test_<case>.py", ".robot", "new .feature file",
    "Feature files", "glue", "where the runner finds steps", "Appium", "accessibility id",
)


def _sheet(build: str, frameworks: list[str], bdd: str | None = None, language: str = "Java") -> FactSheetModel:
    return PLAYWRIGHT_TS.model_copy(update={
        "primary_language": _fact(language), "build_tool": _fact(build), "test_frameworks": _fact(frameworks),
        "automation_tools": _fact(["Selenium"]), "bdd_tool": _fact(bdd),
    })


def test_the_rules_are_the_same_for_every_stack_one_fixed_folder_and_the_rest_left_to_the_framework() -> None:
    stacks = [
        PLAYWRIGHT_TS,
        _sheet("Maven", ["TestNG"]),
        _sheet("Maven", ["TestNG"], "Cucumber"),
        _sheet("pip", ["pytest"], language="Python"),
        PLAYWRIGHT_TS.model_copy(update={"automation_tools": _fact(["Appium"])}),
    ]

    rules = {
        build_generate_prompt([LOGIN], sheet, FINDINGS, COMMAND, 2, BRIEFING).split(RULES, 1)[1] for sheet in stacks
    }

    assert len(rules) == 1  # nothing in the rules depends on the stack
    [text] = rules
    assert "Put your new test files only in a new folder named 'ngauto'" in text
    assert "Every other file you need goes where this framework expects it" in text
    assert "Add whatever the framework needs to find and run your new tests" in text
    assert [detail for detail in PER_STACK if detail in text] == []


def test_done_means_the_projects_own_test_command_finds_and_runs_the_new_tests_with_or_without_a_check() -> None:
    # A real run: the build check passed, but the feature files sat where the build never copies them and there was no
    # runner, so not one scenario ran. Building or loading the new tests is not enough.
    for command, broken_before in ((COMMAND, False), (NO_CHECK_COMMAND, False), (COMMAND, True)):
        rules = _first(command, broken_before=broken_before).split(RULES, 1)[1]
        assert "the project's own test command would find and run your new tests" in rules
        assert "building or loading them is not enough" in rules
        assert "without running the tests" in rules and "Do not run the whole suite" in rules


def test_the_rules_ask_for_names_the_project_runs_and_for_scratch_output_outside_the_project() -> None:
    # Real runs: 8 of 11 classes were not named like the project's tests, so its plain test command skipped them; a
    # test named after a Hindi title could not be chosen to run; `| tee result.txt` was left in the project root.
    rules = " ".join(_first().split(RULES, 1)[1].split())

    assert "the way the project's normal test command finds tests" in rules
    assert "a plain run of all its tests runs yours too" in rules
    assert "Use only ASCII letters, digits, `_`, `-` and `.` in the names" in rules  # the dot: login.spec.ts
    assert "saved command output (`| tee`, `> file`) outside the project (for example in /tmp)" in rules


def test_the_rules_say_the_terminal_can_check_an_answer_that_looks_wrong() -> None:
    # A real run: the grep tool answered "No files found" for files that were there, and the generator believed it.
    assert TERMINAL_RULE in _first().split(RULES, 1)[1]


# ---------- generic: nothing from another stack ----------

def test_a_playwright_projects_messages_carry_nothing_from_other_stacks() -> None:
    assert COMMAND.check
    messages = {
        "first": _first(instructions=["reuse the login fixture"]),
        "first, not building": _first(instructions=["reuse the login fixture"], broken_before=True),
        "next batch": build_next_batch_prompt([LOGOUT], 2, 2, ["reuse the login fixture"], STATE),
        "fix": build_fix_prompt(ERRORS, COMMAND.check, STATE),
    }

    found = {name: [detail for detail in OTHER_STACKS if detail in text] for name, text in messages.items()}

    assert found == {"first": [], "first, not building": [], "next batch": [], "fix": []}
