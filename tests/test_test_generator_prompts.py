"""The test generator's messages: the check NG Automate chose, the user's instructions before the rules, where the
work stands, and nothing from another stack in a Playwright TypeScript project's messages. Plain strings, no LLM."""

from collections.abc import Sequence

import pytest

from app.agents.RunnerAgent.TestCommandResolver import resolve_test_command
from app.agents.TestGeneratorAgent.GeneratedTestPlacement import placement_rule
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


def _first(command: TestCommandModel = COMMAND, instructions: Sequence[str] = ()) -> str:
    """The first message for the Playwright project: TD-001 now, one more case later."""
    return build_generate_prompt(
        [LOGIN], PLAYWRIGHT_TS, FINDINGS, command, placement_rule(PLAYWRIGHT_TS), 2, BRIEFING, instructions
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


# ---------- generic: nothing from another stack ----------

def test_a_playwright_projects_messages_carry_nothing_from_other_stacks() -> None:
    assert COMMAND.check
    messages = {
        "first": _first(instructions=["reuse the login fixture"]),
        "next batch": build_next_batch_prompt([LOGOUT], 2, 2, ["reuse the login fixture"], STATE),
        "fix": build_fix_prompt(ERRORS, COMMAND.check, STATE),
    }

    found = {name: [detail for detail in OTHER_STACKS if detail in text] for name, text in messages.items()}

    assert found == {"first": [], "next batch": [], "fix": []}
