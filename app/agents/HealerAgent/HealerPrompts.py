"""The healer's task message: the failure, the project and earlier attempts. How to fix it is the agent's call.

The rules are the same for every stack: they say how to work, never how one framework does it. The failure and the
project supply the details, with the check command NG Automate chose for it. The user's own instructions come first.
"""

from collections.abc import Sequence

from app.agents.TestGeneratorAgent.GeneratedTestPlacement import UNVERIFIED_MARKER
from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import HealOutcomeModel
from app.models.healMemoryModel import HealMemoryModel, HealMemoryResult
from app.models.testRunModel import FailureKind, TestCommandModel, TestRunResultModel
from app.utils.AgentPromptRules import PLAIN_TOOL_VALUES_RULE, TERMINAL_RULE, VISIBLE_BROWSER_RULE, user_instructions_block

MAX_OUTPUT_IN_PROMPT = 6000
MAX_PREVIOUS_SUMMARY_CHARS = 300
MAX_MEMORY_PROBLEM_CHARS = 200
MAX_MEMORY_SUMMARY_CHARS = 300
MAX_MEMORY_DIFF_CHARS = 600
_MEMORY_OUTCOME: dict[HealMemoryResult, str] = {
    HealMemoryResult.FIXED: "worked: the tests passed afterwards",
    HealMemoryResult.CHANGED: "fixed that problem; a different failure came next",
    HealMemoryResult.NOT_FIXED: "did not work: the same failure came back",
    HealMemoryResult.WORSE: "made it worse: fewer tests passed afterwards (or none ran)",
    HealMemoryResult.BLOCKED: "the healer was blocked by the environment",
    HealMemoryResult.UNFINISHED: "unfinished: the healer stopped before the tests ran again",
}
# The first line of a reply that reports a blocker.
BLOCKED_MARKER = "BLOCKED:"
OUTPUT_SECTION = "\nEnd of the output:\n```\n{output}\n```\n"
# Failures inside the tests (a locator, an error thrown by the test code): the failing tests and where they failed are
# the facts. The build log around them is mostly warnings, which lead a small model astray (in a real run a Selenium
# CDP-version warning sent the healer to pom.xml for a test that checked an element that had gone).
TEST_LEVEL_KINDS = frozenset({FailureKind.LOCATOR_FAILURE, FailureKind.TEST_CODE_FAILURE})
TESTS_ONLY_NOTE = (
    "\nThe lines above are the failing tests (class or module, test, message) and where they failed in the project.\n"
    "Start in those files. The build output is left out: here it holds only warnings, not the cause.\n"
)

_HEAL_PROMPT = """You are fixing a test-automation project in /workspace/project so that its tests can BUILD and RUN.

Project: {stack}
Test command (run from /workspace/project): {command}
Last result: exit code {exit_code}; {kind} — {reason}
Key lines:
{evidence}
{output_section}{previous}{memories}{user_block}
Find the cause yourself and fix it. You work in your own sandbox with a terminal: read the project, change code or
configuration, install or upgrade dependencies, download tools, drivers or browsers (sudo works). If one approach does
not work, try another one.

Rules (every change you make is kept; a change that breaks them is reported to the user):
- Fix only technical problems: dependencies and versions, build configuration, imports, syntax, tools and drivers,
  locators that do not find their element, and errors thrown by the test code itself.
- A build that does not compile or load lists its errors with file and line: fix all of them. A name, type or module it
  cannot find is usually a missing import (check that a relative path is right for the file's folder) or a member that
  does not exist: search the project for where it is defined and import it from there, add the missing member, or call
  one that exists. Do not stop while any build error is left.
- Test code that throws an error (a wrong key or argument, a missing method, a null value, a bad cast): fix the
  test code so it does what its test case describes, reusing the project's existing helpers, page objects and
  locator keys. Never change what a test checks.
- A locator that does not find its element, or any other problem you can fix: fix it. The user's test data (mounted
  read-only at /workspace/test-data) often lists several locators for the same element; when the one in the test does
  not match, use another the test data gives for that element (or one from the project's page objects, or from the
  live page — `ngauto-find-elements <page URL>` lists the page's elements with locators checked on the live page).
  Change only the locator, never the action or the assertion, and end the line with the comment
  {marker}: locator changed by the healer (in that file's comment syntax).
- NEVER delete, skip or disable tests; NEVER remove, weaken or comment out assertions or verification steps;
  NEVER wrap failing code in try/except or empty catch blocks.
- Fix the failing file itself: never write a new test, or a copy of a test under another name, to get around a failure.
- Do not try to fix failing assertions or application behaviour: that is not your job. One exception: a test in an
  `ngauto` folder was written by NG Automate's test generator from the user's test data (in /workspace/test-data).
  When such a test does not do what its test case says (it checks another element or value than the case's own
  verify step), correct it to follow its test case exactly.
- Start from the evidence above: it names the failing test and where in the project it failed. Never add debugging
  code (prints, sleeps, pauses) or a dependency the failure does not need.
- Change dependencies, scripts, compiler or test-runner settings only when the failure needs it; never add build tools,
  compilers or type packages just to check your fix, and never write compiled copies of files into the project.
- Keep the project's style and versions where possible.
- Change project files only with the file_editor tool (view, then str_replace; to add lines, str_replace with the line
  just above as old_str and that line plus the new lines as new_str (not `insert`: its first line loses its
  indentation); `create` for a new file). When str_replace is rejected (`did not appear verbatim`, or `Multiple
  occurrences ... Please ensure it is unique`), never send the same call again: view the file and include the whole
  enclosing method or statement, with enough surrounding lines, so old_str matches exactly once. `create` never
  overwrites a file: a test file in an `ngauto` folder (written by NG Automate's test generator) that is badly broken
  may be rewritten whole, by deleting it with `rm` in the terminal and then creating it again; edit every other file in
  place. For a new file, make its folder with `mkdir -p` first but never `touch` the file (the editor refuses to
  overwrite an existing file, so the file would stay empty). Do not edit them with sed, echo or other shell commands, do
  not create backup copies, and do not run git.
- Put downloads, temporary files and saved command output (`| tee`, `> file`) outside the project (for example in
  /tmp), never in /workspace/project.
- {screen_rule}
- {plain_values}
- {terminal_rule}
- You may run the test command to check your fix{check_hint}.
  As soon as the command gets past this error, stop and reply.

When you are done, reply with a short summary: what was wrong, what you changed, and why.

Only if the environment itself stops you and no other approach is left, start your reply with "{blocked}" and the
reason, then copy the exact command you ran and the error line it printed. A blocker is something nobody in this
sandbox can work around: a service the user must provide (a device farm such as BrowserStack, a test account, a
license), or a dependency or tool that cannot be obtained and has no alternative. Anything fixable — even by trying
another locator, another version or another approach — is not a blocker: fix it first. A failed attempt is not a
blocker: try another way first."""

# Only for a stack with a quick check (TestCommandModel.check), the one NG Automate chose for the project. It only
# builds or loads the tests, so it proves a fix of a build or load error, never one of a locator or of the test code.
_CHECK_HINT = (
    " (for a build or load error, the quicker check `{check}` is enough: it builds or loads the tests without running "
    "them)"
)


def build_heal_prompt(
    result: TestRunResultModel,
    command: TestCommandModel,
    fact_sheet: FactSheetModel,
    previous: Sequence[HealOutcomeModel] = (),
    memories: Sequence[HealMemoryModel] = (),
    instructions: Sequence[str] = (),
) -> str:
    """The failure, the project, earlier fixes and memories, then the user's `instructions` for this run (they come
    before the rules)."""
    evidence = "\n".join(f"- {line}" for line in result.classification.evidence) or "- (none)"
    return _HEAL_PROMPT.format(
        stack=describe_stack(fact_sheet),
        # With its setup: after changing what the project declares (e.g. requirements.txt) it must install again.
        command=f"{command.setup} && {command.command}" if command.setup else command.command,
        exit_code=result.exit_code,
        kind=result.classification.kind.value,
        reason=result.classification.reason,
        evidence=evidence,
        output_section=(
            TESTS_ONLY_NOTE
            if result.classification.kind in TEST_LEVEL_KINDS
            else OUTPUT_SECTION.format(output=result.output_tail[-MAX_OUTPUT_IN_PROMPT:].replace("```", "'''"))
        ),
        previous=_previous(previous),
        memories=_memories(memories),
        user_block=user_instructions_block(instructions),
        blocked=BLOCKED_MARKER,
        marker=UNVERIFIED_MARKER,
        screen_rule=VISIBLE_BROWSER_RULE,
        plain_values=PLAIN_TOOL_VALUES_RULE,
        terminal_rule=TERMINAL_RULE,
        check_hint=_CHECK_HINT.format(check=command.check) if command.check else "",
    )


def describe_stack(fact_sheet: FactSheetModel) -> str:
    """e.g. "Java · Maven · TestNG · Selenium"."""
    parts: list[str] = []
    for fact in (fact_sheet.primary_language, fact_sheet.build_tool, fact_sheet.test_frameworks,
                 fact_sheet.automation_tools, fact_sheet.bdd_tool):
        value = fact.value
        if value:
            parts.append(value if isinstance(value, str) else ", ".join(value))
    return " · ".join(parts) or "unknown"


def _previous(outcomes: Sequence[HealOutcomeModel]) -> str:
    """Earlier fixes in this run did not make the tests pass: say what they did (their changes are all still there)."""
    if not outcomes:
        return ""
    lines: list[str] = []
    for number, outcome in enumerate(outcomes, start=1):
        summary = " ".join(outcome.summary.split())
        if len(summary) > MAX_PREVIOUS_SUMMARY_CHARS:
            summary = summary[: MAX_PREVIOUS_SUMMARY_CHARS - 1] + "…"
        line = f"- Fix {number}: {summary}"
        if outcome.changes:
            line += f" Changed: {', '.join(change.path for change in outcome.changes)}."
        for path in sorted({violation.file for violation in outcome.violations}):
            rules = ", ".join(sorted({v.rule for v in outcome.violations if v.file == path}))
            line += f" Broke the rules in {path} ({rules})."
        if outcome.blocker_rejected:
            line += " Its blocker was not accepted: the quoted error was not in the output of any command it ran."
        lines.append(line)
    return (
        "\nEarlier fixes in this run did not make the tests pass (the result above is after them). "
        "Do not repeat an approach that did not work:\n" + "\n".join(lines) + "\n"
    )


def _memories(memories: Sequence[HealMemoryModel]) -> str:
    """Similar problems from earlier runs: what was tried and what came of it. Context for the agent, not orders."""
    if not memories:
        return ""
    blocks: list[str] = []
    for memory in memories:
        block = f"- {_MEMORY_OUTCOME[memory.result]} ({memory.stack}): {memory.failure_kind}"
        if memory.failure_evidence:
            block += f"\n  Error: {_short(memory.failure_evidence[0], MAX_MEMORY_PROBLEM_CHARS)}"
        block += f"\n  Tried: {_short(memory.fix_summary, MAX_MEMORY_SUMMARY_CHARS)}"
        if memory.changed_files:
            block += f"\n  Changed: {', '.join(memory.changed_files[:5])}"
        if memory.blocker:
            block += f"\n  Blocker: {_short(memory.blocker, MAX_MEMORY_SUMMARY_CHARS)}"
        if memory.next_run:
            block += f"\n  Next run: {memory.next_run}"
        # Only a fix that worked is worth reusing; the diff of one that did not would just be copied by a small model.
        if memory.diff and memory.succeeded:
            diff = memory.diff[:MAX_MEMORY_DIFF_CHARS].replace("```", "'''")
            block += f"\n  Diff:\n```\n{diff}\n```"
        blocks.append(block)
    return (
        "\nSimilar problems from earlier runs of this organization (found by similarity: they may not apply here; "
        "use what helps, and do not repeat what did not work):\n" + "\n".join(blocks) + "\n"
    )


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
