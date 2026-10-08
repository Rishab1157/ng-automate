"""The healer's task message: the failure, the project and earlier attempts. How to fix it is the agent's call."""

from collections.abc import Sequence

from app.models.analyzerModel import FactSheetModel
from app.models.healerModel import HealOutcomeModel
from app.models.healMemoryModel import HealMemoryModel, HealMemoryResult
from app.models.testRunModel import TestCommandModel, TestRunResultModel

MAX_OUTPUT_IN_PROMPT = 6000
MAX_PREVIOUS_SUMMARY_CHARS = 300
MAX_MEMORY_PROBLEM_CHARS = 200
MAX_MEMORY_SUMMARY_CHARS = 300
MAX_MEMORY_DIFF_CHARS = 600
_MEMORY_OUTCOME: dict[HealMemoryResult, str] = {
    HealMemoryResult.FIXED: "worked: the tests passed afterwards",
    HealMemoryResult.CHANGED: "fixed that problem; a different failure came next",
    HealMemoryResult.NOT_FIXED: "did not work: the same failure came back",
    HealMemoryResult.BLOCKED: "the healer was blocked by the environment",
    HealMemoryResult.UNFINISHED: "unfinished: the healer stopped before the tests ran again",
}
# The first line of a reply that reports a blocker.
BLOCKED_MARKER = "BLOCKED:"

_HEAL_PROMPT = """You are fixing a test-automation project in /workspace/project so that its tests can BUILD and RUN.

Project: {stack}
Test command (run from /workspace/project): {command}
Last result: exit code {exit_code}; {kind} — {reason}
Key lines:
{evidence}

End of the output:
```
{output}
```
{previous}{memories}
Find the cause yourself and fix it. You work in your own sandbox with a terminal: read the project, change code or
configuration, install or upgrade dependencies, download tools, drivers or browsers. If one approach does not work,
try another one.

Rules (a change that breaks them is rolled back automatically):
- Fix only technical problems: dependencies and versions, build configuration, imports, syntax, tools and drivers.
- NEVER delete, skip or disable tests; NEVER remove, weaken or comment out assertions or verification steps;
  NEVER wrap failing code in try/except or empty catch blocks.
- Do not try to fix failing assertions or application behaviour: that is not your job.
- Keep the project's style and versions where possible.
- Change project files only with the file_editor tool (view, then str_replace). Do not edit them with sed, echo or
  other shell commands, do not create backup copies, and do not run git.
- You may run the test command (or a quicker build step such as `mvn -q -DskipTests compile`) to check your fix.
  As soon as the command gets past this error, stop and reply.

When you are done, reply with a short summary: what was wrong, what you changed, and why.

Only if the environment itself stops you and no other approach is left (for example the network or a permission
forbids a download you need), start your reply with "{blocked}" and the reason, then copy the exact command you ran
and the error line it printed. A failed attempt is not a blocker: try another way first."""


def build_heal_prompt(
    result: TestRunResultModel,
    command: TestCommandModel,
    fact_sheet: FactSheetModel,
    previous: Sequence[HealOutcomeModel] = (),
    memories: Sequence[HealMemoryModel] = (),
) -> str:
    evidence = "\n".join(f"- {line}" for line in result.classification.evidence) or "- (none)"
    return _HEAL_PROMPT.format(
        stack=describe_stack(fact_sheet),
        command=command.command,
        exit_code=result.exit_code,
        kind=result.classification.kind.value,
        reason=result.classification.reason,
        evidence=evidence,
        output=result.output_tail[-MAX_OUTPUT_IN_PROMPT:].replace("```", "'''"),
        previous=_previous(previous),
        memories=_memories(memories),
        blocked=BLOCKED_MARKER,
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
    """Earlier fixes in this run did not make the tests pass: say what they did, and what was rolled back."""
    if not outcomes:
        return ""
    lines: list[str] = []
    for number, outcome in enumerate(outcomes, start=1):
        summary = " ".join(outcome.summary.split())
        if len(summary) > MAX_PREVIOUS_SUMMARY_CHARS:
            summary = summary[: MAX_PREVIOUS_SUMMARY_CHARS - 1] + "…"
        line = f"- Fix {number}: {summary}"
        if outcome.changes:
            line += f" Kept changes: {', '.join(change.path for change in outcome.changes)}."
        for path in outcome.reverted_files:
            rules = ", ".join(sorted({v.rule for v in outcome.violations if v.file == path}))
            line += f" Rolled back {path} ({rules})."
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
        if memory.diff:
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
