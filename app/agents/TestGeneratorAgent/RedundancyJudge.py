"""Whether a test an earlier generation wrote in the 'ngauto' folder already does what a test case says, judged by
meaning: one small LLM call per case (no agent, no tools), before the generator writes anything.

The test data can come in any form (a story in prose, Gherkin, a sheet, a JSON export), with or without names, ids,
values or locators, so neither ids nor matching values can tell whether a case is already tested: "a shopper signs in
and sees the products" and a test named SignInTest can be the same test. The model compares what the case describes
with what the earlier tests do and names the one that already does all of it, or none.

Its answer is checked like the agent's reports (ExistingCoverage.check_claim: a test file in an 'ngauto' folder
before this run, and the test it names is in it), and it must be one of the tests it was shown. A failed call, an
unusable answer or a check that fails leaves the case in: a wrong skip loses a test without anyone noticing.
"""

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.TestGeneratorAgent.ExistingCoverage import CoverageClaim, check_claim, parse_path
from app.models.llmModel import LlmConfigModel
from app.models.testDataModel import TestCaseSpecModel
from app.models.tokenUsageModel import TokenUsageModel
from app.utils.StructuredOutput import complete_json

logger = logging.getLogger(__name__)

Complete = Callable[..., Any]

# The earlier tests shown for one case: the most alike first, each cut to a size the model reads well.
MAX_TESTS_CHARS = 12_000
MAX_TEST_CHARS = 4_000
JUDGE_TIMEOUT_SECONDS = 180

INSTRUCTIONS = """You check whether a test case is already automated. You get one test case (from the user's test data,
in any wording) and automated tests that already exist. Decide whether ONE of these tests already does everything the
case describes: the same flow, the same data and the same checks (the test may do more).

Judge by what the test code does, never by names, ids or wording: a case written in other words can be the same test,
and a test named like the case can test something else. A test of the same page or flow with other data, another user
or another check is NOT the same (a valid login and a refused login are different tests). When you are not sure, the
answer is no.

Reply with JSON only: covered (true or false); file (the test's file exactly as shown after "File:", or "" when not
covered); test (the test's name exactly as the file writes it, or ""); reason (one short sentence)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "boolean"},
        "file": {"type": "string"},
        "test": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["covered", "file", "test", "reason"],
    "additionalProperties": False,
}

_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


@dataclass(frozen=True)
class Verdict:
    """The judge's answer for one case: the earlier test that covers it (None: none does), why, and what went wrong
    when it could not be used (`problem`: the call failed, or its answer did not hold)."""

    claim: CoverageClaim | None
    reason: str = ""
    problem: str | None = None


def judge_case(
    llm_config: LlmConfigModel,
    case: TestCaseSpecModel,
    tests: Mapping[str, str],
    files_before: Mapping[str, str],
    project_root: Path,
    complete: Complete = complete_json,
    on_usage: Callable[[TokenUsageModel], None] | None = None,
) -> Verdict:
    """Whether one of the earlier `tests` (ExistingCoverage.earlier_test_files) already does what the case says.
    Never raises."""
    shown = _shown(case, tests)
    if not shown:
        return Verdict(None)
    text = f"Test case:\n{describe_case(case)}\n\nExisting tests:\n\n" + "\n\n".join(
        f"File: {path}\n{body}" for path, body in shown.items()
    )
    try:
        answer = complete(llm_config, INSTRUCTIONS, text, "already_tested", SCHEMA, _parse,
                          timeout_seconds=JUDGE_TIMEOUT_SECONDS, on_usage=on_usage)
    except Exception as error:  # a failed check never stops the generation: the case is written
        logger.warning("Could not check whether case %s is already tested: %s", case.id, type(error).__name__)
        return Verdict(None, problem=f"the check failed ({str(error)[:200] or type(error).__name__})")
    if not answer["covered"]:
        return Verdict(None, answer["reason"])
    claim = CoverageClaim(case.id, parse_path(answer["file"]), answer["test"].strip())
    if claim.path not in shown:
        return Verdict(None, answer["reason"], f"it named {claim.path or 'no file'}, which is not one of the tests shown")
    problem = check_claim(claim, files_before, project_root)
    if problem is not None:
        return Verdict(None, answer["reason"], problem)
    return Verdict(claim, answer["reason"])


def describe_case(case: TestCaseSpecModel) -> str:
    """The case as the test data gives it: title, what it is about, preconditions, steps (in its own words where it
    has them) and data."""
    lines = [f"Title: {case.title}"]
    if case.description:
        lines.append(f"About: {case.description}")
    lines.extend(f"Given: {precondition}" for precondition in case.preconditions)
    for number, step in enumerate(case.steps, start=1):
        parts = [step.text or step.action]
        for label, value in (("element", step.target or step.target_hint), ("value", step.value),
                             ("expected", step.expected)):
            if value:
                parts.append(f'{label}="{value}"')
        lines.append(f"{number}. " + " ".join(parts))
    if case.test_data:
        lines.append("Data: " + ", ".join(f"{key}={value}" for key, value in case.test_data.items()))
    return "\n".join(lines)


def _shown(case: TestCaseSpecModel, tests: Mapping[str, str]) -> dict[str, str]:
    """The tests to show for the case: all of them while they fit, else the ones sharing the most words with it."""
    words = _words(describe_case(case))
    ranked = sorted(tests, key=lambda path: (-len(words & _words(f"{path}\n{tests[path]}")), path))
    shown: dict[str, str] = {}
    used = 0
    for path in ranked:
        body = tests[path][:MAX_TEST_CHARS]
        if used + len(body) > MAX_TESTS_CHARS and shown:
            break
        shown[path] = body
        used += len(body)
    return shown


def _words(text: str) -> set[str]:
    """Lowercase words of three letters or more, camelCase and snake_case split (signInWorks: sign, works)."""
    return {word.lower() for word in _WORD.findall(text) if len(word) >= 3}


def _parse(reply: str) -> dict[str, Any]:
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON: {error}") from None
    if not isinstance(data, dict) or not isinstance(data.get("covered"), bool):
        raise ValueError("covered must be true or false")
    return {"covered": data["covered"], "file": str(data.get("file") or ""), "test": str(data.get("test") or ""),
            "reason": str(data.get("reason") or "")[:300]}
