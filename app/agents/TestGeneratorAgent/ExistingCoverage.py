"""Test cases that an earlier generation's tests already cover: the project's 'ngauto' folder holds the tests NG
Automate wrote before, and a case must not get a second test for what one of them already does. Plain code, no LLM.

Only the 'ngauto' folder counts (what NG Automate wrote), never the project's own tests. Redundancy is decided by what
the tests do, never by ids or names: new test data may give the same case another id, or none, and the same id may now
mean another case.

The first check is exact. A case's facts are the values its steps type or select, the texts they check and its test
data: what any test of it must carry, whatever the framework. An earlier test file holding all of them (at least two)
does the same or more. Locators and URLs do not count: the project's page objects and configuration usually hold them,
not the test. A case whose test data gives no such values (a story in prose) is left to RedundancyJudge, which compares
by meaning; its answer must pass check_claim.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.agents.RunnerAgent.TestCommandResolver import is_test_file
from app.agents.TestGeneratorAgent.GeneratedTestPlacement import GENERATED_DIR
from app.models.testDataModel import TestCaseSpecModel
from app.services.sandboxService.SandboxService import SANDBOX_PROJECT_DIR

# Why a case gets no new test (the user sees it with the case).
SAME_VALUES = "an earlier generation already tests it in {path} (it has all of the case's values and checks)"
JUDGED_SAME = "an earlier generation already tests it: {test} in {path} ({why})"
# Fewer facts than this never make a test "the same": one value proves little.
MIN_FACTS_FOR_SAME = 2
MIN_FACT_CHARS = 3
MIN_TEST_NAME_CHARS = 3

# Why a claim that an earlier test covers a case does not hold.
NOT_EARLIER = "{path} was not in the '" + GENERATED_DIR + "' folder before this run"
DELETED = "{path} is no longer there"
NOT_A_TEST_FILE = "{path} is not a test file"
NAME_NOT_FOUND = "{path} has no test named '{test}'"
NAME_TOO_SHORT = "it gives no test name"

_QUOTES = "`'\"*"
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class CoverageClaim:
    """That an earlier test covers a case: the case id, the file (project-relative) and the test's name."""

    case_id: str
    path: str
    test: str


def is_generated_path(path: str) -> bool:
    """Whether a project-relative path is inside a folder (package) named 'ngauto'."""
    return GENERATED_DIR in path.split("/")[:-1]


def earlier_test_files(files_before: Mapping[str, str]) -> dict[str, str]:
    """The test files in an 'ngauto' folder of the project before this run (path -> text, sorted by path)."""
    return {path: text for path, text in sorted(files_before.items())
            if is_generated_path(path) and is_test_file(path, lambda text=text: text)}


def case_facts(case: TestCaseSpecModel) -> list[str]:
    """What a test of the case must carry, normalized and without repeats: the values its steps type or select, the
    texts they check and its test data (shorter ones prove nothing)."""
    facts = [_normalized(text) for text in _raw_facts(case)]
    return list(dict.fromkeys(fact for fact in facts if len(fact) >= MIN_FACT_CHARS))


def same_values_test(case: TestCaseSpecModel, tests: Mapping[str, str]) -> str | None:
    """The first earlier test file (earlier_test_files) that holds every value and check of the case, or None: also
    when the case has too few of them to tell (a story in prose), or a value too short to count (a cart badge that
    "shows 1": the valid-login test holds every other value of such a case, yet does not check the badge; the judge
    decides)."""
    facts = case_facts(case)
    if len(facts) < MIN_FACTS_FOR_SAME or any(len(_normalized(text)) < MIN_FACT_CHARS for text in _raw_facts(case)):
        return None
    for path, text in tests.items():
        normalized = _normalized(text)
        if all(_holds(normalized, fact) for fact in facts):
            return path
    return None


def _raw_facts(case: TestCaseSpecModel) -> list[str]:
    """The case's values, checked texts and test data as given (non-empty)."""
    raw = [part for step in case.steps for part in (step.value, step.expected)] + list(case.test_data.values())
    return [text for text in raw if text and text.strip()]


def check_claim(claim: CoverageClaim, files_before: Mapping[str, str], project_root: Path) -> str | None:
    """None when the claim holds (a test file in an 'ngauto' folder named that test before this run, and is still
    there), else why not. Whether that test does what the case says is the judge's call."""
    if len(claim.test) < MIN_TEST_NAME_CHARS:
        return NAME_TOO_SHORT
    text = files_before.get(claim.path)
    if text is None or not is_generated_path(claim.path):
        return NOT_EARLIER.format(path=claim.path or "the file")
    if not (project_root / claim.path).is_file():
        return DELETED.format(path=claim.path)
    if not is_test_file(claim.path, lambda: text):
        return NOT_A_TEST_FILE.format(path=claim.path)
    if not _names(text, claim.test):
        return NAME_NOT_FOUND.format(path=claim.path, test=claim.test)
    return None


def parse_path(path: str) -> str:
    """The path from the project root, "/"-separated, however a model wrote it (absolute in the sandbox, ./x)."""
    path = _unquote(path).replace("\\", "/").strip()
    prefix = f"{SANDBOX_PROJECT_DIR.rstrip('/')}/"
    if path.startswith(prefix):
        path = path[len(prefix):]
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def _normalized(text: str) -> str:
    """Case, runs of spaces, escapes and the kind of quotes aside: a value reads the same in any language's string."""
    return _SPACES.sub(" ", text.replace("\\", "").replace('"', "'")).strip().casefold()


def _holds(text: str, fact: str) -> bool:
    """Whether the (normalized) text holds the fact as a whole value: "admin" is not in "administrator"."""
    return re.search(rf"(?<!\w){re.escape(fact)}(?!\w)", text) is not None


def _names(text: str, test: str) -> bool:
    """Whether the text names the test as a whole name (case and runs of spaces aside): "login" is not "loginFails"."""
    words = [re.escape(word) for word in _SPACES.split(test.strip())]
    return re.search(rf"(?<!\w){r'\s+'.join(words)}(?!\w)", text, re.IGNORECASE) is not None


def _unquote(text: str) -> str:
    return text.strip().strip(_QUOTES).strip()
