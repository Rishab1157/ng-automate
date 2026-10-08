"""Edits the healer may not make: they would hide a failure instead of fixing it, or leave the project worse.

The healer fixes technical problems (dependencies, compile errors, configuration). Removing assertions, deleting
or skipping tests, commenting checks out, or swallowing failures makes a red build green without fixing anything,
so every change it makes to a test file is checked here and rejected when it does any of that.

Small models also make clumsy edits: a build file that no longer parses (e.g. XML mangled by a sed command) or
backup copies left behind. Those are caught here too, so they can be rolled back.
"""

import json
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import PurePosixPath

from app.models.healerModel import FileChangeModel, GuardViolationModel
from app.utils.ProjectSnapshot import ProjectSnapshot

MAX_LINE_CHARS = 200

_TEST_FILE_NAMES = re.compile(
    r"(Test|Tests|Steps|StepDefinitions|IT)\.(java|kt|groovy|cs)$|^test_.*\.py$|_test\.py$"
    r"|\.(spec|test)\.(ts|tsx|js|jsx|mjs|cjs)$|\.feature$|\.robot$",
)
_TEST_FOLDERS = {"test", "tests", "spec", "specs", "steps", "__tests__", "e2e", "features", "stepdefinitions"}

_ASSERTION = re.compile(
    r"\bassert\w*\s*[\s(]|\bAssert\.|\bassertThat\b|\bexpect\s*\(|\.should\s*\(|\bverify\s*\(|\.to(Be|Equal|Have)\w*\s*\("
    r"|^\s*(Then|And|But)\s",
)
_TEST_DECLARATION = re.compile(
    r"@Test\b|^\s*(async\s+)?def\s+test_|^\s*(it|test)\s*\(|^\s*Scenario( Outline)?:|\[(Test|Fact|TestMethod)\]"
)
_SKIP = re.compile(
    r"@Ignore\b|@Disabled\b|enabled\s*=\s*false|@pytest\.mark\.skip|pytest\.skip\(|\b(it|test|describe)\.skip\s*\("
    r"|\bxit\s*\(|\bxdescribe\s*\(|^\s*@skip\b|@unittest\.skip|\[Ignore\]",
)
_COMMENT = re.compile(r"^\s*(//|#|/\*|\*|<!--|--)")
_STRAY_FILE = re.compile(r"\.(bak|backup|orig|rej|swp)$|~$", re.IGNORECASE)
# Declarations that could make the XML parser expand entities: such files are not checked.
_XML_DECLARATIONS = ("<!DOCTYPE", "<!ENTITY")
_SWALLOW = re.compile(r"except\b[^:]*:\s*(pass)?\s*$|catch\s*\([^)]*\)\s*\{\s*\}")


def is_test_file(path: str) -> bool:
    pure = PurePosixPath(path.replace("\\", "/"))
    if _TEST_FILE_NAMES.search(pure.name):
        return True
    return any(part.lower() in _TEST_FOLDERS for part in pure.parts[:-1])


def check_test_integrity(path: str, before: str, after: str) -> list[GuardViolationModel]:
    """Violations in one changed file. Non-test files are never checked ([])."""
    if not is_test_file(path):
        return []
    removed, added = _diff_lines(before, after)
    violations: list[GuardViolationModel] = []

    for line in removed:
        if _COMMENT.match(line):
            continue
        if _ASSERTION.search(line):
            violations.append(_violation(path, "assertion_removed", line))
        elif _TEST_DECLARATION.search(line):
            violations.append(_violation(path, "test_removed", line))

    for line in added:
        if _SKIP.search(line):
            violations.append(_violation(path, "test_skipped", line))
        elif _COMMENT.match(line) and _ASSERTION.search(_COMMENT.sub("", line, count=1)):
            violations.append(_violation(path, "assertion_commented", line))
        elif _SWALLOW.search(line) and _swallow_is_new(line, before):
            violations.append(_violation(path, "failure_swallowed", line))
    return violations


def check_changes(
    snapshot: ProjectSnapshot,
    changes: list[FileChangeModel],
    *,
    allow_deletes: bool = True,
    additions_only: bool = False,
) -> list[GuardViolationModel]:
    """Every guard, over everything an agent changed since `snapshot`. A deleted file counts as every line removed.

    `additions_only`: existing files may only gain lines (the test generator adds code; it never edits what is there).
    """
    violations: list[GuardViolationModel] = []
    for change in changes:
        before = snapshot.content(change.path) or ""
        after = "" if change.change == "deleted" else (snapshot.root / change.path).read_text(encoding="utf-8")
        if change.change == "added":
            violations.extend(check_new_file(change.path))
        elif change.change == "modified":
            violations.extend(check_build_file(change.path, before, after))
            if additions_only:
                violations.extend(check_only_additions(change.path, before, after))
        elif not allow_deletes:
            violations.append(_violation(change.path, "file_deleted", change.path))
            continue
        violations.extend(check_test_integrity(change.path, before, after))
    return violations


def check_only_additions(path: str, before: str, after: str) -> list[GuardViolationModel]:
    """An existing file whose lines were changed or removed (moving or re-indenting a line is fine)."""
    removed, _ = _diff_lines(before, after)
    return [_violation(path, "existing_code_changed", removed[0])] if removed else []


def check_new_file(path: str) -> list[GuardViolationModel]:
    """A file the healer added: backup copies and editor leftovers are not part of a fix."""
    if _STRAY_FILE.search(PurePosixPath(path).name):
        return [_violation(path, "stray_file", path)]
    return []


def check_build_file(path: str, before: str, after: str) -> list[GuardViolationModel]:
    """An XML or JSON file (pom.xml, testng.xml, package.json, ...) that parsed before the change and no longer does."""
    suffix = PurePosixPath(path).suffix.lower()
    parses = _xml_parses if suffix in (".xml", ".pom") else _json_parses if suffix == ".json" else None
    if parses is None or not parses(before) or parses(after):
        return []
    _, added = _diff_lines(before, after)
    return [_violation(path, "build_file_broken", added[0] if added else "(file emptied)")]


def _xml_parses(text: str) -> bool:
    if any(declaration in text for declaration in _XML_DECLARATIONS):
        return True  # not checked
    try:
        ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return False
    return True


def _json_parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _diff_lines(before: str, after: str) -> tuple[list[str], list[str]]:
    """Lines removed and added, as multisets of stripped lines: moving or re-indenting a line is not a change."""
    old = Counter(line.strip() for line in before.splitlines() if line.strip())
    new = Counter(line.strip() for line in after.splitlines() if line.strip())
    return list((old - new).elements()), list((new - old).elements())


def _swallow_is_new(line: str, before: str) -> bool:
    return line not in {existing.strip() for existing in before.splitlines()}


def _violation(path: str, rule: str, line: str) -> GuardViolationModel:
    text = line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"
    return GuardViolationModel(file=path, rule=rule, line=text)
