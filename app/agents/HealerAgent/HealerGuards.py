"""Edits that look like hiding a failure instead of fixing it, or that leave the project worse: found and reported.

The healer fixes technical problems (dependencies, compile errors, configuration). Removing assertions, deleting
or skipping tests, commenting checks out, or swallowing failures makes a red build green without fixing anything,
so every change an agent makes to a test file is checked here, and each one that does any of that is reported.

Small models also make clumsy edits: a build file that no longer parses (e.g. XML mangled by a sed command) or that
lost what it had, an import of a project class or module that does not exist, backup copies or saved command output
left behind. Those are reported too.

Nothing is ever rolled back (the user's rule): the project's own framework may need the agents' changes, so every
change stays, and the user reviews what is reported here.
"""

import fnmatch
import json
import os
import posixpath
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from app.agents.AnalyzerAgent.FactSheetCollector import MANIFEST_FILES
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
# A check commented out: assertion CODE in a comment (a call, or an assert statement with an expression), never prose
# that mentions one ("// Get the product names and assert first"). Gherkin steps count only in step files.
_COMMENTED_ASSERTION = re.compile(
    r"\b(assert\w*|Assert\.\w+|expect|verify)\s*\(|\.(should|to(Be|Equal|Have))\w*\s*\("
    r"|^\s*assert\s.*(\w\.\w|[(\[=<>!])"
)
_GHERKIN_CHECK = re.compile(r"^\s*(Then|And|But)\s")
_STEP_FILES = (".feature", ".robot")
_STRAY_FILE = re.compile(r"\.(bak|backup|orig|rej|swp)$|~$", re.IGNORECASE)
# Archives and compiled files: a jar the agent downloaded (a real run left an empty one in lib/ after a 404) is code
# nobody can review, and a push would take it along.
_BINARY_SUFFIXES = (".jar", ".war", ".ear", ".class", ".zip", ".gz", ".tgz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".whl")
# Output an agent saved while it worked, at the project root (`... | tee result.txt`): a log, or a text file named so.
_SCRATCH_FILE = re.compile(r"\.(log|out)$|(^|[^a-z])(result|output|out|log|tmp|temp|debug)s?\.txt$", re.IGNORECASE)
# Declarations that could make the XML parser expand entities: such files are not checked.
_XML_DECLARATIONS = ("<!DOCTYPE", "<!ENTITY")
_SWALLOW = re.compile(r"except\b[^:]*:\s*(pass)?\s*$|catch\s*\([^)]*\)\s*\{\s*\}")
# What a build or dependency file (MANIFEST_FILES) must keep (a real run: a `sed -i` cut a pom.xml from 590 lines to
# 258, its repositories and dependencies gone). A pom.xml: its entries, by tag -> (section, the children that name one);
# a package.json: these sections' packages; any other one: its lines, of which it may lose up to this many, or up to
# this share.
_POM_ENTRIES = {
    "dependency": ("dependencies", ("groupId", "artifactId")),
    "plugin": ("plugins", ("groupId", "artifactId")),
    "repository": ("repositories", ("id",)),
}
_PACKAGE_SECTIONS = ("dependencies", "devDependencies", "peerDependencies")
MAX_REMOVED_BUILD_LINES = 20
MAX_REMOVED_BUILD_SHARE = 0.2
# Imports, to find the ones that point to a project class or module that does not exist (a made-up helper).
_JVM_SOURCES = (".java", ".kt", ".groovy")
_JVM_PACKAGE = re.compile(r"^[\s\ufeff]*package\s+([\w.]+)", re.MULTILINE)  # a file may start with a BOM
# A declared type's name; the lookahead lets matches overlap, so Kotlin's `enum class Color` gives Color too.
_JVM_TYPE = re.compile(r"\b(?:class|interface|enum|record|object|typealias)\s+(?=(\w+))")
_JVM_IMPORT = re.compile(r"^import\s+(?:static\s+)?([\w.]*\w)(\.\*)?")
_JS_SOURCES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
# A relative module path: `from './x'`, `import './x'`, `require('./x')`, `import('../x')`.
_JS_RELATIVE_IMPORT = re.compile(r"""\b(?:from|import|require)\s*\(?\s*['"](\.\.?/[^'"]*)['"]""")
# How Node and TypeScript find a module: as written, with one of these endings, or as a folder's index file.
_JS_EXTENSIONS = (".ts", ".tsx", ".d.ts", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".json")
_PY_RELATIVE_IMPORT = re.compile(r"^from\s+(\.+)([\w.]*)\s+import\b")


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
        elif _COMMENT.match(line) and _is_assertion_code(path, _COMMENT.sub("", line, count=1)):
            violations.append(_violation(path, "assertion_commented", line))
        elif _SWALLOW.search(line) and _swallow_is_new(line, before):
            violations.append(_violation(path, "failure_swallowed", line))
    return violations


def check_changes(
    snapshot: ProjectSnapshot, changes: list[FileChangeModel], *, allow_deletes: bool = True
) -> list[GuardViolationModel]:
    """Every guard, over everything an agent changed since `snapshot`. A deleted file counts as every line removed;
    without `allow_deletes`, deleting a file is reported on its own."""
    violations: list[GuardViolationModel] = []
    edited: dict[str, tuple[str, str]] = {}
    for change in changes:
        before = snapshot.content(change.path) or ""
        after = "" if change.change == "deleted" else (snapshot.root / change.path).read_text(encoding="utf-8")
        edited[change.path] = (before, after)
        if change.change == "added":
            violations.extend(check_new_file(change.path, snapshot.files))
        elif change.change == "modified":
            violations.extend(check_build_file(change.path, before, after))
        elif not allow_deletes:
            violations.append(_violation(change.path, "file_deleted", change.path))
            continue
        violations.extend(check_test_integrity(change.path, before, after))
    violations.extend(check_imports(snapshot.root, snapshot.files, edited))
    # Binary and very large files are not among `changes` (nothing in them to read line by line): new ones still count.
    violations.extend(_violation(path, "binary_or_large_file", path) for path in snapshot.added_other_files())
    return violations


def check_new_file(path: str, project_files: Mapping[str, str] | None = None) -> list[GuardViolationModel]:
    """A file an agent added: backup copies and editor leftovers are not part of a fix, and neither is command output
    it saved at the project root. A file in a folder was put there on purpose, and a log that the project's files
    (before the agent's work) name is the project's own, e.g. the one its logging writes when the tests run."""
    if _STRAY_FILE.search(PurePosixPath(path).name):
        return [_violation(path, "stray_file", path)]
    if PurePosixPath(path).suffix.lower() in _BINARY_SUFFIXES:  # also an empty one: it reads as text
        return [_violation(path, "binary_or_large_file", path)]
    if "/" not in path and _SCRATCH_FILE.search(path) and not any(
        path in text for text in (project_files or {}).values()
    ):
        return [_violation(path, "scratch_file", path)]
    return []


def check_build_file(path: str, before: str, after: str) -> list[GuardViolationModel]:
    """An XML or JSON file (pom.xml, testng.xml, package.json, ...) that parsed before the change and no longer does,
    or a build or dependency file that lost what it had (see _removed_from_build_file). Additions and small edits are
    never reported."""
    suffix = PurePosixPath(path).suffix.lower()
    parses = _xml_parses if suffix in (".xml", ".pom") else _json_parses if suffix == ".json" else None
    if parses is not None and parses(before) and not parses(after):
        _, added = _diff_lines(before, after)
        return [_violation(path, "build_file_broken", added[0] if added else "(file emptied)")]
    removed = _removed_from_build_file(PurePosixPath(path).name.lower(), before, after)
    return [_violation(path, "build_file_reduced", removed)] if removed else []


def check_imports(
    root: Path, project_files: Mapping[str, str], edited: Mapping[str, tuple[str, str]]
) -> list[GuardViolationModel]:
    """Imports an agent added that point to a class or module of the project that does not exist: a helper it made up
    (a real run: three tests imported utilities.BrowserUtil, which the project never had). The build check finds these
    too, but it only runs on a project that built before the agent's work.

    A JVM import counts only in a package the project's own sources declare (or below one), a JS/TS or Python import
    only when it is relative: library imports are never reported. `root`: the project as the agent left it;
    `project_files`: its text files before the agent's work; `edited`: each file it changed, as (before, after) text.
    Only the import lines it added are checked."""
    now = {**project_files, **{path: after for path, (_, after) in edited.items()}}
    declarations: tuple[set[str], set[str]] | None = None  # the project's JVM packages and classes, read once needed
    violations: list[GuardViolationModel] = []
    for path, (before, after) in edited.items():
        folder = posixpath.dirname(path)
        for line in dict.fromkeys(_diff_lines(before, after)[1]):
            if _COMMENT.match(line):
                continue
            if path.endswith(_JVM_SOURCES) and (imported := _JVM_IMPORT.match(line)):
                declarations = declarations or _jvm_declarations(now)
                missing = _jvm_import_missing(imported.group(1), bool(imported.group(2)), *declarations)
            elif path.endswith(_JS_SOURCES):
                missing = any(_js_module_missing(root, folder, spec) for spec in _JS_RELATIVE_IMPORT.findall(line))
            elif path.endswith(".py") and (imported := _PY_RELATIVE_IMPORT.match(line)):
                missing = _py_module_missing(root, folder, imported.group(1), imported.group(2))
            else:
                continue
            if missing:
                violations.append(_violation(path, "import_not_found", line))
    return violations


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


def _removed_from_build_file(name: str, before: str, after: str) -> str | None:
    """What a build or dependency file (`name`, lower-case) lost, in a few words, or None. A pom.xml or package.json
    loses entries (a package moved to another section is kept); any other one, or one that does not parse, too many of
    its lines."""
    if not any(fnmatch.fnmatchcase(name, glob.lower()) for glob in MANIFEST_FILES):
        return None
    entries = _pom_entries if name == "pom.xml" else _package_entries if name == "package.json" else None
    old, new = (entries(before), entries(after)) if entries else (None, None)
    if old is None or new is None:
        return _lines_removed(before, after)
    kept = {entry for _, entry in new}
    removed = sorted((section, entry) for section, entry in old if entry not in kept)
    if not removed:
        return None
    sections = ", ".join(f"{section} ({count})" for section, count in Counter(s for s, _ in removed).items())
    return f"removed from {sections}: {', '.join(entry for _, entry in removed)}"


def _lines_removed(before: str, after: str) -> str | None:
    """'shrank from 590 to 258 lines' when the file lost more than MAX_REMOVED_BUILD_LINES of its lines (blank ones not
    counted) and more than MAX_REMOVED_BUILD_SHARE of them; None for a smaller loss."""
    old, new = (sum(1 for line in text.splitlines() if line.strip()) for text in (before, after))
    lost = old - new
    if lost > MAX_REMOVED_BUILD_LINES and lost > old * MAX_REMOVED_BUILD_SHARE:
        return f"shrank from {old} to {new} lines"
    return None


def _pom_entries(text: str) -> set[tuple[str, str]] | None:
    """A pom.xml's dependencies and plugins (groupId:artifactId) and repositories (id), as (section, name), wherever
    they are (profiles, dependencyManagement, ...). None when it does not parse or is not parsed (see _xml_parses)."""
    if any(declaration in text for declaration in _XML_DECLARATIONS):
        return None
    try:
        project = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    namespace = project.tag[: project.tag.find("}") + 1]  # "{http://maven.apache.org/POM/4.0.0}", or ""
    entries: set[tuple[str, str]] = set()
    for tag, (section, keys) in _POM_ENTRIES.items():
        for element in project.iter(namespace + tag):
            parts = [(element.findtext(namespace + key) or "").strip() for key in keys]
            if name := ":".join(part for part in parts if part):
                entries.add((section, name))
    return entries


def _package_entries(text: str) -> set[tuple[str, str]] | None:
    """A package.json's packages of every dependency section, as (section, name). None when it does not parse."""
    try:
        package = json.loads(text)
    except ValueError:
        return None
    if not isinstance(package, dict):
        return None
    return {
        (section, name)
        for section in _PACKAGE_SECTIONS
        if isinstance(package.get(section), dict)
        for name in package[section]
    }


def _jvm_declarations(files: Mapping[str, str]) -> tuple[set[str], set[str]]:
    """The packages the project's JVM sources declare, and their classes as package.Name: each file's own name and
    every type declared in it (a Kotlin file can hold several, and so can the agent's new files)."""
    packages: set[str] = set()
    classes: set[str] = set()
    for path, text in files.items():
        if not path.endswith(_JVM_SOURCES) or (declared := _JVM_PACKAGE.search(text)) is None:
            continue  # not JVM code, or in the default package (nothing can import from it)
        package = declared.group(1)
        packages.add(package)
        names = {PurePosixPath(path).stem, *(match.group(1) for match in _JVM_TYPE.finditer(text))}
        classes.update(f"{package}.{name}" for name in names)
    return packages, classes


def _jvm_import_missing(name: str, wildcard: bool, packages: set[str], classes: set[str]) -> bool:
    """`name` (an import without its `.*`) is in a package the project declares, or below one, and names no class or
    package the project has. Outside the project's packages it is a library's: never missing."""
    segments = name.split(".")
    inside = next((size for size in range(len(segments), 0, -1) if ".".join(segments[:size]) in packages), 0)
    if not inside:
        return False
    for size in range(inside + 1, len(segments) + 1):
        if segments[size - 1][:1].isupper():  # the class (a nested class or a static member may follow it)
            return ".".join(segments[:size]) not in classes
    # No class: a package's `.*`, or a Kotlin top-level function or property, whose package must exist.
    return (name if wildcard else ".".join(segments[:-1])) not in packages


def _js_module_missing(root: Path, folder: str, spec: str) -> bool:
    """A relative JS/TS module path (`spec`, in a file in `folder`) that leads to no file. One that leads out of the
    project is not checked."""
    target = posixpath.normpath(posixpath.join(folder, spec))
    if target == ".." or target.startswith("../"):
        return False
    stem, suffix = posixpath.splitext(target)
    bases = (target, stem) if suffix in (".js", ".jsx", ".mjs", ".cjs") else (target,)  # `./x.js` can name x.ts
    candidates = [
        target,
        *(base + ending for base in bases for ending in _JS_EXTENSIONS),
        *(f"{target}/index{ending}" for ending in _JS_EXTENSIONS),
    ]
    return not any(os.path.isfile(root / candidate) for candidate in candidates)


def _py_module_missing(root: Path, folder: str, dots: str, module: str) -> bool:
    """A relative Python import (`from ..pages.login import X`, in a file in `folder`) whose module is no file or
    package. One that leads out of the project is not checked."""
    target = posixpath.normpath(posixpath.join(folder, *[".."] * (len(dots) - 1), *filter(None, module.split("."))))
    if target == ".." or target.startswith("../"):
        return False
    return not (os.path.isfile(root / f"{target}.py") or os.path.isdir(root / target))


def _diff_lines(before: str, after: str) -> tuple[list[str], list[str]]:
    """Lines removed and added, as multisets of stripped lines: moving or re-indenting a line is not a change."""
    old = Counter(line.strip() for line in before.splitlines() if line.strip())
    new = Counter(line.strip() for line in after.splitlines() if line.strip())
    return list((old - new).elements()), list((new - old).elements())


def _is_assertion_code(path: str, text: str) -> bool:
    """`text` (a comment without its comment sign) is an assertion written as code, not prose about one."""
    return bool(_COMMENTED_ASSERTION.search(text)) or (path.endswith(_STEP_FILES) and bool(_GHERKIN_CHECK.match(text)))


def _swallow_is_new(line: str, before: str) -> bool:
    return line not in {existing.strip() for existing in before.splitlines()}


def _violation(path: str, rule: str, line: str) -> GuardViolationModel:
    text = line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"
    return GuardViolationModel(file=path, rule=rule, line=text)
