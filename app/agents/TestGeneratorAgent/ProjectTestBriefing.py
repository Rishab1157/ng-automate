"""A short, read-only briefing of the project's own tests and the shared code they use, inlined into the generator
prompt.

A small local model tends to invent helper methods (open/type/click) instead of reading the base class first,
and then calls methods that do not exist. Handing it the project's own base class, helpers and
locator/config files in the prompt makes it follow the real API from its first file. The briefing only reads
existing files; it never changes the project.

The shared code often lives outside the test folders (a root pages/ or fixtures/ folder, src/main/java, a resources/
folder), and every framework names those folders differently. So it is found from the project's own references, not
from folder names: one or two existing tests, the files they import or load and the files those import, the paths the
analyzer named as important, and the helper and config files at the project root.
"""

import os
import posixpath
import re
from collections import deque
from collections.abc import Sequence
from pathlib import Path

# Keep the briefing small: a handful of files, a few thousand characters (a small local model reads it on every
# request).
MAX_FILES = 10
MAX_TOTAL_CHARS = 10000
MAX_FILE_CHARS = 3000
SEPARATOR = "\n\n"
# Existing tests shown as examples, picked among the first candidates (one that uses other project files first).
EXAMPLE_TESTS = 2
EXAMPLE_CANDIDATES = 20
# How far references are followed from an example: the files it imports, then the files those import.
REFERENCE_DEPTH = 2
# Files shown from one folder the analyzer named as important.
FILES_PER_SHARED_FOLDER = 3
# The project files and folders looked at, and the characters read of one file (its references are near the top).
MAX_PROJECT_FILES = 20_000
MAX_PROJECT_FOLDERS = 50_000
MAX_READ_CHARS = 100_000
# Names like a base class, a helper, or where locators/config live: shown first, because the model needs the API.
PRIORITY_HINTS = (
    "base", "abstract", "helper", "common", "shared", "support", "util", "fixture", "conftest",
    "driver", "config", "locator", "selector", "page", "step", "hook", "world",
)
SOURCE_SUFFIXES = {
    ".java", ".kt", ".kts", ".groovy", ".scala", ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".rb", ".cs", ".go", ".rs", ".php", ".robot", ".resource", ".feature", ".properties", ".ini",
    ".cfg", ".conf", ".yaml", ".yml", ".toml", ".json",
}
# Data and settings, not code: never an example test, and what they name is not followed.
DATA_SUFFIXES = {".properties", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".toml", ".json"}
# Never the project's own code: dependencies, build output and reports. Hidden folders (.git, .tox, .auth) are
# skipped too: they hold tools' state, caches and saved logins.
SKIP_DIRS = {
    "node_modules", "target", "build", "out", "dist", "bin", "obj", "vendor", ".git", ".venv",
    "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache", ".gradle", ".idea", ".vscode",
    "site-packages", "bower_components", "allure-results", "ngauto-results",
}
# The file a reference to a folder loads: '../pages' -> pages/index.ts, a Python package -> its __init__.py.
INDEX_NAMES = ("index", "__init__")
# A test file by its name, in most languages: test_login.py, LoginTest.java, login.spec.ts, login_test.go.
TEST_FILE_NAME = re.compile(r"test|spec", re.IGNORECASE)
# A path written relative to the file, quoted or not: '../fixtures/test', "./pages", Resource    ../common.resource
RELATIVE_PATH = re.compile(r"(?<![\w.])\.\.?[\\/][^\s'\"`;,()<>{}\[\]|]+")
# A line that loads other code, and the rest of it: import a.b.C; | from a.b import c | using A.B; | use A\B; |
# require 'a/b' | require_once 'a/b.php' | } from '@pages/login'; | Resource    common.resource
IMPORT_LINE = re.compile(
    r"^[ \t}]*(import|from|export|using|use|require_relative|require_once|require|include_once|include|resource"
    r"|library|variables)\b(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
QUOTED_NAME = re.compile(r"""(['"`])([^'"`\s]+)\1""")
BARE_NAME = re.compile(r"[^\s'\"`;,()<>{}\[\]|*=]+")
# Between the parts of a module name: a.b.C, A::B, A\B.
MODULE_SEPARATOR = re.compile(r"\.|::|\\")
# 'src/a.py:12' or 'src/a.py:3-9' (a path citing a line) -> 'src/a.py'
LINE_SUFFIX = re.compile(r":\d+(-\d+)?$")
# The words of a file: the classes it names ('new CartPage(driver)') among the files of a package it imports.
WORD = re.compile(r"\w+")


def project_test_briefing(
    project_dir: Path, test_dirs: list[str], generated_dir: str, shared_paths: Sequence[str] = ()
) -> str:
    """The project's own tests and the shared code they use, as text ready to inline. Empty string when there is
    nothing. `shared_paths`: project-relative files or folders that hold shared code (the analyzer's important paths).
    """
    files = _ProjectFiles(Path(project_dir).resolve(), generated_dir)
    ordered, examples = _select_files(files, test_dirs, shared_paths)
    kept: dict[str, str] = {}
    used = 0
    # The first example test is weighed first: the shared code shown before it can never push it out.
    for path in [*examples[:1], *ordered]:
        if len(kept) >= MAX_FILES:
            break
        if path in kept:
            continue
        section = files.section(path)
        cost = len(section) + (len(SEPARATOR) if kept else 0)
        if section and used + cost <= MAX_TOTAL_CHARS:
            kept[path] = section
            used += cost
    return SEPARATOR.join(kept[path] for path in ordered if path in kept)


def _select_files(
    files: "_ProjectFiles", test_dirs: list[str], shared_paths: Sequence[str]
) -> tuple[list[str], list[str]]:
    """The files to show, in the order shown, and the example tests among them: the shared code an example test uses,
    then the example; then the shared paths, the helpers at the project root and the rest of the test folders."""
    tests = files.in_folders(test_dirs or [])
    examples = files.examples(tests)
    ordered: list[str] = []
    for example in examples:
        ordered += [*files.shared_code(example), example]
    ordered += files.in_paths(shared_paths)
    ordered += files.root_helpers()
    # Priority-named first, then smaller files (a base class is usually small and central).
    ordered += sorted(tests, key=files.rank)[:MAX_FILES]
    return list(dict.fromkeys(ordered)), examples


class _ProjectFiles:
    """The project's source and config files by project-relative path, and the project files each one references."""

    def __init__(self, root: Path, generated_dir: str) -> None:
        self.root = root
        self.sizes = _walk(root, generated_dir)
        # Lower-case path without its suffix -> files: 'pages/loginpage' -> ['pages/LoginPage.ts'].
        self.by_stem: dict[str, list[str]] = {}
        # Its last part -> (that path, file): to find a module by the end of its path.
        self.by_name: dict[str, list[tuple[str, str]]] = {}
        # A folder as a package name (lower-case, dots as slashes) -> the files directly inside it:
        # 'shop/tests/pages' -> Shop.Tests/Pages/*.cs.
        self.by_package: dict[str, list[str]] = {}
        for path in self.sizes:
            stem = _without_suffix(path).lower()
            self.by_stem.setdefault(stem, []).append(path)
            self.by_name.setdefault(stem.rpartition("/")[2], []).append((stem, path))
            self.by_package.setdefault(_package_key(posixpath.dirname(path)), []).append(path)
        # Its last part -> such folders: to find a package by the end of its path.
        self.packages_by_name: dict[str, list[str]] = {}
        for package in self.by_package:
            self.packages_by_name.setdefault(package.rpartition("/")[2], []).append(package)
        self._texts: dict[str, str] = {}
        self._uses: dict[str, list[str]] = {}

    def rank(self, path: str) -> tuple[int, int, str]:
        return _priority(_name(path)), self.sizes[path], path

    def text(self, path: str) -> str:
        """The start of a file (the briefing shows less of it). Empty when it cannot be read or is not text."""
        if path not in self._texts:
            try:
                # utf-8-sig: a byte order mark is not part of the first line ('using Shop.Pages;').
                with open(self.root / path, encoding="utf-8-sig", errors="replace") as file:
                    text = file.read(MAX_READ_CHARS)
            except OSError:
                text = ""
            # A NUL character: binary content behind a source suffix (a .ts video) or UTF-16 text, never shown.
            self._texts[path] = "" if "\x00" in text else text
        return self._texts[path]

    def section(self, path: str) -> str:
        """One file as the briefing shows it: its path, then its text. Empty for a blank file."""
        text = self.text(path)
        if not text.strip():
            return ""
        suffix = ""
        if len(text) > MAX_FILE_CHARS:
            text, suffix = text[:MAX_FILE_CHARS], "\n... (truncated)"
        return f"--- {path} ---\n{text}{suffix}"

    def in_folders(self, folders: Sequence[str]) -> dict[str, int]:
        """The files inside these folders -> how many of the folders hold each (a test is often inside two:
        src/test and src/test/java/com/acme/tests)."""
        prefixes = [f"{folder.lower()}/" for folder in map(_project_path, folders) if folder]
        counts = {path: sum(path.lower().startswith(prefix) for prefix in prefixes) for path in self.sizes}
        return {path: count for path, count in counts.items() if count}

    def examples(self, tests: dict[str, int]) -> list[str]:
        """One or two of the project's own tests: files of its test folders that hold text and are not data. Deeper
        in the test folders, named like a test, not named like a helper (nor inside a helper folder such as pages/ or
        support/), using other project files, then smaller ones first. A test named after a page (LoginPageTest) is
        still a test: it is only ranked after the others."""

        def order(path: str) -> tuple[int, bool, bool]:
            return -tests[path], TEST_FILE_NAME.search(_name(path)) is None, _is_helper(path)

        candidates = sorted(
            (path for path in tests if _suffix(path) not in DATA_SUFFIXES and self.sizes[path]),
            key=lambda path: (*order(path), self.sizes[path], path),
        )[:EXAMPLE_CANDIDATES]
        readable = [path for path in candidates if self.text(path).strip()]
        # Among equally test-like files, the ones that use shared code show the most.
        return sorted(readable, key=lambda path: (*order(path), not self.uses(path)))[:EXAMPLE_TESTS]

    def shared_code(self, example: str) -> list[str]:
        """The project files an example uses: the files it references, then the files those reference."""
        found: list[str] = []
        level = [example]
        for _ in range(REFERENCE_DEPTH):
            deeper: list[str] = []
            for referring in level:
                for path in self.uses(referring):
                    if path != example and path not in found:
                        found.append(path)
                        deeper.append(path)
            level = deeper[:MAX_FILES]
        return found

    def in_paths(self, paths: Sequence[str]) -> list[str]:
        """The files of these project-relative files and folders; of a folder, its first files (helpers by name, then
        smaller ones)."""
        found: list[str] = []
        for path in filter(None, map(_project_path, paths)):
            if path in self.sizes:
                found.append(path)
                continue
            prefix = f"{path.lower()}/"
            inside = (file for file in self.sizes if self.sizes[file] and file.lower().startswith(prefix))
            found += sorted(inside, key=self.rank)[:FILES_PER_SHARED_FOLDER]
        return found

    def root_helpers(self) -> list[str]:
        """The helper and config files at the project root (conftest.py, playwright.config.ts), smaller first."""
        return sorted((path for path in self.sizes if "/" not in path and _is_helper(path)), key=self.rank)

    def uses(self, path: str) -> list[str]:
        """The project files `path` references, in the order it names them. Data files reference nothing."""
        if path not in self._uses:
            found: list[str] = []
            if _suffix(path) not in DATA_SUFFIXES:
                text = self.text(path)
                folder = posixpath.dirname(path)
                words: set[str] | None = None
                for keys in _references(text):
                    matches = next(filter(None, (self.find(key, folder) for key in keys)), [])
                    if not matches:
                        # A package or a namespace ('import com.acme.pages.*;', 'using Shop.Pages;'): the files in it
                        # this file names ('new CartPage(driver)').
                        words = set(WORD.findall(text)) if words is None else words
                        package = self.package(keys[0], folder)
                        matches = [file for file in package if _without_suffix(_name(file)) in words]
                    # 'pages/login' may be login.py and login.json: the referring file's own kind is meant.
                    same_kind = [file for file in matches if _suffix(file) == _suffix(path)]
                    for file in same_kind or matches:
                        if file != path and file not in found:
                            found.append(file)
            self._uses[path] = found
        return self._uses[path]

    def find(self, key: str, folder: str) -> list[str]:
        """The files a reference names, for a file in `folder`: './x' and '../x' from that folder; another path or a
        module next to the file, at the project root, or (two parts or more) anywhere its path ends with it."""
        near = self.at(_project_path(posixpath.join(folder, key)))
        if near or key.startswith(("./", "../")):
            return near
        return self.at(_project_path(key)) or (self.ending(key) if "/" in key else [])

    def at(self, path: str | None) -> list[str]:
        """The file at `path`; else the files named like it with any source suffix ('fixtures/test' or
        'fixtures/test.js' -> fixtures/test.ts); else the index file of the folder it names."""
        if path is None:
            return []
        if path in self.sizes:
            return [path]
        stem = path.lower()
        found = self.by_stem.get(stem) or self.by_stem.get(_without_suffix(stem))
        for index in INDEX_NAMES:
            found = found or self.by_stem.get(f"{stem}/{index}")
        return list(found or [])

    def ending(self, key: str) -> list[str]:
        """The files whose path without its suffix ends with `key` ('com/acme/LoginPage' ->
        src/main/java/com/acme/LoginPage.java, 'pages/x.js' -> src/pages/x.ts); else the index file of such a
        folder."""
        key = _without_suffix(key.lower())
        found = self._ending(key)
        for index in INDEX_NAMES:
            found = found or self._ending(f"{key}/{index}")
        return found

    def _ending(self, key: str) -> list[str]:
        named = self.by_name.get(key.rpartition("/")[2], [])
        return [path for stem, path in named if stem == key or stem.endswith(f"/{key}")]

    def package(self, key: str, folder: str) -> list[str]:
        """The files directly inside the folder a reference names, found like `find` finds a file: next to the file,
        at the project root, or (two parts or more) anywhere its path ends with it. A dotted folder is one package
        name: 'Shop.Tests.Pages' -> Shop.Tests/Pages/."""
        places = [posixpath.join(folder, key)] if key.startswith(("./", "../")) else [posixpath.join(folder, key), key]
        for place in filter(None, map(_project_path, places)):
            if files := self.by_package.get(_package_key(place)):
                return files
        if "/" not in key or key.startswith(("./", "../")):
            return []
        wanted = _package_key(key)
        named = self.packages_by_name.get(wanted.rpartition("/")[2], [])
        return [file for package in named if package.endswith(f"/{wanted}") for file in self.by_package[package]]


def _walk(root: Path, generated_dir: str) -> dict[str, int]:
    """Project-relative path -> size of the project's source and config files, shallow ones first. Skipped and hidden
    folders, the generated tests' folder, links and names that are not text are left out."""
    sizes: dict[str, int] = {}
    folders = deque([""])
    for _ in range(MAX_PROJECT_FOLDERS):
        if not folders:
            break
        folder = folders.popleft()
        try:
            with os.scandir(root / folder) as scanned:
                entries = sorted(scanned, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            path = f"{folder}/{entry.name}" if folder else entry.name
            try:
                if entry.is_symlink() or entry.is_junction() or not _is_text(entry.name):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIP_DIRS and entry.name != generated_dir and not entry.name.startswith("."):
                        folders.append(path)
                elif entry.is_file(follow_symlinks=False) and _suffix(entry.name) in SOURCE_SUFFIXES:
                    sizes[path] = entry.stat(follow_symlinks=False).st_size
                    if len(sizes) >= MAX_PROJECT_FILES:
                        return sizes
            except OSError:
                continue
    return sizes


def _references(text: str) -> list[tuple[str, ...]]:
    """The other files a file names, in the order it names them, each as the keys to try in turn: a path relative to
    the file ('../fixtures/test'), or a module or path on an import line ('com.acme.pages.LoginPage', '@pages/x')."""
    found = [(match.start(), (_path_key(match.group().replace("\\", "/")),)) for match in RELATIVE_PATH.finditer(text)]
    for line in IMPORT_LINE.finditer(text):
        found += [(line.start(), keys) for keys in _named_on_line(line.group(1).lower(), line.group(2))]
    found.sort(key=lambda item: item[0])
    return list(dict.fromkeys(keys for _, keys in found if keys[0]))


def _named_on_line(keyword: str, rest: str) -> list[tuple[str, ...]]:
    """What one import line names, each as the keys to try in turn."""
    quoted = [name for _, name in QUOTED_NAME.findall(rest)]
    if quoted:  # import { a } from '../a' | require 'pages/login': the quoted names are the modules
        return [_keys(name) for name in quoted]
    names = BARE_NAME.findall(rest)
    if keyword == "from" and "import" in names[1:]:  # from a.b import c, d: c and d are inside a.b
        module, inside = names[0], names[names.index("import") + 1:]
        joined = [module + name if module.endswith(".") else f"{module}.{name}" for name in inside]
        return [(_module_key(name), _module_key(module)) for name in joined] or [_keys(module)]
    # A lone word is a module only when the line names nothing else (import utils), not in prose (use the menu).
    return [_keys(name) for name in names if len(names) == 1 or any(mark in name for mark in "./\\:")]


def _keys(name: str) -> tuple[str, ...]:
    """The keys one written name stands for, the most exact first: a path; or a module, then the module around it
    ('a.b.C.method' -> 'a/b/C/method', then 'a/b/C')."""
    if "/" in name or name.startswith((".\\", "..\\")):
        return (_path_key(name.replace("\\", "/")),)
    key = _module_key(name)
    around = key.rpartition("/")[0]
    return (key, around) if around.strip("./") else (key,)


def _module_key(name: str) -> str:
    """'a.b.C', 'A::B' or 'A\\B' -> 'a/b/C'. Leading dots are the file's own folder: '.x' -> './x', '..x' -> '../x'."""
    dots = len(name) - len(name.lstrip("."))
    path = "/".join(part for part in MODULE_SEPARATOR.split(name[dots:]) if part)
    if not dots:
        return path
    return ("./" if dots == 1 else "../" * (dots - 1)) + path


def _path_key(path: str) -> str:
    """A written path as a key: './x' and '../x' as written; others without a scheme ('package:'), alias marks
    ('@pages/x', '~/x') or variables ('${CURDIR}/x')."""
    if path.startswith(("./", "../")):
        return path
    parts = (part.lstrip("@~") for part in path.rpartition(":")[2].split("/"))
    return "/".join(part for part in parts if part and not any(mark in part for mark in "${}%"))


def _project_path(path: str) -> str | None:
    """A project-relative path in its plain form ('./pages/' -> 'pages', 'src\\a.py:12' -> 'src/a.py'). None for the
    project itself and for paths outside it."""
    plain = LINE_SUFFIX.sub("", path.strip().replace("\\", "/"))
    if not plain or plain.startswith("/") or ":" in plain:
        return None
    normal = posixpath.normpath(plain)
    return None if normal in (".", "..") or normal.startswith("../") else normal


def _package_key(folder: str) -> str:
    """A folder as a package name: lower-case, its dots as slashes ('Shop.Tests/Pages' -> 'shop/tests/pages')."""
    return folder.lower().replace(".", "/")


def _without_suffix(path: str) -> str:
    stem, suffix = posixpath.splitext(path)
    return stem if suffix.lower() in SOURCE_SUFFIXES else path


def _suffix(path: str) -> str:
    return posixpath.splitext(path)[1].lower()


def _name(path: str) -> str:
    return path.rpartition("/")[2]


def _is_text(name: str) -> bool:
    """False for a name that is not valid text (raw bytes on Linux, a lone surrogate on Windows): a prompt cannot
    carry it."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_helper(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in PRIORITY_HINTS)


def _priority(name: str) -> int:
    return 0 if _is_helper(name) else 1
