"""Facts about a project found by plain code: languages, build tool, test frameworks and tools.

No LLM and no network, and the same answer every time for the same files: the analyzer agent
receives this sheet as given truth. The tree is walked without following links, and only known
manifest files are read, each capped. pom.xml is searched as text, never parsed as XML (XML bombs),
and every pattern runs in linear time, so a hostile file cannot stall the scan.
"""

import codecs
import fnmatch
import json
import logging
import os
import re
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource
from app.utils.ArchiveUtils import SKIPPED_DIRS

logger = logging.getLogger(__name__)

# Work limits: a huge or hostile tree costs at most this much.
MAX_SCANNED_FILES = 200_000
MAX_SCANNED_FOLDERS = 50_000
MAX_MANIFESTS_READ = 200
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_TOTAL_BYTES = 32 * 1024 * 1024

# Output limits.
MAX_MARKER_FILES = 50
MAX_TEST_DIRS = 20
MAX_TREE_ENTRIES = 200
TREE_DEPTH = 2
LANGUAGE_EXAMPLES = 3
MAX_EVIDENCE = 5
# The primary language is "high" confidence above this share of the programming files.
PRIMARY_LANGUAGE_HIGH_SHARE = 0.6

# Cache and dependency folders, on top of the ones project archives leave out. A committed
# virtualenv under another name (env/) still holds its packages in site-packages/.
EXTRA_SKIPPED_DIRS = frozenset({
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".vs", "bower_components", "site-packages",
})

GHERKIN = "Gherkin"
VBSCRIPT = "VBScript"
ROBOT_FRAMEWORK = "Robot Framework"
UFT = "UFT"

# File extension (lower-case) -> language.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".java": "Java",
    ".kt": "Kotlin",
    ".groovy": "Groovy",
    ".scala": "Scala",
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",  # except UFT's Action<N>/Script.mts, see UFT_ACTION_DIR
    ".cts": "TypeScript",
    ".cs": "C#",
    ".rb": "Ruby",
    ".go": "Go",
    ".php": "PHP",
    ".swift": "Swift",
    ".vbs": VBSCRIPT,
    ".qfl": VBSCRIPT,
    ".feature": GHERKIN,
    ".robot": ROBOT_FRAMEWORK,
}
# Counted in language_files, never chosen as the primary language.
NON_PROGRAMMING_LANGUAGES = frozenset({GHERKIN})

# UFT/QTP tests (.tsp), function libraries (.qfl) and shared object repositories (.tsr).
# .usr is left out on purpose: LoadRunner scripts have it too.
UFT_MARKER_EXTENSIONS = frozenset({".tsp", ".qfl", ".tsr"})
# Mobile app builds a mobile test project installs on the device: markers, not proof of a tool.
MOBILE_APP_EXTENSIONS = frozenset({".apk", ".aab", ".ipa"})
# UFT keeps each action's VBScript in Action<N>/Script.mts; any other .mts file is TypeScript.
UFT_ACTION_DIR = re.compile(r"action\d+", re.IGNORECASE)

# Every name the rules below can find belongs to at least one of these facts.
TEST_FRAMEWORKS = frozenset({
    "pytest-playwright",
    "TestNG", "JUnit 4", "JUnit 5", "Spock", "pytest", "unittest", ROBOT_FRAMEWORK, "Jest", "Mocha", "Jasmine",
    "Vitest", "Playwright Test", "Cypress", "WebdriverIO", "TestCafe", "Nightwatch", "NUnit", "xUnit", "MSTest",
    "RSpec", "Minitest", "Ginkgo",
})
AUTOMATION_TOOLS = frozenset({
    "Selenium", "Selenide", "Playwright", "Appium", "Cypress", "WebdriverIO", "REST Assured", UFT, "Puppeteer",
    "Protractor", "TestCafe", "Nightwatch", "Watir", "Capybara",
})
BDD_TOOLS = frozenset({"Cucumber", "JBehave", "Behave", "pytest-bdd", "SpecFlow", "Reqnroll", "Godog", "playwright-bdd"})

# Manifest (file-name glob, any case) -> build tool. Only the shallowest manifests decide.
BUILD_TOOL_FILES: dict[str, str] = {
    "pom.xml": "Maven",
    "build.gradle": "Gradle",
    "build.gradle.kts": "Gradle",
    "settings.gradle": "Gradle",
    "settings.gradle.kts": "Gradle",
    "package.json": "npm",
    "pyproject.toml": "pip",
    "requirements*.txt": "pip",
    "setup.py": "pip",
    "Pipfile": "Pipenv",
    "*.csproj": "dotnet",
    "*.vbproj": "dotnet",
    "*.fsproj": "dotnet",
    "*.sln": "dotnet",
    "Gemfile": "Bundler",
    "go.mod": "Go modules",
}
# Manifest -> lock files that name the tool actually used when they sit next to it (first match wins).
LOCK_FILES: dict[str, dict[str, str]] = {
    "package.json": {
        "pnpm-lock.yaml": "pnpm",
        "yarn.lock": "Yarn",
        "bun.lock": "Bun",
        "bun.lockb": "Bun",
        "package-lock.json": "npm",
    },
    "pyproject.toml": {"poetry.lock": "Poetry", "uv.lock": "uv"},
}
# package.json "packageManager": "<name>@<version>" -> build tool. Wins over lock files.
NODE_PACKAGE_MANAGERS: dict[str, str] = {"npm": "npm", "yarn": "Yarn", "pnpm": "pnpm", "bun": "Bun"}
# Generic tool -> the specific tools of its ecosystem that replace it at the same level
# (a Poetry project that also exports requirements.txt is a Poetry project).
GENERIC_BUILD_TOOLS: dict[str, frozenset[str]] = {
    "pip": frozenset({"Poetry", "Pipenv", "uv"}),
    "npm": frozenset({"Yarn", "pnpm", "Bun"}),
}

# Config files (glob, any case) that exist only when a framework or tool is used.
FRAMEWORK_FILES: dict[str, tuple[str, ...]] = {
    "testng*.xml": ("TestNG",),
    "junit-platform.properties": ("JUnit 5",),
    "conftest.py": ("pytest",),
    "pytest.ini": ("pytest",),
    "playwright.config.*": ("Playwright Test", "Playwright"),
    "cypress.config.*": ("Cypress",),
    "cypress.json": ("Cypress",),
    "wdio.conf.*": ("WebdriverIO",),
    "wdio.*.conf.*": ("WebdriverIO",),
    # Appium capability files (desired capabilities for a device and app).
    "capabilities*.json": ("Appium",),
    "*.capabilities.json": ("Appium",),
    "*.caps.json": ("Appium",),
    "caps.json": ("Appium",),
    "desired_caps*.json": ("Appium",),
    "desiredcapabilities*.json": ("Appium",),
    "jest.config.*": ("Jest",),
    ".mocharc*": ("Mocha",),
    "vitest.config.*": ("Vitest",),
    "nightwatch.conf.*": ("Nightwatch",),
    "protractor.conf.*": ("Protractor",),
    ".testcaferc*": ("TestCafe",),
    "behave.ini": ("Behave",),
    ".behaverc": ("Behave",),
    "cucumber.properties": ("Cucumber",),
    "specflow.json": ("SpecFlow",),
    "reqnroll.json": ("Reqnroll",),
}

# Files whose text is searched for dependencies (glob, any case) -> ecosystem.
MANIFEST_FILES: dict[str, str] = {
    "pom.xml": "maven",
    "*.gradle": "gradle",
    "*.gradle.kts": "gradle",
    "*.versions.toml": "gradle",  # Gradle version catalog
    "package.json": "npm",
    "pyproject.toml": "python",
    "requirements*.txt": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "tox.ini": "python",
    "Pipfile": "python",
    "*.csproj": "nuget",
    "*.vbproj": "nuget",
    "*.fsproj": "nuget",
    "packages.config": "nuget",
    "Directory.Packages.props": "nuget",
    "Gemfile": "ruby",
    "go.mod": "go",
}

# Maven/Gradle groupId or artifactId (lower-case) -> names.
JAVA_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "org.testng": ("TestNG",),
    "testng": ("TestNG",),
    "junit": ("JUnit 4",),  # junit:junit
    "org.junit.vintage": ("JUnit 4",),
    "junit-vintage-engine": ("JUnit 4",),
    "org.junit.jupiter": ("JUnit 5",),
    "org.junit.platform": ("JUnit 5",),
    "junit-jupiter": ("JUnit 5",),
    "junit-jupiter-api": ("JUnit 5",),
    "junit-jupiter-engine": ("JUnit 5",),
    "junit-bom": ("JUnit 5",),
    "org.spockframework": ("Spock",),
    "org.seleniumhq.selenium": ("Selenium",),
    "selenium-java": ("Selenium",),
    "selenide": ("Selenide", "Selenium"),
    "com.microsoft.playwright": ("Playwright",),
    "io.appium": ("Appium",),
    "io.rest-assured": ("REST Assured",),
    "rest-assured": ("REST Assured",),
    "com.jayway.restassured": ("REST Assured",),
    "io.cucumber": ("Cucumber",),
    "info.cukes": ("Cucumber",),
    "org.jbehave": ("JBehave",),
}
# Gradle test task settings -> names. useJUnitPlatform() is left out: JUnit 5, Spock 2 and Cucumber all use it.
GRADLE_TEST_SETTINGS: dict[str, tuple[str, ...]] = {"useTestNG": ("TestNG",), "useJUnit": ("JUnit 4",)}

NPM_DEPENDENCY_SECTIONS = ("dependencies", "devDependencies")
# npm package (lower-case) -> names.
NPM_PACKAGES: dict[str, tuple[str, ...]] = {
    "jest": ("Jest",),
    "@jest/globals": ("Jest",),
    "ts-jest": ("Jest",),
    "mocha": ("Mocha",),
    "jasmine": ("Jasmine",),
    "jasmine-core": ("Jasmine",),
    "vitest": ("Vitest",),
    "@playwright/test": ("Playwright Test", "Playwright"),
    "playwright": ("Playwright",),
    "playwright-core": ("Playwright",),
    "cypress": ("Cypress",),
    "webdriverio": ("WebdriverIO",),
    "@wdio/mocha-framework": ("Mocha",),
    "@wdio/jasmine-framework": ("Jasmine",),
    "@wdio/cucumber-framework": ("Cucumber",),
    "@wdio/appium-service": ("Appium",),
    "appium": ("Appium",),
    "selenium-webdriver": ("Selenium",),
    "puppeteer": ("Puppeteer",),
    "protractor": ("Protractor",),
    "testcafe": ("TestCafe",),
    "nightwatch": ("Nightwatch",),
    "@cucumber/cucumber": ("Cucumber",),
    "cucumber": ("Cucumber",),
    "@badeball/cypress-cucumber-preprocessor": ("Cucumber",),
    "cypress-cucumber-preprocessor": ("Cucumber",),
    "playwright-bdd": ("playwright-bdd",),
}
NPM_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {"@wdio/": ("WebdriverIO",)}

# Python package (normalized: lower-case, "-" separators) -> names.
PYTHON_PACKAGES: dict[str, tuple[str, ...]] = {
    "pytest": ("pytest",),
    "unittest-xml-reporting": ("unittest",),
    "xmlrunner": ("unittest",),
    "robotframework": (ROBOT_FRAMEWORK,),
    "robotframework-seleniumlibrary": ("Selenium",),
    "robotframework-selenium2library": ("Selenium",),
    "robotframework-browser": ("Playwright",),
    "robotframework-appiumlibrary": ("Appium",),
    "selenium": ("Selenium",),
    "seleniumbase": ("Selenium",),
    "pytest-selenium": ("Selenium",),
    "playwright": ("Playwright",),
    "pytest-playwright": ("Playwright", "pytest-playwright"),
    "appium-python-client": ("Appium",),
    "behave": ("Behave",),
    "pytest-bdd": ("pytest-bdd",),
}
# Plugins and libraries that only exist for one framework.
PYTHON_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "pytest-": ("pytest",),
    "robotframework-": (ROBOT_FRAMEWORK,),
    "behave-": ("Behave",),
}

# NuGet package id prefix (lower-case) -> names.
NUGET_PACKAGE_PREFIXES: dict[str, tuple[str, ...]] = {
    "nunit": ("NUnit",),
    "xunit": ("xUnit",),
    "mstest": ("MSTest",),
    "selenium.": ("Selenium",),
    "microsoft.playwright": ("Playwright",),
    "appium.webdriver": ("Appium",),
    "specflow": ("SpecFlow",),
    "reqnroll": ("Reqnroll",),
}

# Ruby gem (lower-case) -> names.
RUBY_GEMS: dict[str, tuple[str, ...]] = {
    "rspec": ("RSpec",),
    "rspec-core": ("RSpec",),
    "minitest": ("Minitest",),
    "cucumber": ("Cucumber",),
    "selenium-webdriver": ("Selenium",),
    "watir": ("Watir",),
    "capybara": ("Capybara",),
    "appium_lib": ("Appium",),
}

# Go module path prefix -> names.
GO_MODULES: dict[str, tuple[str, ...]] = {
    "github.com/onsi/ginkgo": ("Ginkgo",),
    "github.com/cucumber/godog": ("Godog",),
    "github.com/playwright-community/playwright-go": ("Playwright",),
    "github.com/tebeka/selenium": ("Selenium",),
}

# Folders that hold tests (when something is inside them).
TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "e2e", "spec", "specs", "features", "cypress", "playwright"})
# .NET test projects such as Shop.Tests/ or Shop.UnitTests/.
TEST_PROJECT_DIR = re.compile(
    r".+\.(unit|integration|ui|api|e2e|acceptance|functional|automation)?(tests?|specs?)", re.IGNORECASE
)

_SKIPPED_DIRS = SKIPPED_DIRS | EXTRA_SKIPPED_DIRS
_LOCK_FILE_NAMES = frozenset(name for locks in LOCK_FILES.values() for name in locks)
_CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}

# Removed before searching a pom.xml: commented-out, excluded and only version-managed dependencies.
_MAVEN_IGNORED_BLOCKS = (
    ("<!--", "-->"),
    ("<exclusions>", "</exclusions>"),
    ("<dependencyManagement>", "</dependencyManagement>"),
)
_MAVEN_ID = re.compile(r"<(?:groupId|artifactId)>\s*([^<\s]+)\s*</(?:groupId|artifactId)>")
_GRADLE_LINE_COMMENT = re.compile(r"^[ \t]*//[^\n]*", re.MULTILINE)
_GRADLE_COORDINATE = re.compile(r"""["']([\w.\-]+):([\w.\-]+)""")
_GRADLE_MAP_NOTATION = re.compile(r"""\bgroup\s*[:=]\s*["']([\w.\-]+)["']\s*,\s*name\s*[:=]\s*["']([\w.\-]+)["']""")
_GRADLE_TEST_SETTING = re.compile(r"\b(" + "|".join(GRADLE_TEST_SETTINGS) + r")\b")
# Requirement lines and TOML keys ("pytest = ..."), quoted requirements ("pytest>=8"), and VCS URLs (#egg=name).
# [ \t]* rather than \s* after ^: \s also eats newlines, which would make many blank lines quadratic.
_PY_LINE_NAME = re.compile(r"^[ \t]*([A-Za-z0-9][\w.\-]*)", re.MULTILINE)
_PY_QUOTED_NAME = re.compile(r"""["']([A-Za-z0-9][\w.\-]*)""")
_PY_EGG_NAME = re.compile(r"[#&]egg=([A-Za-z0-9][\w.\-]*)")
_PY_NAME_SEPARATORS = re.compile(r"[-_.]+")
_PYTEST_SECTION = re.compile(r"^[ \t]*\[(?:tool[.:])?pytest\b", re.MULTILINE)
_POETRY_SECTION = re.compile(r"^[ \t]*\[tool\.poetry\b", re.MULTILINE)
_UV_SECTION = re.compile(r"^[ \t]*\[tool\.uv\b", re.MULTILINE)
# [^<>]*? stays inside one tag, so the search is linear even on hostile input.
_NUGET_PACKAGE = re.compile(
    r"""<(?:PackageReference|PackageVersion|package)\b[^<>]*?\b(?:Include|id)\s*=\s*["']([^"'<>]+)["']""",
    re.IGNORECASE,
)
_RUBY_GEM = re.compile(r"""^[ \t]*gem[ \t]+["']([^"'\n]+)["']""", re.MULTILINE)


class _GlobRules[T]:
    """File-name globs (any case) checked with one regex per table; the first matching glob wins."""

    def __init__(self, rules: dict[str, T]) -> None:
        self._values = tuple(rules.values())
        alternatives = (f"(?P<g{index}>{fnmatch.translate(glob)})" for index, glob in enumerate(rules))
        self._pattern = re.compile("|".join(alternatives), re.IGNORECASE)

    def get(self, name: str) -> T | None:
        match = self._pattern.match(name)
        if match is None or match.lastgroup is None:
            return None
        return self._values[int(match.lastgroup[1:])]


_BUILD_TOOL_RULES = _GlobRules(BUILD_TOOL_FILES)
_FRAMEWORK_FILE_RULES = _GlobRules(FRAMEWORK_FILES)
_MANIFEST_RULES = _GlobRules(MANIFEST_FILES)


@dataclass(frozen=True)
class _Hit:
    """A framework or tool name found in one file."""

    name: str
    path: str
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True)
class _Found:
    """What one manifest says: names, plus the build tool when the file itself names it."""

    names: frozenset[str] = frozenset()
    build_tool: str | None = None


@dataclass(frozen=True)
class _Folder:
    path: str  # repo-relative, "" for the project root
    depth: int  # 0 for the project root
    files: list[str]
    dirs: list[str]


@dataclass
class _Scan:
    total_files: int = 0
    language_files: Counter[str] = field(default_factory=Counter)
    language_examples: dict[str, list[str]] = field(default_factory=dict)
    hits: list[_Hit] = field(default_factory=list)
    marker_files: set[str] = field(default_factory=set)
    build_files: list[tuple[str, str]] = field(default_factory=list)  # (path, build tool)
    lock_files: dict[tuple[str, str], str] = field(default_factory=dict)  # (folder, lower-case name) -> path
    manifests: list[tuple[str, str]] = field(default_factory=list)  # (path, ecosystem)
    test_dirs: dict[str, bool] = field(default_factory=dict)  # candidate folder -> has files inside
    tree: list[tuple[int, str]] = field(default_factory=list)  # (depth, entry)


def collect_fact_sheet(project_dir: Path) -> FactSheetModel:
    """Facts about the project in `project_dir`. Blocking file IO: run it in a worker thread."""
    root = Path(project_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder not found: {root}")

    scan = _scan(root)
    manifest_hits, named_build_tools = _read_manifests(root, scan.manifests)
    hits = [*scan.hits, *manifest_hits]
    examples = scan.language_examples
    if scan.language_files[ROBOT_FRAMEWORK]:
        hits += [_Hit(ROBOT_FRAMEWORK, path) for path in examples[ROBOT_FRAMEWORK]]
    primary_language = _primary_language(scan)
    if primary_language.value == VBSCRIPT and not any(hit.name == UFT for hit in hits):
        # Mostly VBScript but none of UFT's own files: very likely UFT, not proven.
        hits += [_Hit(UFT, path, Confidence.MEDIUM) for path in examples[VBSCRIPT]]

    return FactSheetModel(
        total_files=scan.total_files,
        language_files=dict(sorted(scan.language_files.items(), key=lambda item: (-item[1], item[0]))),
        primary_language=primary_language,
        build_tool=_build_tool(scan, named_build_tools),
        test_frameworks=_names_fact(hits, TEST_FRAMEWORKS, single=False),
        automation_tools=_names_fact(hits, AUTOMATION_TOOLS, single=False),
        bdd_tool=_names_fact(hits, BDD_TOOLS, single=True),
        marker_files=sorted(scan.marker_files, key=_shallow_first)[:MAX_MARKER_FILES],
        test_dirs=sorted((path for path, used in scan.test_dirs.items() if used), key=_shallow_first)[:MAX_TEST_DIRS],
        feature_file_count=scan.language_files[GHERKIN],
        # All first-level entries before any second-level one, then shown in path order.
        top_level_tree=sorted(entry for _, entry in sorted(scan.tree)[:MAX_TREE_ENTRIES]),
    )


def _scan(root: Path) -> _Scan:
    scan = _Scan()
    # Folder -> the test-folder candidates it is in (itself included).
    inside_tests: dict[str, tuple[str, ...]] = {"": ()}
    for folder in _walk(root):
        test_parents = inside_tests.pop(folder.path, ())
        for name in folder.dirs:
            path = _join(folder.path, name)
            if _is_test_dir(name):
                scan.test_dirs.setdefault(path, False)
                inside_tests[path] = (*test_parents, path)
            else:
                inside_tests[path] = test_parents
            if folder.depth < TREE_DEPTH:
                scan.tree.append((folder.depth + 1, f"{path}/"))
        if folder.files:
            for path in test_parents:
                scan.test_dirs[path] = True
        in_uft_action = UFT_ACTION_DIR.fullmatch(_name_of(folder.path)) is not None
        for name in folder.files:
            path = _join(folder.path, name)
            _add_file(scan, path, name.lower(), in_uft_action)
            if folder.depth < TREE_DEPTH:
                scan.tree.append((folder.depth + 1, path))
    return scan


def _add_file(scan: _Scan, path: str, name: str, in_uft_action: bool) -> None:
    """`name` is the lower-case file name."""
    scan.total_files += 1
    extension = os.path.splitext(name)[1]
    language = LANGUAGE_BY_EXTENSION.get(extension)
    if extension == ".mts" and in_uft_action:
        language = VBSCRIPT
        scan.hits.append(_Hit(UFT, path))
    if language:
        scan.language_files[language] += 1
        examples = scan.language_examples.setdefault(language, [])
        if len(examples) < LANGUAGE_EXAMPLES:
            examples.append(path)
    if extension in UFT_MARKER_EXTENSIONS:
        scan.hits.append(_Hit(UFT, path))
        scan.marker_files.add(path)
    if extension in MOBILE_APP_EXTENSIONS:
        scan.marker_files.add(path)

    if build_tool := _BUILD_TOOL_RULES.get(name):
        scan.build_files.append((path, build_tool))
        scan.marker_files.add(path)
    if name in _LOCK_FILE_NAMES:
        scan.lock_files[(_parent_of(path), name)] = path
        scan.marker_files.add(path)
    if names := _FRAMEWORK_FILE_RULES.get(name):
        scan.hits.extend(_Hit(found, path) for found in names)
        scan.marker_files.add(path)
    if ecosystem := _MANIFEST_RULES.get(name):
        scan.manifests.append((path, ecosystem))


def _walk(root: Path) -> Iterator[_Folder]:
    """Breadth-first and sorted by name: the same tree always gives the same facts, shallow files first."""
    queue: deque[tuple[str, int]] = deque([("", 0)])
    files_left, folders_left = MAX_SCANNED_FILES, MAX_SCANNED_FOLDERS
    truncated = False
    while queue and files_left > 0:
        path, depth = queue.popleft()
        files, dirs = _list_folder(root / path)
        truncated = truncated or len(files) > files_left or len(dirs) > folders_left
        files, dirs = files[:files_left], dirs[:folders_left]
        files_left -= len(files)
        folders_left -= len(dirs)
        queue.extend((_join(path, name), depth + 1) for name in dirs)
        yield _Folder(path, depth, files, dirs)
    if truncated or queue:
        logger.warning(
            "Project scan stopped at %d files / %d folders: facts may be incomplete",
            MAX_SCANNED_FILES, MAX_SCANNED_FOLDERS,
        )


def _list_folder(folder: Path) -> tuple[list[str], list[str]]:
    """Regular files and the sub-folders to walk, sorted. Links, junctions and skipped folders are left out."""
    try:
        with os.scandir(folder) as scanned:
            entries = sorted(islice(scanned, MAX_SCANNED_FILES + MAX_SCANNED_FOLDERS), key=lambda entry: entry.name)
    except OSError:
        logger.debug("Skipped unreadable folder %s", folder)
        return [], []

    files: list[str] = []
    dirs: list[str] = []
    for entry in entries:
        try:
            if entry.is_symlink() or entry.is_junction() or not _is_utf8(entry.name):
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _SKIPPED_DIRS:
                    dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False):  # never FIFOs or devices: reading one could hang
                files.append(entry.name)
        except OSError:
            continue
    return files, dirs


def _is_utf8(name: str) -> bool:
    """Undecodable names (raw bytes on Linux) cannot be stored in MongoDB, so they are left out."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_test_dir(name: str) -> bool:
    return name.lower() in TEST_DIR_NAMES or TEST_PROJECT_DIR.fullmatch(name) is not None


def _read_manifests(root: Path, manifests: list[tuple[str, str]]) -> tuple[list[_Hit], dict[str, str]]:
    """Names found in manifests (shallowest first, capped), and the build tool some manifests name (by path)."""
    hits: list[_Hit] = []
    named_build_tools: dict[str, str] = {}
    budget = MAX_MANIFEST_TOTAL_BYTES
    ordered = sorted(manifests, key=lambda manifest: _shallow_first(manifest[0]))
    if len(ordered) > MAX_MANIFESTS_READ:
        logger.info("Reading only the %d shallowest of %d manifest files", MAX_MANIFESTS_READ, len(ordered))
    for path, ecosystem in ordered[:MAX_MANIFESTS_READ]:
        if budget <= 0:
            logger.info("Manifest read budget used up; remaining manifest files skipped")
            break
        data = _read_head(root / path, min(MAX_MANIFEST_BYTES, budget))
        if data is None:
            continue
        budget -= len(data)
        found = _MANIFEST_READERS[ecosystem](_decode(data))
        hits.extend(_Hit(name, path) for name in sorted(found.names))
        if found.build_tool:
            named_build_tools[path] = found.build_tool
    return hits, named_build_tools


def _read_head(path: Path, limit: int) -> bytes | None:
    try:
        with path.open("rb") as file:
            return file.read(limit)
    except OSError:
        logger.debug("Skipped unreadable manifest %s", path)
        return None


def _decode(data: bytes) -> str:
    # Windows tools sometimes save project files as UTF-16 or with a UTF-8 BOM (which json.loads rejects).
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _read_maven(text: str) -> _Found:
    for start, end in _MAVEN_IGNORED_BLOCKS:
        text = _drop_blocks(text, start, end)
    ids = {value.lower() for value in _MAVEN_ID.findall(text)}
    return _Found(frozenset(_lookup(ids, JAVA_DEPENDENCIES)))


def _read_gradle(text: str) -> _Found:
    text = _GRADLE_LINE_COMMENT.sub("", text)
    ids: set[str] = set()
    for group, artifact in [*_GRADLE_COORDINATE.findall(text), *_GRADLE_MAP_NOTATION.findall(text)]:
        ids.update((group.lower(), artifact.lower()))
    names = _lookup(ids, JAVA_DEPENDENCIES)
    for setting in _GRADLE_TEST_SETTING.findall(text):
        names.update(GRADLE_TEST_SETTINGS[setting])
    return _Found(frozenset(names))


def _read_npm(text: str) -> _Found:
    try:
        manifest = json.loads(text)
    except (ValueError, RecursionError):
        return _Found()
    if not isinstance(manifest, dict):
        return _Found()
    packages: set[str] = set()
    for section in NPM_DEPENDENCY_SECTIONS:
        dependencies = manifest.get(section)
        if isinstance(dependencies, dict):
            packages.update(name.lower() for name in dependencies)
    names = _lookup(packages, NPM_PACKAGES, NPM_PACKAGE_PREFIXES)
    return _Found(frozenset(names), _node_package_manager(manifest.get("packageManager")))


def _node_package_manager(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return NODE_PACKAGE_MANAGERS.get(value.split("@", 1)[0].strip().lower())


def _read_python(text: str) -> _Found:
    raw = [*_PY_LINE_NAME.findall(text), *_PY_QUOTED_NAME.findall(text), *_PY_EGG_NAME.findall(text)]
    packages = {_PY_NAME_SEPARATORS.sub("-", name).lower() for name in raw}
    names = _lookup(packages, PYTHON_PACKAGES, PYTHON_PACKAGE_PREFIXES)
    if _PYTEST_SECTION.search(text):  # [pytest], [tool:pytest] or [tool.pytest.ini_options]
        names.add("pytest")
    build_tool = "Poetry" if _POETRY_SECTION.search(text) else "uv" if _UV_SECTION.search(text) else None
    return _Found(frozenset(names), build_tool)


def _read_nuget(text: str) -> _Found:
    text = _drop_blocks(text, "<!--", "-->")
    packages = {package.strip().lower() for package in _NUGET_PACKAGE.findall(text)}
    return _Found(frozenset(_lookup(packages, {}, NUGET_PACKAGE_PREFIXES)))


def _read_ruby(text: str) -> _Found:
    gems = {gem.strip().lower() for gem in _RUBY_GEM.findall(text)}
    return _Found(frozenset(_lookup(gems, RUBY_GEMS)))


def _read_go(text: str) -> _Found:
    return _Found(frozenset(name for module, names in GO_MODULES.items() if module in text for name in names))


_MANIFEST_READERS: dict[str, Callable[[str], _Found]] = {
    "maven": _read_maven,
    "gradle": _read_gradle,
    "npm": _read_npm,
    "python": _read_python,
    "nuget": _read_nuget,
    "ruby": _read_ruby,
    "go": _read_go,
}


def _lookup(
    packages: Iterable[str],
    exact: dict[str, tuple[str, ...]],
    prefixes: dict[str, tuple[str, ...]] | None = None,
) -> set[str]:
    names: set[str] = set()
    for package in packages:
        names.update(exact.get(package, ()))
        for prefix, found in (prefixes or {}).items():
            if package.startswith(prefix):
                names.update(found)
    return names


def _drop_blocks(text: str, start: str, end: str) -> str:
    """Remove every start...end block; an unclosed block runs to the end. Linear, unlike a lazy regex."""
    parts: list[str] = []
    position = 0
    while (begin := text.find(start, position)) != -1:
        parts.append(text[position:begin])
        finish = text.find(end, begin + len(start))
        if finish == -1:
            return "".join(parts)
        position = finish + len(end)
    parts.append(text[position:])
    return "".join(parts)


def _primary_language(scan: _Scan) -> FactModel:
    counts = {
        language: count for language, count in scan.language_files.items() if language not in NON_PROGRAMMING_LANGUAGES
    }
    if not counts:
        return _unknown()
    language = min(counts, key=lambda name: (-counts[name], name))
    share = counts[language] / sum(counts.values())
    confidence = Confidence.HIGH if share > PRIMARY_LANGUAGE_HIGH_SHARE else Confidence.MEDIUM
    return FactModel(
        value=language,
        source=FactSource.CODE,
        evidence=list(scan.language_examples[language]),
        confidence=confidence,
    )


def _build_tool(scan: _Scan, named_build_tools: dict[str, str]) -> FactModel:
    if not scan.build_files:
        return _unknown()
    shallowest = min(_level(path) for path, _ in scan.build_files)
    evidence: dict[str, list[str]] = {}
    for path, tool in scan.build_files:
        if _level(path) == shallowest:
            actual, lock_file = _actual_build_tool(path, tool, scan.lock_files, named_build_tools)
            evidence.setdefault(actual, []).extend([path, lock_file] if lock_file else [path])
    for generic, specific in GENERIC_BUILD_TOOLS.items():
        if generic in evidence and not specific.isdisjoint(evidence):
            del evidence[generic]
    tools = sorted(evidence, key=str.casefold)
    if len(tools) == 1:
        return _fact(tools[0], evidence, Confidence.HIGH)
    return _fact(tools, evidence, Confidence.MEDIUM)


def _actual_build_tool(
    path: str, tool: str, lock_files: dict[tuple[str, str], str], named_build_tools: dict[str, str]
) -> tuple[str, str | None]:
    """npm or pip may really be Yarn, Poetry, ...: the manifest itself, or a lock file next to it, says so."""
    locks = LOCK_FILES.get(_name_of(path).lower())
    if locks is None:
        return tool, None
    if path in named_build_tools:
        return named_build_tools[path], None
    folder = _parent_of(path)
    for lock_name, locked_tool in locks.items():
        if lock_path := lock_files.get((folder, lock_name)):
            return locked_tool, lock_path
    return tool, None


def _names_fact(hits: list[_Hit], allowed: frozenset[str], *, single: bool) -> FactModel:
    """`single`: the fact names one thing (bdd_tool), so finding several at once is only medium confidence."""
    evidence: dict[str, list[str]] = {}
    strongest: dict[str, Confidence] = {}
    for hit in hits:
        if hit.name in allowed:
            evidence.setdefault(hit.name, []).append(hit.path)
            current = strongest.get(hit.name)
            if current is None or _CONFIDENCE_RANK[hit.confidence] > _CONFIDENCE_RANK[current]:
                strongest[hit.name] = hit.confidence
    if not evidence:
        return _unknown()

    names = sorted(evidence, key=str.casefold)
    confidence = _weakest(strongest.values())
    if not single:
        return _fact(names, evidence, confidence)
    if len(names) == 1:
        return _fact(names[0], evidence, confidence)
    return _fact(names, evidence, _weakest([confidence, Confidence.MEDIUM]))


def _fact(value: str | list[str], evidence: dict[str, list[str]], confidence: Confidence) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=_pick_evidence(evidence), confidence=confidence)


def _pick_evidence(paths_by_name: dict[str, list[str]]) -> list[str]:
    """At least one file per name, then more up to MAX_EVIDENCE; shallowest files first."""
    ordered = [sorted(set(paths), key=_shallow_first) for paths in paths_by_name.values()]
    limit = max(MAX_EVIDENCE, len(ordered))
    chosen = list(dict.fromkeys(paths[0] for paths in ordered))
    for path in sorted({path for paths in ordered for path in paths}, key=_shallow_first):
        if len(chosen) >= limit:
            break
        if path not in chosen:
            chosen.append(path)
    return sorted(chosen, key=_shallow_first)


def _weakest(confidences: Iterable[Confidence]) -> Confidence:
    return min(confidences, key=_CONFIDENCE_RANK.__getitem__)


def _unknown() -> FactModel:
    return FactModel(value=None, source=FactSource.CODE, evidence=[], confidence=Confidence.LOW)


def _join(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _name_of(path: str) -> str:
    return path.rpartition("/")[2]


def _parent_of(path: str) -> str:
    return path.rpartition("/")[0]


def _level(path: str) -> int:
    """0 for an entry in the project root."""
    return path.count("/")


def _shallow_first(path: str) -> tuple[int, str]:
    return _level(path), path
