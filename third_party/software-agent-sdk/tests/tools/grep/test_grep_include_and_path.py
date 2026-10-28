"""Tests for how the grep tool reads include, path and the regex.

Each test runs on every search backend that is installed (ripgrep, system grep,
Python search), with the process folder somewhere else, as in the agent server.
"""

import time
from pathlib import Path

import pytest

from openhands.tools.grep import GrepAction
from openhands.tools.grep.impl import GrepExecutor
from openhands.tools.utils import _check_grep_available, _check_ripgrep_available


BACKENDS = [
    pytest.param(
        "ripgrep",
        marks=pytest.mark.skipif(
            not _check_ripgrep_available(), reason="ripgrep not available"
        ),
    ),
    pytest.param(
        "grep",
        marks=pytest.mark.skipif(
            not _check_grep_available(), reason="grep not available"
        ),
    ),
    "python",
]

JAVA = ["src/test/java/pages/LoginPage.java", "src/test/java/tests/LoginTest.java"]
PROPERTIES = "src/test/resources/config.properties"


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture(autouse=True)
def _process_folder_elsewhere(tmp_path_factory, monkeypatch):
    """The agent server's own folder is never the agent's working directory."""
    monkeypatch.chdir(tmp_path_factory.mktemp("server_cwd"))


def _write(root: Path, files: dict[str, str | bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "project",
        {
            "pom.xml": (
                "<project><artifactId>demo</artifactId><!-- mark --></project>\n"
            ),
            "README.md": "Run the public tests. mark\n",
            "src/test/java/pages/LoginPage.java": (
                "package pages;\nimport org.openqa.selenium.By;\n"
                "public class LoginPage { } // mark\n"
            ),
            "src/test/java/tests/LoginTest.java": (
                "package tests;\nimport org.junit.Test;\n"
                "public class LoginTest {\n  @Test void login() { } // mark\n}\n"
            ),
            PROPERTIES: "base.url=http://localhost\ntimeout = 30\n# mark\n",
            ".hidden/Secret.java": "public class Secret { } // mark\n",
        },
    )


def _search(working_dir: Path, backend: str, **kwargs):
    executor = GrepExecutor(working_dir=str(working_dir))
    executor._search_backend = backend
    return executor(GrepAction(**kwargs))


def _found(observation, root: Path) -> list[str]:
    base = root.resolve()
    return sorted(Path(m).relative_to(base).as_posix() for m in observation.matches)


# --- include -----------------------------------------------------------------


@pytest.mark.parametrize(
    "include, expected",
    [
        ("*.java", JAVA),
        ("**/*.java", JAVA),
        ("src/test/java/**/*.java", JAVA),
        ("./src/test/java/**/*.java", JAVA),
        ("src/test/java/*/*.java", JAVA),
        ("src/**/pages/*.java", JAVA[:1]),
        ("src/test/java/**", JAVA),
        ("**", JAVA + ["README.md", "pom.xml", PROPERTIES]),
        ("*.{java,xml}", JAVA + ["pom.xml"]),
        ("**/*.{java,properties}", JAVA + [PROPERTIES]),
        ("{pom.xml,src/**/*.properties}", ["pom.xml", PROPERTIES]),
        ("*.{j{ava,son},xml}", JAVA + ["pom.xml"]),
        ("/pom.xml", ["pom.xml"]),
        ("Login*.java", JAVA),
        ("src/*.java", []),  # "*" does not cross folders
    ],
)
def test_include_matches_file_names_and_relative_paths(
    project, backend, include, expected
):
    observation = _search(project, backend, pattern="mark", include=include)

    assert observation.is_error is False, observation.text
    assert _found(observation, project) == sorted(expected)


def test_include_is_relative_to_the_path_being_searched(project, backend):
    observation = _search(
        project, backend, pattern="mark", path="src/test/java", include="pages/*.java"
    )

    assert _found(observation, project) == JAVA[:1]


def test_include_may_be_an_absolute_path_inside_the_search_folder(project, backend):
    include = f"{project.resolve().as_posix()}/src/**/*.java"

    observation = _search(project, backend, pattern="mark", include=include)

    assert _found(observation, project) == JAVA


def test_include_never_reaches_into_hidden_folders(project, backend):
    _write(project, {".env": "mark\n"})

    every_java = _search(project, backend, pattern="mark", include="**/*.java")
    dot_env = _search(project, backend, pattern="mark", include="**/.env")

    assert _found(every_java, project) == JAVA
    assert _found(dot_env, project) == [".env"]


def test_include_naming_a_folder_says_to_use_path(project, backend):
    observation = _search(project, backend, pattern="mark", include="src/test/java")

    assert observation.is_error is False
    assert observation.matches == []
    assert "No files found" in observation.text
    assert "is a folder" in observation.text
    assert "path='src/test/java'" in observation.text

    # ...and the suggested call works.
    retry = _search(project, backend, pattern="mark", path="src/test/java")
    assert _found(retry, project) == JAVA


@pytest.mark.parametrize(
    "include, suggestion",
    [
        ("*.java,*.xml", "{*.java,*.xml}"),
        ("*.java, *.xml", "{*.java,*.xml}"),
        ("*.java *.xml", "{*.java,*.xml}"),
        ("pom.xml,build.gradle", "{pom.xml,build.gradle}"),
    ],
)
def test_include_list_suggests_braces(project, backend, include, suggestion):
    observation = _search(project, backend, pattern="mark", include=include)

    assert observation.matches == []
    assert "not a list" in observation.text
    assert suggestion in observation.text

    retry = _search(project, backend, pattern="mark", include=suggestion)
    assert retry.matches


def test_unmatched_path_include_explains_what_it_is_relative_to(project, backend):
    observation = _search(
        project, backend, pattern="mark", include="test/java/**/*.java"
    )

    assert observation.matches == []
    assert "relative to the search folder" in observation.text
    assert str(project.resolve()) in observation.text


def test_unmatched_include_in_a_hidden_folder_says_to_use_path(project, backend):
    hidden = _search(project, backend, pattern="mark", include=".hidden/*.java")
    dot_slash = _search(project, backend, pattern="mark", include="./x/**/*.java")

    assert hidden.matches == []
    assert "start with '.'" in hidden.text
    assert dot_slash.matches == []
    assert "start with '.'" not in dot_slash.text

    retry = _search(project, backend, pattern="mark", path=".hidden")
    assert _found(retry, project) == [".hidden/Secret.java"]


def test_include_with_spaces_and_unicode(tmp_path, backend):
    root = _write(
        tmp_path / "prøject",
        {
            "src/ünï cödé/Ünï Fïle.java": "mark café\n",
            "src/plain/A.java": "mark\n",
            "notes/my notes.txt": "mark\n",
        },
    )

    in_folder = _search(root, backend, pattern="mark", include="src/ünï cödé/*.java")
    by_name = _search(root, backend, pattern="mark", include="**/Ünï*.java")
    braces = _search(root, backend, pattern="mark", include="*.{java,txt}")
    exact = _search(root, backend, pattern="mark", include="my notes.txt")
    unicode_regex = _search(root, backend, pattern="café")

    assert _found(in_folder, root) == ["src/ünï cödé/Ünï Fïle.java"]
    assert _found(by_name, root) == ["src/ünï cödé/Ünï Fïle.java"]
    assert _found(braces, root) == [
        "notes/my notes.txt",
        "src/plain/A.java",
        "src/ünï cödé/Ünï Fïle.java",
    ]
    assert _found(exact, root) == ["notes/my notes.txt"]
    assert _found(unicode_regex, root) == ["src/ünï cödé/Ünï Fïle.java"]


def test_include_still_caps_results_at_100(tmp_path, backend):
    _write(tmp_path, {f"logs/day{i:03d}/run.txt": "mark\n" for i in range(250)})

    observation = _search(tmp_path, backend, pattern="mark", include="logs/**/*.txt")

    assert observation.is_error is False
    assert len(observation.matches) == 100
    assert observation.truncated is True


def test_brace_bomb_include_is_refused_quickly(project, backend):
    started = time.monotonic()

    observation = _search(project, backend, pattern="mark", include="{a,b}" * 30)

    assert time.monotonic() - started < 5
    assert observation.is_error is True
    assert "include" in observation.text


@pytest.mark.parametrize("include", ["*.{java", "*.java}", "[*.java", "**/**/**/*"])
def test_odd_include_patterns_do_not_crash(project, backend, include):
    observation = _search(project, backend, pattern="mark", include=include)

    assert observation.is_error is False, observation.text


# --- regex -------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("public|artifactId", JAVA + ["README.md", "pom.xml"]),
        (r"class\s+Login\w+", JAVA),
        (r"(?:Login)(?:Page|Test)\b", JAVA),
        (r"class \w+(?= \{)", JAVA),  # look-ahead
        (r"timeout\s*=\s*\d+", [PROPERTIES]),
        ("^import", JAVA),  # ^ and $ work per line, as in grep and ripgrep
        ("// mark$", JAVA),
        (r"(?i)LOGINPAGE", JAVA[:1]),
    ],
)
def test_pattern_is_a_python_style_regex(project, backend, pattern, expected):
    observation = _search(project, backend, pattern=pattern)

    assert observation.is_error is False, observation.text
    assert _found(observation, project) == sorted(expected)


@pytest.mark.parametrize("pattern", ["-x", "-e", "--headless", "-rf"])
def test_pattern_starting_with_a_dash_is_searched_not_read_as_an_option(
    tmp_path, backend, pattern
):
    _write(tmp_path, {"flags.txt": f"chrome {pattern} on\n", "other.txt": "none\n"})

    observation = _search(tmp_path, backend, pattern=pattern)

    assert observation.is_error is False, observation.text
    assert _found(observation, tmp_path) == ["flags.txt"]


def test_python_only_regex_syntax_still_finds_files(tmp_path, backend):
    _write(tmp_path, {"menu.txt": "café\n", "other.txt": "cafe\n"})

    observation = _search(
        tmp_path, backend, pattern=r"caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    )

    assert _found(observation, tmp_path) == ["menu.txt"]


@pytest.mark.parametrize("pattern", ["foo\nbar", r"foo\nbar", r"(?s)foo.bar"])
def test_pattern_spanning_lines_is_searched_across_lines(tmp_path, backend, pattern):
    _write(
        tmp_path,
        {"both.txt": "foo\nbar\n", "foo.txt": "foo\n", "bar.txt": "bar\n"},
    )

    observation = _search(tmp_path, backend, pattern=pattern)

    assert observation.is_error is False, observation.text
    assert _found(observation, tmp_path) == ["both.txt"]


def test_binary_files_are_skipped(tmp_path, backend):
    _write(tmp_path, {"text.txt": "mark\n", "image.bin": b"mark\x00\x01\x02"})

    observation = _search(tmp_path, backend, pattern="mark")

    assert _found(observation, tmp_path) == ["text.txt"]


@pytest.mark.parametrize("pattern", ["[invalid", "(unclosed", "*start", r"\K"])
def test_invalid_regex_is_an_error_not_an_empty_result(project, backend, pattern):
    observation = _search(project, backend, pattern=pattern)

    assert observation.is_error is True
    assert "Invalid regex pattern" in observation.text


# --- path --------------------------------------------------------------------


def test_relative_path_starts_at_the_working_directory(project, backend):
    observation = _search(project, backend, pattern="mark", path="src/test/java")

    assert observation.is_error is False, observation.text
    assert observation.search_path == str((project / "src/test/java").resolve())
    assert _found(observation, project) == JAVA


def test_parent_segments_in_a_relative_path_start_at_the_working_directory(
    tmp_path, backend
):
    working_dir = tmp_path / "a" / "b" / "work"
    working_dir.mkdir(parents=True)
    _write(tmp_path / "a", {"note.txt": "mark\n"})

    observation = _search(working_dir, backend, pattern="mark", path="../..")

    assert observation.is_error is False, observation.text
    assert observation.search_path == str((tmp_path / "a").resolve())
    assert _found(observation, tmp_path / "a") == ["note.txt"]


def test_absolute_path_still_works(project, backend):
    folder = project / "src" / "test" / "java" / "pages"

    observation = _search(project, backend, pattern="mark", path=str(folder))

    assert _found(observation, project) == JAVA[:1]


def test_missing_relative_path_names_the_folder_it_tried(project, backend):
    observation = _search(project, backend, pattern="mark", path="src/test/jav")

    assert observation.is_error is True
    assert "not a valid directory" in observation.text
    assert str((project / "src/test/jav").resolve()) in observation.text
    assert "working directory" in observation.text


def test_file_as_path_says_how_to_search_it(project, backend):
    observation = _search(project, backend, pattern="mark", path="pom.xml")

    assert observation.is_error is True
    assert "not a valid directory" in observation.text
    assert "is a file" in observation.text
    assert "include='pom.xml'" in observation.text
