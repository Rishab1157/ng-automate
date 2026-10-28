"""Tests for braces and path handling in the glob tool.

Each test runs on ripgrep (when installed) and on the Python glob fallback, with
the process folder somewhere else, as in the agent server.
"""

import time
from pathlib import Path

import pytest

from openhands.tools.glob import GlobAction
from openhands.tools.glob.impl import GlobExecutor
from openhands.tools.utils import _check_ripgrep_available


BACKENDS = [
    pytest.param(
        "ripgrep",
        marks=pytest.mark.skipif(
            not _check_ripgrep_available(), reason="ripgrep not available"
        ),
    ),
    "python",
]

JAVA = ["src/test/java/pages/LoginPage.java", "src/test/java/tests/LoginTest.java"]


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture(autouse=True)
def _process_folder_elsewhere(tmp_path_factory, monkeypatch):
    """The agent server's own folder is never the agent's working directory."""
    monkeypatch.chdir(tmp_path_factory.mktemp("server_cwd"))


def _write(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "project",
        [
            "pom.xml",
            "README.md",
            "config/app.json",
            "config/app.yaml",
            *JAVA,
            ".hidden/Secret.java",
        ],
    )


def _glob(working_dir: Path, backend: str, **kwargs):
    executor = GlobExecutor(working_dir=str(working_dir))
    executor._ripgrep_available = backend == "ripgrep"
    return executor(GlobAction(**kwargs))


def _found(observation, root: Path) -> list[str]:
    base = root.resolve()
    return sorted(
        Path(f).resolve().relative_to(base).as_posix() for f in observation.files
    )


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("**/*.{json,yaml}", ["config/app.json", "config/app.yaml"]),
        ("*.{json,yaml}", ["config/app.json", "config/app.yaml"]),
        ("src/**/*.{java,xml}", JAVA),
        ("{pom.xml,README.md}", ["README.md", "pom.xml"]),
        ("**/*.{java}", JAVA),
        ("src/**/*.java", JAVA),
        ("src/test/java/pages/*.java", JAVA[:1]),
    ],
)
def test_braces_and_relative_patterns(project, backend, pattern, expected):
    observation = _glob(project, backend, pattern=pattern)

    assert observation.is_error is False, observation.text
    assert _found(observation, project) == sorted(expected)


def test_relative_path_starts_at_the_working_directory(project, backend):
    observation = _glob(project, backend, pattern="*.java", path="src/test/java")

    assert observation.is_error is False, observation.text
    assert observation.search_path == str((project / "src/test/java").resolve())
    assert _found(observation, project) == JAVA


def test_parent_segments_in_a_relative_path_start_at_the_working_directory(
    tmp_path, backend
):
    working_dir = tmp_path / "a" / "b" / "work"
    working_dir.mkdir(parents=True)
    _write(tmp_path / "a", ["note.txt"])

    observation = _glob(working_dir, backend, pattern="*.txt", path="../..")

    assert observation.is_error is False, observation.text
    assert observation.search_path == str((tmp_path / "a").resolve())
    assert _found(observation, tmp_path / "a") == ["note.txt"]


def test_missing_relative_path_names_the_working_directory(project, backend):
    observation = _glob(project, backend, pattern="*.java", path="src/test/jav")

    assert observation.is_error is True
    assert "is not a valid directory" in observation.text
    assert "working directory" in observation.text


def test_absolute_pattern_with_a_path_searches_the_pattern_folder(project, backend):
    pattern = f"{project.resolve().as_posix()}/src/test/java/pages/*.java"

    observation = _glob(project, backend, pattern=pattern, path=str(project))

    assert observation.is_error is False, observation.text
    assert _found(observation, project) == JAVA[:1]


def test_folder_pattern_says_how_to_list_its_files(project, backend):
    observation = _glob(project, backend, pattern="src/test/java")

    assert observation.is_error is False
    assert observation.files == []
    assert "is a folder" in observation.text
    assert "src/test/java/**/*" in observation.text

    retry = _glob(project, backend, pattern="src/test/java/**/*")
    assert _found(retry, project) == JAVA


def test_spaces_and_unicode_names(tmp_path, backend):
    root = _write(
        tmp_path / "prøject",
        ["src/ünï cödé/Ünï Fïle.java", "notes/my notes.txt", "src/plain/A.java"],
    )

    in_folder = _glob(root, backend, pattern="src/ünï cödé/*.java")
    braces = _glob(root, backend, pattern="**/*.{java,txt}")

    assert _found(in_folder, root) == ["src/ünï cödé/Ünï Fïle.java"]
    assert _found(braces, root) == [
        "notes/my notes.txt",
        "src/plain/A.java",
        "src/ünï cödé/Ünï Fïle.java",
    ]


@pytest.mark.parametrize("pattern", ["*.{java", "*.java}", "[*.java"])
def test_unbalanced_patterns_get_an_answer_or_a_clear_error(project, backend, pattern):
    observation = _glob(project, backend, pattern=pattern)

    if backend == "python":
        # fnmatch reads an unclosed bracket or brace as a plain character.
        assert observation.is_error is False
        assert observation.files == []
    elif observation.is_error:
        # ripgrep refuses "*.{java" and "[*.java": say so, not "No files found".
        assert "ripgrep could not use the pattern" in observation.text
    else:
        # ripgrep reads "*.java}" as "*.java".
        assert _found(observation, project) == JAVA


def test_brace_bomb_is_refused_quickly(project):
    started = time.monotonic()

    observation = _glob(project, "python", pattern="{a,b}" * 30)

    assert time.monotonic() - started < 5
    assert observation.is_error is True
    assert "more than" in observation.text


def test_nested_braces(project, backend):
    """The Python fallback expands nested braces; ripgrep refuses them, and says so."""
    observation = _glob(project, backend, pattern="*.{j{ava,son},xml}")

    if backend == "ripgrep":
        assert observation.is_error is True
        assert "nested" in observation.text
    else:
        assert _found(observation, project) == sorted(
            JAVA + ["config/app.json", "pom.xml"]
        )
