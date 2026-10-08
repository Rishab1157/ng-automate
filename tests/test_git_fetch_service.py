"""Which branch a git connection clones: its own, else main, else the repo's default branch."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.exceptions import ErrorMessages, GitFetchError
from app.models.gitConnectionModel import GitCloneSourceModel, GitConnectionMapper
from app.services.gitConnectionService import GitFetchService

REPO_URL = "https://github.com/acme/tests.git"
MAIN_MISSING = "warning: Could not find remote branch main to clone.\nfatal: Remote branch main not found in upstream origin\n"


def _source(branch: str | None = None) -> GitCloneSourceModel:
    return GitCloneSourceModel(id="c1", userdefined_name="Tests", repo_url=REPO_URL, branch=branch, token=SecretStr("tok-123456789"))


class FakeGit:
    """Stands in for subprocess.run: answers each git call in order and records the arguments."""

    def __init__(self, *answers: tuple[int, str, str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append(command)
        code, stdout, stderr = self.answers.pop(0)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


@pytest.fixture
def fake_git(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(*answers: tuple[int, str, str]) -> FakeGit:
        fake = FakeGit(*answers)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


def _clone(source: GitCloneSourceModel, tmp_path: Path) -> str:
    service = GitFetchService(timeout_seconds=5)
    return service._clone(source, tmp_path / "repo", env={}, token="tok-123456789")


def _branch_args(command: list[str]) -> list[str]:
    return command[command.index("--branch"):command.index("--branch") + 2] if "--branch" in command else []


def test_connection_branch_is_cloned(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git((0, "", ""))

    assert _clone(_source("develop"), tmp_path) == "develop"
    assert _branch_args(git.calls[0]) == ["--branch", "develop"]


def test_connection_without_branch_clones_main(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git((0, "", ""))

    assert _clone(_source(None), tmp_path) == "main"
    assert _branch_args(git.calls[0]) == ["--branch", "main"]


def test_repo_without_main_falls_back_to_its_default_branch(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git((128, "", MAIN_MISSING), (0, "", ""), (0, "master\n", ""))

    assert _clone(_source(None), tmp_path) == "master"
    assert _branch_args(git.calls[1]) == []  # the remote's HEAD
    assert git.calls[2][-3:] == ["symbolic-ref", "--short", "HEAD"]


def test_missing_connection_branch_is_an_error_not_a_fallback(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git((128, "", MAIN_MISSING.replace("main", "release")))

    with pytest.raises(GitFetchError) as caught:
        _clone(_source("release"), tmp_path)

    assert caught.value.message == ErrorMessages.GIT_BRANCH_NOT_FOUND.format(branch="release")
    assert len(git.calls) == 1


def test_other_clone_errors_are_not_treated_as_a_missing_branch(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git((128, "", "fatal: Authentication failed for 'https://github.com/acme/tests.git/'\n"))

    with pytest.raises(GitFetchError) as caught:
        _clone(_source(None), tmp_path)

    assert caught.value.message == ErrorMessages.GIT_AUTH_FAILED
    assert len(git.calls) == 1


def test_option_like_branch_is_refused_before_git_runs(fake_git: Any, tmp_path: Path) -> None:
    git = fake_git()

    with pytest.raises(GitFetchError):
        GitFetchService().fetch_to_archive(_source("--upload-pack=x"), tmp_path / "a.zip", 10_000)

    assert git.calls == []


@pytest.mark.parametrize(("stored", "expected"), [("develop", "develop"), ("", None), ("  ", None), (None, None)])
def test_mapper_keeps_only_a_real_branch(stored: str | None, expected: str | None) -> None:
    doc = {"_id": "c1", "repo_url": REPO_URL, "branch": stored}

    assert GitConnectionMapper.to_clone_source_model(doc).branch == expected


@pytest.mark.network
def test_real_repo_whose_default_branch_is_master(tmp_path: Path) -> None:
    source = GitCloneSourceModel(id="c1", userdefined_name="Hello", repo_url="https://github.com/octocat/Hello-World.git")

    result = GitFetchService(timeout_seconds=120).fetch_to_archive(source, tmp_path / "hello.zip", 50 * 1024 * 1024)

    assert result.branch == "master"
    assert len(result.commit_sha) == 40
    assert (tmp_path / "hello.zip").is_file()
