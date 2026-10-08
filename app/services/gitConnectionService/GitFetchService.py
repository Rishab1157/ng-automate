"""Fetch one branch of a Git repository (latest commit only) into a zip.

The repo URL, token and branch all come from the QXcel git connection. A connection without a branch gets
GIT_DEFAULT_BRANCH ("main"); a repo that has no such branch (older repos use "master") falls back to the repo's own
default branch. A branch the connection names explicitly must exist: falling back would test the wrong code.

The token reaches git only through environment variables, so it never appears in the
URL, the command line, .git/config, logs or error messages.
"""

import base64
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr

from app.config import settings
from app.core.exceptions import ErrorMessages, GitFetchError, PayloadTooLargeError
from app.models.gitConnectionModel import GitCloneSourceModel, GitFetchResultModel
from app.utils.ArchiveUtils import create_archive, directory_size

# Username paired with a token in HTTP Basic auth, per provider.
_TOKEN_USERNAMES = {"GITHUB": "x-access-token", "GITLAB": "oauth2", "BITBUCKET": "x-token-auth"}
_DEFAULT_TOKEN_USERNAME = "x-access-token"
# Branch names only; a leading "-" could be read by git as an option.
_BRANCH_PATTERN = re.compile(r"^(?!-)[\w.\-/]+$")
# Shown in errors when the repo's default branch is used.
_DEFAULT_BRANCH_LABEL = "default"


class _BranchNotFoundError(GitFetchError):
    """The remote has no such branch."""


class GitFetchService:
    def __init__(self, timeout_seconds: int = settings.GIT_CLONE_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_to_archive(self, source: GitCloneSourceModel, destination: Path, max_bytes: int) -> GitFetchResultModel:
        """Clone the connection's branch of its repo and save it as a zip at `destination`."""
        _check_repo_url(source.repo_url)
        if source.branch is not None:
            _check_branch(source.branch)
        token = _secret(source.token)
        env = _git_env(token, source.provider_code)

        with tempfile.TemporaryDirectory(prefix="ngauto-clone-", ignore_cleanup_errors=True) as tmp:
            repo_dir = Path(tmp) / "repo"
            branch = self._clone(source, repo_dir, env, token)
            commit_sha = self._run_git(["-C", str(repo_dir), "rev-parse", "HEAD"], env, token, branch).strip()

            if directory_size(repo_dir) > max_bytes:
                raise PayloadTooLargeError(ErrorMessages.PROJECT_TOO_LARGE.format(limit_mb=max_bytes // (1024 * 1024)))
            create_archive(repo_dir, destination)

        return GitFetchResultModel(commit_sha=commit_sha, branch=branch)

    def _clone(self, source: GitCloneSourceModel, repo_dir: Path, env: dict[str, str], token: str | None) -> str:
        """Clone into `repo_dir`; returns the branch that was cloned."""
        if source.branch is not None:
            self._run_git(_clone_args(source.repo_url, repo_dir, source.branch), env, token, source.branch)
            return source.branch
        default = settings.GIT_DEFAULT_BRANCH
        try:
            self._run_git(_clone_args(source.repo_url, repo_dir, default), env, token, default)
            return default
        except _BranchNotFoundError:
            shutil.rmtree(repo_dir, ignore_errors=True)
        # No "main": clone whatever the remote's HEAD points to, and read its name.
        self._run_git(_clone_args(source.repo_url, repo_dir, None), env, token, _DEFAULT_BRANCH_LABEL)
        return self._run_git(
            ["-C", str(repo_dir), "symbolic-ref", "--short", "HEAD"], env, token, _DEFAULT_BRANCH_LABEL
        ).strip()

    def _run_git(self, args: list[str], env: dict[str, str], token: str | None, branch: str) -> str:
        # credential.helper= : never use credentials stored on this machine, only the connection's token.
        command = ["git", "-c", "credential.helper=", "-c", "core.symlinks=false", *args]
        try:
            result = subprocess.run(
                command, env=env, capture_output=True, encoding="utf-8", errors="replace",
                timeout=self.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            raise GitFetchError(ErrorMessages.GIT_TIMEOUT.format(seconds=self.timeout_seconds)) from None
        if result.returncode != 0:
            message = _describe_failure(result.stderr, token, branch)
            if message == ErrorMessages.GIT_BRANCH_NOT_FOUND.format(branch=branch):
                raise _BranchNotFoundError(message)
            raise GitFetchError(message)
        return result.stdout


def _clone_args(repo_url: str, repo_dir: Path, branch: str | None) -> list[str]:
    """Latest commit of one branch; without `branch`, the remote's default branch."""
    selected = ["--branch", branch] if branch is not None else []
    return [
        "clone", "--depth", "1", "--single-branch", "--no-recurse-submodules", *selected,
        "--", repo_url, str(repo_dir),
    ]


def _check_repo_url(repo_url: str) -> None:
    """Only remote https repos: file://, local paths or ssh could expose the server's own files."""
    parts = urlsplit(repo_url)
    if parts.scheme != "https" or not parts.hostname:
        raise GitFetchError(ErrorMessages.GIT_ONLY_HTTPS)
    if parts.username or parts.password:
        raise GitFetchError(ErrorMessages.GIT_CREDENTIALS_IN_URL)


def _check_branch(branch: str) -> None:
    if not _BRANCH_PATTERN.match(branch):
        raise GitFetchError(ErrorMessages.GIT_INVALID_BRANCH.format(branch=branch))


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def _git_env(token: str | None, provider_code: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # never hang waiting for a password
    env["GCM_INTERACTIVE"] = "never"  # Git Credential Manager (Windows): no pop-ups
    if token:
        username = _TOKEN_USERNAMES.get((provider_code or "").upper(), _DEFAULT_TOKEN_USERNAME)
        basic = base64.b64encode(f"{username}:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {basic}"
    return env


def _describe_failure(stderr: str, token: str | None, branch: str) -> str:
    text = stderr.replace(token, "***") if token else stderr
    lowered = text.lower()
    if "authentication failed" in lowered or "403" in lowered or "401" in lowered:
        return ErrorMessages.GIT_AUTH_FAILED
    if "remote branch" in lowered and "not found" in lowered:
        return ErrorMessages.GIT_BRANCH_NOT_FOUND.format(branch=branch)
    if "not found" in lowered:
        return ErrorMessages.GIT_REPO_NOT_FOUND
    if "could not resolve host" in lowered:
        return ErrorMessages.GIT_HOST_UNREACHABLE
    last_line = text.strip().splitlines()[-1] if text.strip() else "unknown error"
    return ErrorMessages.GIT_FAILED.format(reason=last_line)
