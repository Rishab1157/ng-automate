"""Fetch one branch of a Git repository (latest commit only) into a zip.

The token reaches git only through environment variables, so it never appears in the
URL, the command line, .git/config, logs or error messages.
"""

import base64
import os
import re
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


class GitFetchService:
    def __init__(self, timeout_seconds: int = settings.GIT_CLONE_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_to_archive(
        self, source: GitCloneSourceModel, branch: str, destination: Path, max_bytes: int
    ) -> GitFetchResultModel:
        """Clone `branch` of the source repo and save it as a zip at `destination`."""
        _check_repo_url(source.repo_url)
        _check_branch(branch)
        token = _secret(source.token)
        env = _git_env(token, source.provider_code)

        with tempfile.TemporaryDirectory(prefix="ngauto-clone-", ignore_cleanup_errors=True) as tmp:
            repo_dir = Path(tmp) / "repo"
            self._run_git(
                ["clone", "--depth", "1", "--single-branch", "--no-recurse-submodules",
                 "--branch", branch, "--", source.repo_url, str(repo_dir)],
                env, token, branch,
            )
            commit_sha = self._run_git(["-C", str(repo_dir), "rev-parse", "HEAD"], env, token, branch).strip()

            if directory_size(repo_dir) > max_bytes:
                raise PayloadTooLargeError(ErrorMessages.PROJECT_TOO_LARGE.format(limit_mb=max_bytes // (1024 * 1024)))
            create_archive(repo_dir, destination)

        return GitFetchResultModel(commit_sha=commit_sha, branch=branch)

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
            raise GitFetchError(_describe_failure(result.stderr, token, branch))
        return result.stdout


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
