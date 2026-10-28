"""Push a git project's files back to its repository: to the connection's branch, or to a new branch.

One commit of the repository is fetched into a temporary folder. The project's files are put over it (the project's
version always wins), the files deleted in the project since it was fetched are deleted, and `git add -A` decides what
is committed: the repository's .gitignore keeps ignored files out. Files only the remote has (a teammate's) are left
alone. Tool folders and test results are never pushed.

The token is handled as in GitFetchService (same helpers): only through environment variables, never in the URL, the
command line, .git/config, logs or error messages.
"""

import logging
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from app.config import settings
from app.core.exceptions import ErrorMessages, GitPushError
from app.models.gitConnectionModel import (
    GitCloneSourceModel,
    GitCommitAuthorModel,
    GitPushBase,
    GitPushResultModel,
    GitPushTarget,
)
from app.utils.ArchiveUtils import EXPORT_SKIPPED_DIRS, iter_files

from .GitFetchService import check_branch, check_repo_url, describe_failure, git_env, run_git, scrub, secret_value

logger = logging.getLogger(__name__)

# A new branch is named after the connection: its name in lowercase letters, digits and . _ - only.
NEW_BRANCH_FALLBACK_NAME = "ngautomate"
MAX_BRANCH_NAME_PART_CHARS = 50
_BRANCH_NAME_UNSAFE = re.compile(r"[^a-z0-9._-]+")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
# Names a repository's own folder goes by (also on Windows, where ".GIT", ".git." and the short name "git~1" are it).
_GIT_FOLDER_NAMES = frozenset({".git", "git~1"})
# How the push of a token without write access fails, per provider (GitHub, GitLab, Bitbucket).
_WRITE_DENIED = re.compile(
    r"returned error: 403|permission to \S+ denied|write access to repository not granted"
    r"|not allowed to push code to this project|lack one or more required privilege scopes"
)
_REJECTED = re.compile(r"\[(?:remote )?rejected\][^(\n]*\((?P<reason>[^)\n]+)\)", re.IGNORECASE)
_REMOTE_SAID = re.compile(r"^remote:[ \t]*(?:error:[ \t]*)?(?P<text>.*?)[ \t]*$", re.MULTILINE | re.IGNORECASE)
_OUTDATED_REASONS = frozenset({"fetch first", "non-fast-forward"})
MAX_REASON_CHARS = 300
_WORK_FOLDER_LABEL = "<work folder>"


class GitPushService:
    def __init__(self, timeout_seconds: int = settings.GIT_CLONE_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    def push(
        self,
        source: GitCloneSourceModel,
        *,
        project_dir: Path,
        snapshot: Path,
        target: GitPushTarget,
        branch: str,
        base_commit: str,
        message: str,
        author: GitCommitAuthorModel,
    ) -> GitPushResultModel:
        """Push the files of `project_dir` to the connection's repo as one commit by `author`.

        snapshot: the project as it was fetched (its source.zip); a file it has that `project_dir` no longer has was
        deleted in NG Automate. branch: the connection's branch (target current; a new branch falls back to its latest
        commit). base_commit: the commit the project was fetched at, where a new branch starts.
        """
        check_repo_url(source.repo_url, GitPushError)
        check_branch(branch, GitPushError)
        token = secret_value(source.token)
        env = git_env(token, source.provider_code) | _author_env(author)

        with tempfile.TemporaryDirectory(prefix="ngauto-push-", ignore_cleanup_errors=True) as tmp:
            repo = _Repo(Path(tmp), env, token, self.timeout_seconds)
            repo.init(source.repo_url)
            if target == GitPushTarget.CURRENT:
                repo.fetch_branch(branch)
                based_on = GitPushBase.BRANCH_HEAD
            else:
                based_on = repo.fetch_commit(base_commit, branch)
            repo.git("checkout", "-q", "-B", branch, "FETCH_HEAD", branch=branch)

            files_changed = repo.put_project(project_dir, snapshot)
            if files_changed == 0:
                pushed_to = branch if target == GitPushTarget.CURRENT else None
                return GitPushResultModel(
                    target=target, branch=pushed_to, repo_url=source.repo_url,
                    web_url=_web_url(source.repo_url, source.provider_code, pushed_to),
                    nothing_to_push=True, based_on=based_on,
                )

            if target == GitPushTarget.CURRENT:
                pushed_to = branch
            else:
                name = new_branch_name(source.userdefined_name, _timestamp())
                pushed_to = _free_name(name, repo.remote_branches(name))
            commit_sha = repo.commit(message)
            repo.push(pushed_to, set_upstream=target == GitPushTarget.NEW_BRANCH)

        logger.info(
            "Pushed %d file(s) to %s, branch %s (commit %s, on the %s)",
            files_changed, source.repo_url, pushed_to, commit_sha[:8], based_on.value,
        )
        return GitPushResultModel(
            target=target, branch=pushed_to, repo_url=source.repo_url,
            web_url=_web_url(source.repo_url, source.provider_code, pushed_to),
            commit_sha=commit_sha, files_changed=files_changed, based_on=based_on,
        )


class _Repo:
    """The temporary repository one push works in."""

    def __init__(self, folder: Path, env: dict[str, str], token: str | None, timeout_seconds: int) -> None:
        self.folder = folder
        self.path = folder / "repo"
        self.env = env
        self.token = token
        self.timeout_seconds = timeout_seconds

    def init(self, repo_url: str) -> None:
        self._run(["init", "-q", str(self.path)], branch="")
        self.git("remote", "add", "origin", repo_url, branch="")

    def fetch_branch(self, branch: str) -> None:
        """The branch's latest commit (into FETCH_HEAD)."""
        self.git("fetch", "-q", "--depth", "1", "--no-tags", "origin", f"refs/heads/{branch}", branch=branch)

    def fetch_commit(self, commit_sha: str, branch: str) -> GitPushBase:
        """The project's commit; the branch's latest commit when the remote no longer has it."""
        if _COMMIT_SHA.match(commit_sha):
            args = ["-C", str(self.path), "fetch", "-q", "--depth", "1", "--no-tags", "origin", commit_sha]
            result = run_git(args, self.env, self.timeout_seconds, GitPushError)
            if result.returncode == 0:
                return GitPushBase.RECORDED_COMMIT
            reason = describe_failure(self._without_work_folder(result.stderr), self.token, branch)
            logger.info("Commit %s cannot be fetched (%s): the new branch starts from %s", commit_sha[:8], reason, branch)
        else:
            logger.info("The project has no usable commit id: the new branch starts from %s", branch)
        self.fetch_branch(branch)
        return GitPushBase.BRANCH_HEAD

    def put_project(self, project_dir: Path, snapshot: Path) -> int:
        """The project's files over the repo's, its deletions applied, then `git add -A`. Returns the number of files
        the commit changes."""
        root = self.path.resolve()
        git_dir = (root / ".git").resolve()
        project_root = project_dir.resolve()
        for file in iter_files(project_root, EXPORT_SKIPPED_DIRS):
            target = _repo_file(root, git_dir, file.relative_to(project_root).as_posix())
            if target is not None:
                _make_room(root, target)
                # The content only: a file the repo has keeps its mode (an executable gradlew stays executable).
                shutil.copyfile(file, target)
        for path in _snapshot_files(snapshot):
            if os.path.lexists(project_root.joinpath(*PurePosixPath(path).parts)):
                continue
            target = _repo_file(root, git_dir, path)
            if target is not None and (target.is_file() or target.is_symlink()):
                target.unlink()
        self.git("add", "-A", branch="")
        staged = self.git("diff", "--cached", "--name-only", "--no-renames", "-z", branch="")
        return len([path for path in staged.split("\0") if path])

    def remote_branches(self, prefix: str) -> set[str]:
        """The remote's branches whose names start with `prefix`."""
        listing = self.git("ls-remote", "--heads", "origin", f"refs/heads/{prefix}*", branch=prefix)
        return {line.split("\t", 1)[1].removeprefix("refs/heads/") for line in listing.splitlines() if "\t" in line}

    def commit(self, message: str) -> str:
        """Commit everything staged; returns the commit's sha."""
        message_file = self.folder / "COMMIT_MESSAGE"
        message_file.write_text(message, encoding="utf-8", newline="\n")
        self.git("commit", "-q", f"--file={message_file}", branch="")
        return self.git("rev-parse", "HEAD", branch="").strip()

    def push(self, branch: str, set_upstream: bool) -> None:
        """The commit to `branch` of the remote (a new branch is created there)."""
        upstream = ["-u"] if set_upstream else []
        refspec = f"HEAD:refs/heads/{branch}"
        self._run(["-C", str(self.path), "push", "-q", *upstream, "origin", refspec], branch, describe_push_failure)

    def git(self, *args: str, branch: str) -> str:
        """A git command in the repo (`branch`: the branch named in its error messages)."""
        return self._run(["-C", str(self.path), *args], branch)

    def _run(self, args: list[str], branch: str, describe=describe_failure) -> str:
        result = run_git(args, self.env, self.timeout_seconds, GitPushError)
        if result.returncode != 0:
            raise GitPushError(describe(self._without_work_folder(result.stderr), self.token, branch))
        return result.stdout

    def _without_work_folder(self, text: str) -> str:
        """Error text without this server's temporary paths."""
        for folder in {str(self.folder.resolve()), str(self.folder)}:
            text = text.replace(folder, _WORK_FOLDER_LABEL)
        return text


def describe_push_failure(stderr: str, token: str | None, branch: str) -> str:
    """Why a push failed, in plain words (git's own text is scrubbed of the token first)."""
    text = scrub(stderr, token)
    lowered = text.lower()
    if "authentication failed" in lowered or "returned error: 401" in lowered:
        return ErrorMessages.GIT_AUTH_FAILED
    if _WRITE_DENIED.search(lowered):
        return ErrorMessages.GIT_NO_WRITE_ACCESS
    rejected = _REJECTED.search(text)
    if rejected and rejected["reason"].strip().lower() in _OUTDATED_REASONS:
        return ErrorMessages.GIT_PUSH_OUTDATED.format(branch=branch)
    if rejected:
        said = [match["text"] for match in _REMOTE_SAID.finditer(text) if match["text"]]
        said = [line for line in said if not line.lower().startswith("trace:")]
        reason = " ".join(said[:3])[:MAX_REASON_CHARS] or rejected["reason"].strip()
        return ErrorMessages.GIT_PUSH_REJECTED.format(reason=reason)
    return describe_failure(text, None, branch)


def new_branch_name(connection_name: str, timestamp: str) -> str:
    """<connection name>-<timestamp>, a valid branch name whatever the connection is called: lowercase ASCII letters,
    digits and . _ - only, no "..", not starting or ending with a dot or a dash."""
    ascii_name = unicodedata.normalize("NFKD", connection_name).encode("ascii", "ignore").decode().lower()
    name = _BRANCH_NAME_UNSAFE.sub("-", ascii_name)
    name = re.sub(r"\.{2,}", ".", re.sub(r"-{2,}", "-", name))
    name = name.strip(".-")[:MAX_BRANCH_NAME_PART_CHARS].strip(".-")
    return f"{name or NEW_BRANCH_FALLBACK_NAME}-{timestamp}"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _free_name(name: str, taken: set[str]) -> str:
    """`name`, or name-2, name-3, ... when the remote has it already."""
    if name not in taken:
        return name
    number = 2
    while f"{name}-{number}" in taken:
        number += 1
    return f"{name}-{number}"


def _author_env(author: GitCommitAuthorModel) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": author.name,
        "GIT_AUTHOR_EMAIL": author.email,
        "GIT_COMMITTER_NAME": author.name,
        "GIT_COMMITTER_EMAIL": author.email,
    }


def _snapshot_files(snapshot: Path) -> list[str]:
    """The project's files as it was fetched ("/"-separated paths in its source.zip)."""
    try:
        with zipfile.ZipFile(snapshot) as archive:
            return [entry.filename.replace("\\", "/") for entry in archive.infolist() if not entry.is_dir()]
    except (OSError, zipfile.BadZipFile):
        logger.warning("The project's source.zip cannot be read: no file is deleted by this push")
        return []


def _repo_file(root: Path, git_dir: Path, path: str) -> Path | None:
    """Where a project file goes in the repo. None for what a push never touches: tool folders and test results, the
    repository's own .git folder, and anything outside the repo."""
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts:
        return None
    if any(part in EXPORT_SKIPPED_DIRS for part in parts[:-1]) or any(_is_git_folder(part) for part in parts):
        return None
    target = root.joinpath(*parts)
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root) or resolved.is_relative_to(git_dir):
        return None
    return target


def _is_git_folder(name: str) -> bool:
    return name.lower().rstrip(" .") in _GIT_FOLDER_NAMES


def _make_room(root: Path, target: Path) -> None:
    """The project's file wins over what the repo has in its way: a file where the project has a folder, or a folder
    where the project has a file."""
    current = root
    for part in target.relative_to(root).parts[:-1]:
        current = current / part
        if current.is_symlink() or current.is_file():
            current.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)


def _web_url(repo_url: str, provider_code: str | None, branch: str | None) -> str | None:
    """The branch in the provider's web UI: GitHub and GitLab (also self-hosted), and bitbucket.org."""
    parts = urlsplit(repo_url)
    host = (parts.hostname or "").lower()
    path = parts.path.strip("/").removesuffix(".git").strip("/")
    if branch is None or not host or not path:
        return None
    base = f"https://{host}{f':{parts.port}' if parts.port else ''}/{path}"
    ref = quote(branch, safe="/")
    provider = (provider_code or "").upper()
    if host == "github.com" or provider == "GITHUB":
        return f"{base}/tree/{ref}"
    if host == "gitlab.com" or provider == "GITLAB":
        return f"{base}/-/tree/{ref}"
    if host == "bitbucket.org":
        return f"{base}/src/{ref}"
    return None
