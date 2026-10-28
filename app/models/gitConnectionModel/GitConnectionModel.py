from enum import Enum

from pydantic import BaseModel, SecretStr


class GitCloneSourceModel(BaseModel):
    """Everything needed to clone a QXcel git connection. Internal only: never returned by the API."""

    id: str
    userdefined_name: str
    repo_url: str
    # None when the connection has no branch: the fetch then uses the default branch.
    branch: str | None = None
    provider_code: str | None = None
    # SecretStr: hidden from print, repr and logs.
    token: SecretStr | None = None


class GitFetchResultModel(BaseModel):
    commit_sha: str
    branch: str


class GitPushTarget(str, Enum):
    CURRENT = "current"  # the connection's branch, at its latest commit
    NEW_BRANCH = "new_branch"  # a new branch, from the commit the project was created from


class GitPushBase(str, Enum):
    """The commit the pushed files were put on."""

    RECORDED_COMMIT = "recorded_commit"  # the commit the project was created from
    BRANCH_HEAD = "branch_head"  # the branch's latest commit


class GitCommitAuthorModel(BaseModel):
    name: str
    email: str


class GitPushResultModel(BaseModel):
    target: GitPushTarget
    # The branch pushed to. None only when nothing was pushed to a new branch (no branch was created).
    branch: str | None = None
    repo_url: str
    # The branch in the provider's web UI (GitHub, GitLab, Bitbucket); None for other hosts or when nothing was pushed
    # to a new branch.
    web_url: str | None = None
    commit_sha: str | None = None  # the new commit; None when nothing was pushed
    files_changed: int = 0
    nothing_to_push: bool = False
    based_on: GitPushBase
