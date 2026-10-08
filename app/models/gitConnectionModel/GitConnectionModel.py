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
