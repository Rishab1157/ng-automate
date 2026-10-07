from pydantic import BaseModel, SecretStr


class GitCloneSourceModel(BaseModel):
    """Everything needed to clone a QXcel git connection. Internal only: never returned by the API."""

    id: str
    userdefined_name: str
    repo_url: str
    branch: str
    provider_code: str | None = None
    # SecretStr: hidden from print, repr and logs.
    token: SecretStr | None = None


class GitFetchResultModel(BaseModel):
    commit_sha: str
    branch: str
