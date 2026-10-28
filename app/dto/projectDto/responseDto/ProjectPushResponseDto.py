from pydantic import BaseModel

from app.models.gitConnectionModel import GitPushBase, GitPushResultModel, GitPushTarget


class ProjectPushResponseDTO(BaseModel):
    target: GitPushTarget
    # The branch pushed to; null only when nothing was pushed to a new branch (no branch was created).
    branch: str | None = None
    repo_url: str
    # The branch in GitHub's, GitLab's or Bitbucket's web UI; null for other hosts.
    web_url: str | None = None
    # The new commit; null when there was nothing to push.
    commit_sha: str | None = None
    files_changed: int
    nothing_to_push: bool
    # recorded_commit: the commit the project was created from; branch_head: the branch's latest commit.
    based_on: GitPushBase

    @classmethod
    def from_model(cls, result: GitPushResultModel) -> "ProjectPushResponseDTO":
        return cls.model_validate(result.model_dump())
