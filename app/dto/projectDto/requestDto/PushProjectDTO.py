from pydantic import BaseModel, Field

from app.models.gitConnectionModel import GitPushTarget

MAX_COMMIT_MESSAGE_CHARS = 2000


class PushProjectDTO(BaseModel):
    target: GitPushTarget = Field(
        description="current: the git connection's branch, at its latest commit (the project's version of a file "
        "always wins; files only the remote has stay). new_branch: a new branch named <connection name>-<UTC time>, "
        "starting from the commit the project was created from."
    )
    message: str | None = Field(
        None,
        max_length=MAX_COMMIT_MESSAGE_CHARS,
        description="The commit message. Defaults to 'NG Automate: project changes'.",
    )
