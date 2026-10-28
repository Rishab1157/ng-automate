"""Push a git project's editable copy back to its repository, as one commit by the user who asks.

Only projects created from a git connection can be pushed (an uploaded project can be downloaded). The connection is
read again from QXcel on every push, for its current repo URL and token. GitPushService does the git work.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import ErrorCode, ErrorMessages, ProjectBusyError, ValidationError
from app.models.authModel import CurrentUserModel
from app.models.gitConnectionModel import GitCommitAuthorModel, GitPushResultModel, GitPushTarget
from app.models.projectModel import ProjectSource
from app.services.gitConnectionService import GitConnectionService, GitPushService
from app.utils.ArchiveUtils import EXPORT_SKIPPED_DIRS, iter_files

from .ProjectService import ProjectService

if TYPE_CHECKING:
    # The run service imports this package: it is loaded on first use, not at import time.
    from app.services.runService import RunService

logger = logging.getLogger(__name__)

# Neutral: a push holds whatever the project's copy has (generated tests, the healer's fixes, the user's own edits).
DEFAULT_COMMIT_MESSAGE = "NG Automate: project changes"
# The commit's author when the user's token has no name or e-mail.
DEFAULT_AUTHOR_NAME = "NG Automate"
DEFAULT_AUTHOR_EMAIL = "ngautomate@noreply.local"
MAX_AUTHOR_CHARS = 200
# Git refuses these in a name or e-mail, and drops these from their ends.
_IDENT_UNSAFE = re.compile(r"[<>\x00-\x1f\x7f]")
_IDENT_END_CHARS = " .,:;\"\\'"


class ProjectGitService:
    def __init__(self, project_service: ProjectService | None = None, run_service: "RunService | None" = None) -> None:
        self.project_service = project_service or ProjectService()
        self._run_service = run_service
        self.storage = self.project_service.storage
        self.git_connection_service = GitConnectionService()
        self.git_push_service = GitPushService()

    @property
    def run_service(self) -> "RunService":
        if self._run_service is None:
            from app.services.runService import RunService

            self._run_service = RunService()
        return self._run_service

    async def push(
        self,
        project_id: str,
        org_id: str,
        current_user: CurrentUserModel,
        target: GitPushTarget,
        message: str | None = None,
    ) -> GitPushResultModel:
        project = await self.project_service.get(project_id, org_id)
        if project.source != ProjectSource.GIT or project.git is None:
            raise ValidationError(ErrorMessages.NOT_A_GIT_PROJECT, error_code=ErrorCode.NOT_A_GIT_PROJECT)
        if await self.run_service.has_active_run(project.id, org_id):
            raise ProjectBusyError(ErrorMessages.PROJECT_BUSY_PUSH)
        source = await self.git_connection_service.get_clone_source(project.git.connection_id, org_id)
        working = await asyncio.to_thread(self.storage.ensure_working_copy, project.id)
        # An empty project would empty the branch: that is never what a push is for.
        if not await asyncio.to_thread(_has_files, working):
            raise ValidationError(ErrorMessages.PROJECT_NOTHING_TO_PUSH)

        result = await asyncio.to_thread(
            self.git_push_service.push,
            source,
            project_dir=working,
            snapshot=self.storage.archive_path(project.id),
            target=target,
            # A connection without a branch: the branch the project was fetched from.
            branch=source.branch or project.git.branch,
            base_commit=project.git.commit_sha,
            message=(message or "").strip() or DEFAULT_COMMIT_MESSAGE,
            author=commit_author(current_user),
        )
        logger.info(
            "Project %s pushed by user %s (org %s): %s, %d file(s)%s",
            project.id, current_user.user_id, org_id, result.branch or "no new branch", result.files_changed,
            ", nothing to push" if result.nothing_to_push else "",
        )
        return result


def commit_author(user: CurrentUserModel) -> GitCommitAuthorModel:
    """The user from their QXcel token; NG Automate's own name and e-mail for what the token does not give."""
    name = _ident(f"{user.firstname} {user.lastname}")
    email = _ident(user.email)
    return GitCommitAuthorModel(
        name=name or DEFAULT_AUTHOR_NAME,
        email=email if "@" in email and " " not in email else DEFAULT_AUTHOR_EMAIL,
    )


def _ident(value: str) -> str:
    return " ".join(_IDENT_UNSAFE.sub("", value).split()).strip(_IDENT_END_CHARS)[:MAX_AUTHOR_CHARS]


def _has_files(folder: Path) -> bool:
    return next(iter_files(folder, EXPORT_SKIPPED_DIRS), None) is not None
