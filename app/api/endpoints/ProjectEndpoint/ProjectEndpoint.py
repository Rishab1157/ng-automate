from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status

from app.core.security import get_current_user, require_permission, resolve_org_id
from app.dto.projectDto import CreateGitProjectDTO, ProjectResponseDTO
from app.dto.projectProfileDto import ProjectProfileResponseDTO
from app.models.authModel import CurrentUserModel
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_PERM
from app.services.projectProfileService import ProjectProfileService
from app.services.projectService import ProjectService

UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter(
    prefix="/tx-agents/ng-automate/project",
    tags=["NG Automate - Projects"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)

project_service = ProjectService()
project_profile_service = ProjectProfileService()


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectResponseDTO,
    summary="Create a project from an uploaded zip of the project folder.",
)
async def upload_project(
    file: UploadFile = File(description="The project folder as one .zip file"),
    name: str | None = Form(None, min_length=1, max_length=200, description="Defaults to the file name"),
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> ProjectResponseDTO:
    project = await project_service.create_from_upload(
        _read_chunks(file), filename=file.filename, name=name, org_id=org_id, user_id=current_user.user_id
    )
    return ProjectResponseDTO.from_model(project)


@router.post(
    "/git",
    status_code=status.HTTP_201_CREATED,
    response_model=ProjectResponseDTO,
    summary="Create a project from a QXcel git connection (latest commit of one branch).",
)
async def create_git_project(
    dto: CreateGitProjectDTO,
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> ProjectResponseDTO:
    project = await project_service.create_from_git(
        git_connection_id=dto.git_connection_id,
        branch=dto.branch,
        name=dto.name,
        org_id=org_id,
        user_id=current_user.user_id,
    )
    return ProjectResponseDTO.from_model(project)


@router.get(
    "/",
    response_model=list[ProjectResponseDTO],
    summary="List the organization's projects, newest first.",
)
async def get_projects(
    limit: int = Query(100, ge=1, le=500),
    org_id: str = Depends(resolve_org_id),
) -> list[ProjectResponseDTO]:
    return [ProjectResponseDTO.from_model(p) for p in await project_service.get_all(org_id, limit)]


@router.get(
    "/{project_id}",
    response_model=ProjectResponseDTO,
    summary="Get one project of the organization.",
)
async def get_project(project_id: str, org_id: str = Depends(resolve_org_id)) -> ProjectResponseDTO:
    return ProjectResponseDTO.from_model(await project_service.get(project_id, org_id))


@router.get(
    "/{project_id}/profile",
    response_model=ProjectProfileResponseDTO,
    summary="Get the project's current profile: the result of its latest completed analysis.",
)
async def get_project_profile(project_id: str, org_id: str = Depends(resolve_org_id)) -> ProjectProfileResponseDTO:
    return ProjectProfileResponseDTO.from_model(await project_profile_service.get_latest(project_id, org_id))


async def _read_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        yield chunk
