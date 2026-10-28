import asyncio
import shutil
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from starlette.types import Receive, Scope, Send

from app.core.security import get_current_user, require_permission, resolve_org_id
from app.dto.projectDto import (
    CreateGitProjectDTO,
    ProjectDeletedResponseDTO,
    ProjectFileContentResponseDTO,
    ProjectFileTreeResponseDTO,
    ProjectPushResponseDTO,
    ProjectResponseDTO,
    PushProjectDTO,
    SaveProjectFileDTO,
)
from app.dto.projectProfileDto import ProjectProfileResponseDTO
from app.dto.testDataDto import (
    SaveTestDataContentDTO,
    TestDataContentResponseDTO,
    TestDataResponseDTO,
    TestDataSummaryResponseDTO,
    UpdateTestDataCasesDTO,
)
from app.models.authModel import CurrentUserModel
from app.models.projectFileModel import ProjectDownloadModel, ProjectFileRoot
from app.models.testDataModel import TestDataStatus
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_PERM
from app.services.projectProfileService import ProjectProfileService
from app.services.projectService import ProjectFileService, ProjectGitService, ProjectService
from app.services.testDataService import TestDataService, test_data_processor
from app.utils.TestDataParser import (
    ACCEPTED_EXTENSIONS,
    MAX_TEST_DATA_BYTES,
    STRUCTURED_EXTENSIONS,
    TEXT_EXTENSIONS,
)

UPLOAD_CHUNK_BYTES = 1024 * 1024
PROJECT_DELETED = "Project deleted successfully"
MODEL_CONNECTION_QUERY = (
    "QXcel model connection (enabled for NG Automate) that reads a free-form source. Ignored when the server uses its "
    "local model (USE_LOCAL_LLM); required otherwise."
)
# The file types the test-data parser accepts, as the upload documents them (".json, .csv").
STRUCTURED_FILE_TYPES = ", ".join(f".{name}" for name in STRUCTURED_EXTENSIONS)
TEXT_FILE_TYPES = ", ".join(f".{name}" for name in TEXT_EXTENSIONS)
TEST_DATA_FILE_TYPES = ", ".join(f".{name}" for name in ACCEPTED_EXTENSIONS)

router = APIRouter(
    prefix="/tx-agents/ng-automate/project",
    tags=["NG Automate - Projects"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)

project_service = ProjectService()
project_profile_service = ProjectProfileService()
test_data_service = TestDataService()
project_file_service = ProjectFileService()
project_git_service = ProjectGitService()


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
    summary="Create a project from a QXcel git connection: its repo URL, token and branch (latest commit). "
    "A connection without a branch uses main, or the repo's default branch when it has no main.",
)
async def create_git_project(
    dto: CreateGitProjectDTO,
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> ProjectResponseDTO:
    project = await project_service.create_from_git(
        git_connection_id=dto.git_connection_id,
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


@router.delete(
    "/{project_id}",
    response_model=ProjectDeletedResponseDTO,
    summary="Delete a project: it no longer shows anywhere (soft delete: its runs stay in the database) and its files "
    "are removed from the server. 409 while one of its runs is still working on it.",
)
async def delete_project(project_id: str, org_id: str = Depends(resolve_org_id)) -> ProjectDeletedResponseDTO:
    await project_service.delete(project_id, org_id)
    return ProjectDeletedResponseDTO(message=PROJECT_DELETED)


@router.get(
    "/{project_id}/download",
    summary="Download the whole project as it is now (its editable copy, with every change the agents and you made) "
    "as <project name>-<YYYYMMDD-HHMMSS>.zip. Tool folders (.git, node_modules, target, .venv, ...) and test results "
    "(ngauto-results, test-results, playwright-report, allure-results) are left out. Uploaded and git projects alike.",
    response_class=FileResponse,
)
async def download_project(project_id: str, org_id: str = Depends(resolve_org_id)) -> FileResponse:
    return _DownloadResponse(await project_file_service.download(project_id, org_id))


@router.post(
    "/{project_id}/git/push",
    response_model=ProjectPushResponseDTO,
    summary="Push the project to its git repository as one commit by you, with the git connection's current URL and "
    "token. target current: onto the connection's branch at its latest commit (the project's version of a file "
    "always wins; files only the remote has stay). target new_branch: a new branch <connection name>-<UTC time> from "
    "the commit the project was created from (or the branch's latest commit when the remote no longer has it: "
    "based_on says which). Files deleted in the project are deleted; the repository's .gitignore keeps ignored files "
    "out; tool folders and test results are never pushed. 400 NOT_A_GIT_PROJECT for uploaded projects (download "
    "them); 404 CONNECTION_NOT_FOUND when the connection is gone or disabled; 409 PROJECT_BUSY while a run uses the "
    "project; 422 GIT_PUSH_FAILED when git fails (e.g. no write access, the remote rejected the push).",
)
async def push_project(
    project_id: str,
    dto: PushProjectDTO,
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> ProjectPushResponseDTO:
    result = await project_git_service.push(project_id, org_id, current_user, dto.target, dto.message)
    return ProjectPushResponseDTO.from_model(result)


class _DownloadResponse(FileResponse):
    """A zip made for one download (sent with its Content-Length). Its folder is removed when the response is over,
    also when the client went away halfway (a background task would not run then)."""

    def __init__(self, download: ProjectDownloadModel) -> None:
        super().__init__(download.path, filename=download.filename, media_type="application/zip")
        self.folder = download.folder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.to_thread(shutil.rmtree, self.folder, ignore_errors=True)


@router.get(
    "/{project_id}/profile",
    response_model=ProjectProfileResponseDTO,
    summary="Get the project's current profile: the result of its latest completed analysis.",
)
async def get_project_profile(project_id: str, org_id: str = Depends(resolve_org_id)) -> ProjectProfileResponseDTO:
    return ProjectProfileResponseDTO.from_model(await project_profile_service.get_latest(project_id, org_id))


@router.post(
    "/{project_id}/test-data",
    status_code=status.HTTP_201_CREATED,
    response_model=TestDataResponseDTO,
    summary="Upload a test-data source for the test generator: our JSON/CSV test-case format (ready at once, 201), "
    f"or JSON / CSV in any other shape or plain text ({TEXT_FILE_TYPES}) that an LLM turns into test cases in the "
    "background (202, status processing: poll it). Invalid or empty files are rejected.",
)
async def upload_test_data(
    project_id: str,
    response: Response,
    file: UploadFile = File(
        description=f"Test cases ({STRUCTURED_FILE_TYPES}) or a free-form source ({TEST_DATA_FILE_TYPES})"
    ),
    model_connection_id: str | None = Query(None, description=MODEL_CONNECTION_QUERY),
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> TestDataResponseDTO:
    # One byte over the limit is enough to reject the file; never read more.
    content = await file.read(MAX_TEST_DATA_BYTES + 1)
    upload = await test_data_service.create(
        project_id=project_id,
        org_id=org_id,
        user_id=current_user.user_id,
        filename=file.filename or "",
        content=content,
        model_connection_id=model_connection_id,
    )
    if upload.status == TestDataStatus.PROCESSING:
        await test_data_processor.submit(upload.id)
        response.status_code = status.HTTP_202_ACCEPTED
    return TestDataResponseDTO.from_model(upload)


@router.get(
    "/{project_id}/test-data",
    response_model=list[TestDataSummaryResponseDTO],
    summary="List the project's test data, newest first (without the cases).",
)
async def get_test_data_list(
    project_id: str,
    limit: int = Query(100, ge=1, le=500),
    org_id: str = Depends(resolve_org_id),
) -> list[TestDataSummaryResponseDTO]:
    return [TestDataSummaryResponseDTO.from_model(u) for u in await test_data_service.get_all(project_id, org_id, limit)]


@router.get(
    "/{project_id}/test-data/{test_data_id}",
    response_model=TestDataResponseDTO,
    summary="Get one test-data upload of the project, with its cases.",
)
async def get_test_data(project_id: str, test_data_id: str, org_id: str = Depends(resolve_org_id)) -> TestDataResponseDTO:
    return TestDataResponseDTO.from_model(await test_data_service.get(test_data_id, project_id, org_id))


@router.put(
    "/{project_id}/test-data/{test_data_id}",
    response_model=TestDataResponseDTO,
    summary="Replace the test cases with the user's reviewed version (once the source has been read).",
)
async def update_test_data_cases(
    project_id: str, test_data_id: str, dto: UpdateTestDataCasesDTO, org_id: str = Depends(resolve_org_id)
) -> TestDataResponseDTO:
    return TestDataResponseDTO.from_model(await test_data_service.update_cases(test_data_id, project_id, org_id, dto.cases))


@router.get(
    "/{project_id}/test-data/{test_data_id}/content",
    response_model=TestDataContentResponseDTO,
    summary="Read a test-data file's text as it is stored, for the editor (content is null when the file is too large "
    "or not text).",
)
async def get_test_data_content(
    project_id: str, test_data_id: str, org_id: str = Depends(resolve_org_id)
) -> TestDataContentResponseDTO:
    return TestDataContentResponseDTO.from_model(await test_data_service.read_content(test_data_id, project_id, org_id))


@router.put(
    "/{project_id}/test-data/{test_data_id}/content",
    response_model=TestDataResponseDTO,
    summary="Save a test-data file's new text from the editor and read it again: our own JSON/CSV test-case format is "
    "ready at once (200); anything else is read by an LLM in the background (202, status processing: poll it). "
    "Invalid text is rejected (400) and nothing changes; 409 while a run uses the project.",
)
async def save_test_data_content(
    project_id: str,
    test_data_id: str,
    dto: SaveTestDataContentDTO,
    response: Response,
    model_connection_id: str | None = Query(None, description=MODEL_CONNECTION_QUERY),
    org_id: str = Depends(resolve_org_id),
) -> TestDataResponseDTO:
    upload = await test_data_service.replace_content(test_data_id, project_id, org_id, dto.content, model_connection_id)
    if upload.status == TestDataStatus.PROCESSING:
        await test_data_processor.submit(upload.id)
        response.status_code = status.HTTP_202_ACCEPTED
    return TestDataResponseDTO.from_model(upload)


@router.post(
    "/{project_id}/test-data/{test_data_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TestDataResponseDTO,
    summary="Read a free-form source again after it failed.",
)
async def retry_test_data(
    project_id: str,
    test_data_id: str,
    model_connection_id: str | None = Query(None, description=MODEL_CONNECTION_QUERY),
    org_id: str = Depends(resolve_org_id),
) -> TestDataResponseDTO:
    upload = await test_data_service.retry(test_data_id, project_id, org_id, model_connection_id)
    await test_data_processor.submit(upload.id)
    return TestDataResponseDTO.from_model(upload)


async def _read_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        yield chunk


@router.get(
    "/{project_id}/files",
    summary="The editor's file tree: the project's editable files and, beside them (never inside), its test-data "
    "files. busy=true while a run uses the project (its files are read-only until the run has finished).",
)
async def get_project_files(project_id: str, org_id: str = Depends(resolve_org_id)) -> ProjectFileTreeResponseDTO:
    return ProjectFileTreeResponseDTO.from_model(await project_file_service.tree(project_id, org_id))


@router.get(
    "/{project_id}/files/content",
    summary="One file's text. Binary or very large files come back with binary=true and no content: download them.",
)
async def get_project_file(
    project_id: str,
    path: str = Query(min_length=1, max_length=1000, description="Path inside the folder, e.g. pom.xml"),
    root: ProjectFileRoot = Query(ProjectFileRoot.PROJECT, description="project, or test-data"),
    org_id: str = Depends(resolve_org_id),
) -> ProjectFileContentResponseDTO:
    return ProjectFileContentResponseDTO.from_model(await project_file_service.read(project_id, org_id, root, path))


@router.put(
    "/{project_id}/files/content",
    summary="Save a text file of the project (a new path creates it). The next run starts from the saved files. "
    "409 while a run uses the project.",
)
async def save_project_file(
    project_id: str, dto: SaveProjectFileDTO, org_id: str = Depends(resolve_org_id)
) -> ProjectFileContentResponseDTO:
    return ProjectFileContentResponseDTO.from_model(await project_file_service.save(project_id, org_id, dto.path, dto.content))


@router.get(
    "/{project_id}/files/download",
    summary="Download one file of the project or of its test data, as it is.",
    response_class=FileResponse,
)
async def download_project_file(
    project_id: str,
    path: str = Query(min_length=1, max_length=1000),
    root: ProjectFileRoot = Query(ProjectFileRoot.PROJECT),
    org_id: str = Depends(resolve_org_id),
) -> FileResponse:
    target = await project_file_service.file_path(project_id, org_id, root, path)
    return FileResponse(target, filename=target.name)
