from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status

from app.core.security import get_current_user, require_permission, resolve_org_id
from app.dto.projectDto import CreateGitProjectDTO, ProjectResponseDTO
from app.dto.projectProfileDto import ProjectProfileResponseDTO
from app.dto.testDataDto import TestDataResponseDTO, TestDataSummaryResponseDTO, UpdateTestDataCasesDTO
from app.models.authModel import CurrentUserModel
from app.models.testDataModel import TestDataStatus
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_PERM
from app.services.projectProfileService import ProjectProfileService
from app.services.projectService import ProjectService
from app.services.testDataService import TestDataService, test_data_processor
from app.utils.TestDataParser import MAX_TEST_DATA_BYTES

UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter(
    prefix="/tx-agents/ng-automate/project",
    tags=["NG Automate - Projects"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)

project_service = ProjectService()
project_profile_service = ProjectProfileService()
test_data_service = TestDataService()


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
    "or JSON in any other shape / plain text (.txt, .md) that an LLM turns into test cases in the background (202, "
    "status processing: poll it). Invalid files are rejected.",
)
async def upload_test_data(
    project_id: str,
    response: Response,
    file: UploadFile = File(description="Test cases (.json, .csv) or a free-form source (.json, .txt, .md)"),
    model_connection_id: str | None = Query(
        None, description="Model that reads a free-form source. Leave empty to use the default model."
    ),
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


@router.post(
    "/{project_id}/test-data/{test_data_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TestDataResponseDTO,
    summary="Read a free-form source again after it failed.",
)
async def retry_test_data(project_id: str, test_data_id: str, org_id: str = Depends(resolve_org_id)) -> TestDataResponseDTO:
    upload = await test_data_service.retry(test_data_id, project_id, org_id)
    await test_data_processor.submit(upload.id)
    return TestDataResponseDTO.from_model(upload)


async def _read_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        yield chunk
