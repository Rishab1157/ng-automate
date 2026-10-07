from fastapi import APIRouter, Depends, Query, status

from app.agents.MasterAgent.RunExecutor import run_executor
from app.core.security import get_current_user, require_permission, resolve_org_id
from app.dto.runDto import CreateRunDTO, RunEventResponseDTO, RunResponseDTO
from app.models.authModel import CurrentUserModel
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_PERM
from app.services.runService import RunService

router = APIRouter(
    prefix="/tx-agents/ng-automate/run",
    tags=["NG Automate - Runs"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)

run_service = RunService()


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunResponseDTO,
    summary="Start analyzing a project. The run continues in the background: poll it and its events.",
)
async def create_run(
    dto: CreateRunDTO,
    model_connection_id: str | None = Query(
        None, description="QXcel model connection enabled for NG Automate. Leave empty to use the default model."
    ),
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> RunResponseDTO:
    run = await run_service.create_run(
        project_id=dto.project_id, org_id=org_id, user_id=current_user.user_id, model_connection_id=model_connection_id
    )
    await run_executor.submit(run.id)
    return RunResponseDTO.from_model(run)


@router.get(
    "/{run_id}",
    response_model=RunResponseDTO,
    summary="Get one run: status, current stage, error and the profile it produced.",
)
async def get_run(run_id: str, org_id: str = Depends(resolve_org_id)) -> RunResponseDTO:
    return RunResponseDTO.from_model(await run_service.get(run_id, org_id))


@router.get(
    "/{run_id}/events",
    response_model=list[RunEventResponseDTO],
    summary="The run's timeline, oldest first. Poll with after_seq set to the last seq you received.",
)
async def get_run_events(
    run_id: str,
    after_seq: int = Query(0, ge=0, description="Only events with a higher seq"),
    limit: int = Query(200, ge=1, le=1000),
    org_id: str = Depends(resolve_org_id),
) -> list[RunEventResponseDTO]:
    events = await run_service.get_events(run_id, org_id, after_seq=after_seq, limit=limit)
    return [RunEventResponseDTO.from_model(event) for event in events]
