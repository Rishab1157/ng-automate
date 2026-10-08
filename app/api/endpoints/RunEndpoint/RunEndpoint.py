import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, Header, Query, Request, WebSocket, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.websockets import WebSocketDisconnect

from app.agents.MasterAgent.RunExecutor import run_executor
from app.core.exceptions import NgAutomateException
from app.core.security import decode_access_token, get_current_user, require_permission, resolve_org_id
from app.dto.runDto import (
    CreateRunCommandDTO,
    CreateRunDTO,
    RunCommandResponseDTO,
    RunEventResponseDTO,
    RunResponseDTO,
    TestAttemptResponseDTO,
)
from app.models.authModel import CurrentUserMapper, CurrentUserModel
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_ALL_ORGS, NGAUTOMATE_ACCESS_PERM
from app.services.runCommandService import RunCommandService
from app.services.runService import RunEventStreamService, RunService, RunStreamItem

logger = logging.getLogger(__name__)

# Browsers reconnect after this many milliseconds when the stream drops.
SSE_RETRY_MS = 3000
SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
# WebSocket: the first message must authenticate within this time (browsers cannot send headers on a WebSocket).
WS_AUTH_TIMEOUT_SECONDS = 10
WS_MAX_MESSAGE_CHARS = 10_000
WS_MAX_COMMANDS_PER_MINUTE = 30
WS_CLOSE_UNAUTHENTICATED = 4401
WS_CLOSE_FORBIDDEN = 4403
WS_CLOSE_NOT_FOUND = 4404

router = APIRouter(
    prefix="/tx-agents/ng-automate/run",
    tags=["NG Automate - Runs"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)

# The WebSocket authenticates with its first message instead of a header, so it cannot share the router's
# header-based permission check.
socket_router = APIRouter(prefix="/tx-agents/ng-automate/run", tags=["NG Automate - Runs"])

run_service = RunService()
run_event_stream_service = RunEventStreamService(run_service)
run_command_service = RunCommandService(run_service)


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunResponseDTO,
    summary="Start a run: analyze a project; or also run and heal its tests (mode=test); or write new tests from "
    "test data, then run and heal them (mode=generate). The run continues in the background: follow it with "
    "GET /{run_id} and its events.",
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
        project_id=dto.project_id,
        org_id=org_id,
        user_id=current_user.user_id,
        model_connection_id=model_connection_id,
        mode=dto.mode,
        test_selector=dto.test_selector,
        test_data_id=dto.test_data_id,
        run_scope=dto.run_scope,
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
    "/{run_id}/tests",
    response_model=list[TestAttemptResponseDTO],
    summary="Test mode: every run of the tests so far and the healer's fix after it, with the changed files' diffs.",
)
async def get_run_tests(run_id: str, org_id: str = Depends(resolve_org_id)) -> list[TestAttemptResponseDTO]:
    run = await run_service.get(run_id, org_id)
    return [TestAttemptResponseDTO.from_model(attempt) for attempt in run.outputs.test_attempts or []]


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


@router.get(
    "/{run_id}/events/stream",
    summary="Live timeline (Server-Sent Events): each event as it happens, then `end` when the run is finished.",
    description=(
        "Frames: `id: <seq>` + `event: <type>` + `data: <event JSON>`; a `: ping` comment every 15 s while idle; "
        "finally `event: end` with the run's status and error. To resume after a dropped connection, send "
        "`Last-Event-ID: <last seq>` (browsers do this automatically) or `?after_seq=`."
    ),
    response_class=StreamingResponse,
)
async def stream_run_events(
    run_id: str,
    request: Request,
    after_seq: int = Query(0, ge=0, description="Start after this seq (ignored when Last-Event-ID is sent)"),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    org_id: str = Depends(resolve_org_id),
) -> StreamingResponse:
    # Checked before streaming starts, so an unknown run is a normal 404 response.
    await run_service.get(run_id, org_id)
    start_seq = int(last_event_id) if last_event_id and last_event_id.isdigit() else after_seq
    items = run_event_stream_service.follow(run_id, org_id, start_seq, request.is_disconnected)
    return StreamingResponse(_sse_frames(items), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post(
    "/{run_id}/commands",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunCommandResponseDTO,
    summary="Steer a running run: message (tell the working agent something), pause, resume, or stop. The worker "
    "applies it within about a second; the outcome shows in the run's events. 409 when the run is finished.",
)
async def create_run_command(
    run_id: str,
    dto: CreateRunCommandDTO,
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> RunCommandResponseDTO:
    command = await run_command_service.create(
        run_id=run_id, org_id=org_id, user_id=current_user.user_id, type=dto.type, text=dto.text
    )
    return RunCommandResponseDTO.from_model(command)


@socket_router.websocket("/{run_id}/ws")
async def run_socket(websocket: WebSocket, run_id: str) -> None:
    """Both ways on one connection: the run's events out, the user's commands in.

    1. Send {"type": "auth", "token": "<QXcel token>", "after_seq": 0, "target_org_id": null} within 10 s.
    2. Receive {"type": "ready"}, then {"type": "event", "event": {...}} as the run goes (same as the SSE stream),
       {"type": "ping"} while idle, and finally {"type": "end", "status": ..., "error": ...}.
    3. Send {"type": "command", "command": {"type": "message" | "pause" | "resume" | "stop", "text": "..."}};
       each is answered with {"type": "ack", "command": {...}} or {"type": "error", "message": "..."}.
    Close codes: 4401 not authenticated, 4403 not allowed, 4404 run not found.
    """
    await websocket.accept()
    access = await _authenticate(websocket, run_id)
    if access is None:
        return
    user, org_id, after_seq = access
    await websocket.send_json({"type": "ready", "run_id": run_id})

    disconnected = asyncio.Event()

    async def is_disconnected() -> bool:
        return disconnected.is_set()

    sender = asyncio.create_task(_send_events(websocket, run_id, org_id, after_seq, is_disconnected))
    receiver = asyncio.create_task(_receive_commands(websocket, run_id, org_id, user))
    done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    disconnected.set()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if sender in done:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # the browser already closed it


async def _authenticate(websocket: WebSocket, run_id: str) -> tuple[CurrentUserModel, str, int] | None:
    try:
        message = await asyncio.wait_for(websocket.receive_text(), WS_AUTH_TIMEOUT_SECONDS)
        auth = json.loads(message) if len(message) <= WS_MAX_MESSAGE_CHARS else None
    except (TimeoutError, ValueError, WebSocketDisconnect):
        auth = None
    if not isinstance(auth, dict) or auth.get("type") != "auth":
        await _close(websocket, WS_CLOSE_UNAUTHENTICATED)
        return None
    try:
        user = CurrentUserMapper.from_token_payload(decode_access_token(str(auth.get("token") or "")))
    except NgAutomateException:
        await _close(websocket, WS_CLOSE_UNAUTHENTICATED)
        return None
    org_id = user.org_id
    target = auth.get("target_org_id")
    if target and target != user.org_id:
        if not ObjectId.is_valid(str(target)) or not user.has_permission(NGAUTOMATE_ACCESS_ALL_ORGS):
            await _close(websocket, WS_CLOSE_FORBIDDEN)
            return None
        org_id = str(target)
    if not user.has_any_permission(NGAUTOMATE_ACCESS_PERM):
        await _close(websocket, WS_CLOSE_FORBIDDEN)
        return None
    try:
        await run_service.get(run_id, org_id)
    except NgAutomateException:
        await _close(websocket, WS_CLOSE_NOT_FOUND)
        return None
    after_seq = auth.get("after_seq")
    return user, org_id, after_seq if isinstance(after_seq, int) and after_seq >= 0 else 0


async def _send_events(websocket: WebSocket, run_id: str, org_id: str, after_seq: int, is_disconnected: Any) -> None:
    async for item in run_event_stream_service.follow(run_id, org_id, after_seq, is_disconnected):
        if item.event is not None:
            event = RunEventResponseDTO.from_model(item.event).model_dump(mode="json")
            await websocket.send_json({"type": "event", "event": event})
        elif item.run is not None:
            error = item.run.error.model_dump() if item.run.error else None
            await websocket.send_json({"type": "end", "status": item.run.status.value, "error": error})
            return
        else:
            await websocket.send_json({"type": "ping"})


async def _receive_commands(websocket: WebSocket, run_id: str, org_id: str, user: CurrentUserModel) -> None:
    sent: list[float] = []
    loop = asyncio.get_running_loop()
    while True:
        try:
            message = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        now = loop.time()
        sent[:] = [at for at in sent if now - at < 60]
        try:
            data = json.loads(message) if len(message) <= WS_MAX_MESSAGE_CHARS else None
            if not isinstance(data, dict) or data.get("type") != "command":
                raise ValueError('expected {"type": "command", "command": {...}}')
            if len(sent) >= WS_MAX_COMMANDS_PER_MINUTE:
                raise ValueError("too many commands: wait a minute")
            dto = CreateRunCommandDTO.model_validate(data.get("command"))
            sent.append(now)
            command = await run_command_service.create(
                run_id=run_id, org_id=org_id, user_id=user.user_id, type=dto.type, text=dto.text
            )
        except PydanticValidationError:
            await websocket.send_json({"type": "error", "message": "invalid command: type must be message, pause, "
                                       "resume or stop, and a message needs text"})
        except (ValueError, NgAutomateException) as error:
            reason = error.message if isinstance(error, NgAutomateException) else str(error)
            await websocket.send_json({"type": "error", "message": reason})
        else:
            await websocket.send_json({"type": "ack", "command": RunCommandResponseDTO.from_model(command).model_dump(mode="json")})


async def _close(websocket: WebSocket, code: int) -> None:
    try:
        await websocket.close(code=code)
    except RuntimeError:
        pass


async def _sse_frames(items: AsyncIterator[RunStreamItem]) -> AsyncIterator[str]:
    yield f"retry: {SSE_RETRY_MS}\n\n"
    async for item in items:
        if item.event is not None:
            data = RunEventResponseDTO.from_model(item.event).model_dump_json()
            yield f"id: {item.event.seq}\nevent: {item.event.type.value}\ndata: {data}\n\n"
        elif item.run is not None:
            error = item.run.error.model_dump() if item.run.error else None
            yield f"event: end\ndata: {json.dumps({'status': item.run.status.value, 'error': error})}\n\n"
        else:
            yield ": ping\n\n"
