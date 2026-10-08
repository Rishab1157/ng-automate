"""The live view of a run's test sandbox: ask for a ticket (with the QXcel token), then open the viewer with it.

Viewer routes take no Authorization header (browsers cannot send one from an iframe or a WebSocket): the ticket in
the path is the credential. Everything they return comes from an untrusted container, so static files are served
with a sandboxing Content-Security-Policy and only noVNC's own files are reachable (websockify --file-only).
"""

import asyncio
import logging
import re

import httpx
from fastapi import APIRouter, Depends, Query, Response, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from app.core.exceptions import ErrorMessages, NotFoundError
from app.core.security import get_current_user, require_permission, resolve_org_id
from app.dto.liveViewDto import LiveViewResponseDTO
from app.models.authModel import CurrentUserModel
from app.models.liveViewModel import LiveViewMode
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_PERM
from app.services.liveViewService import LiveViewService

logger = logging.getLogger(__name__)

RUN_PREFIX = "/tx-agents/ng-automate/run"
VIEWER_PREFIX = "/tx-agents/ng-automate/live"
MAX_STATIC_BYTES = 5 * 1024 * 1024
# noVNC file paths: no "..", no absolute paths, no query tricks.
_SAFE_PATH = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_\-./]{1,200}$")
_STATIC_HEADERS = {
    # Scripts may run (noVNC needs them) but the page gets no access to our origin's storage or cookies.
    "Content-Security-Policy": "sandbox allow-scripts allow-same-origin; frame-ancestors *",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}
_PASSED_CONTENT_TYPES = ("text/html", "text/css", "application/javascript", "text/javascript", "image/", "font/",
                         "application/json", "audio/")

router = APIRouter(
    prefix=RUN_PREFIX,
    tags=["NG Automate - Live view"],
    dependencies=[Depends(require_permission(NGAUTOMATE_ACCESS_PERM))],
)
viewer_router = APIRouter(prefix=VIEWER_PREFIX, tags=["NG Automate - Live view"])

live_view_service = LiveViewService()


@router.post(
    "/{run_id}/live",
    status_code=status.HTTP_201_CREATED,
    response_model=LiveViewResponseDTO,
    summary="Watch (mode=view) or take over (mode=interactive) the browser the run's tests drive, live. Returns a "
    "viewer URL with a short-lived ticket; 409 when the run has no test sandbox running.",
)
async def create_live_view(
    run_id: str,
    mode: LiveViewMode = Query(LiveViewMode.VIEW),
    org_id: str = Depends(resolve_org_id),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> LiveViewResponseDTO:
    ticket, expires_at = await live_view_service.create_ticket(
        run_id=run_id, org_id=org_id, user_id=current_user.user_id, mode=mode
    )
    socket_path = f"{VIEWER_PREFIX.lstrip('/')}/{ticket}/websockify"
    viewer_url = f"{VIEWER_PREFIX}/{ticket}/vnc.html?path={socket_path}&autoconnect=1&resize=scale&reconnect=1"
    if mode == LiveViewMode.VIEW:
        viewer_url += "&view_only=1"
    return LiveViewResponseDTO(viewer_url=viewer_url, mode=mode, expires_at=expires_at)


@viewer_router.websocket("/{ticket}/websockify")
async def live_view_socket(websocket: WebSocket, ticket: str) -> None:
    """The VNC stream between the browser (noVNC) and the sandbox's websockify, byte for byte."""
    target = await live_view_service.resolve(ticket)
    if target is None:
        await websocket.close(code=4404)
        return
    wants_binary = "binary" in websocket.scope.get("subprotocols", [])
    upstream_url = f"ws://127.0.0.1:{target.host_port}/websockify?token={target.vnc_token}"
    try:
        async with connect(upstream_url, subprotocols=["binary"], max_size=None, open_timeout=10) as upstream:
            await websocket.accept(subprotocol="binary" if wants_binary else None)
            await _pump(websocket, upstream)
    except (OSError, TimeoutError, ConnectionClosed) as error:
        logger.info("Live view of run %s closed: %s", target.run_id, type(error).__name__)
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass  # already closed


@viewer_router.get("/{ticket}/{path:path}", summary="noVNC's files, from the sandbox (ticket required).")
async def live_view_file(ticket: str, path: str) -> Response:
    target = await live_view_service.resolve(ticket)
    if target is None or not _SAFE_PATH.match(path):
        raise NotFoundError(ErrorMessages.LIVE_VIEW_NOT_FOUND)
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            upstream = await client.get(f"http://127.0.0.1:{target.host_port}/{path}")
        except httpx.HTTPError:
            raise NotFoundError(ErrorMessages.LIVE_VIEW_NOT_FOUND) from None
    content_type = upstream.headers.get("content-type", "application/octet-stream")
    if upstream.status_code != 200 or len(upstream.content) > MAX_STATIC_BYTES or not content_type.startswith(
        _PASSED_CONTENT_TYPES
    ):
        raise NotFoundError(ErrorMessages.LIVE_VIEW_NOT_FOUND)
    return Response(content=upstream.content, media_type=content_type, headers=_STATIC_HEADERS)


async def _pump(browser: WebSocket, upstream: object) -> None:
    async def to_upstream() -> None:
        while True:
            message = await browser.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes") if message.get("bytes") is not None else message.get("text")
            if data is not None:
                await upstream.send(data)  # type: ignore[attr-defined]

    async def to_browser() -> None:
        async for data in upstream:  # type: ignore[attr-defined]
            if isinstance(data, bytes):
                await browser.send_bytes(data)
            else:
                await browser.send_text(data)

    tasks = [asyncio.create_task(to_upstream()), asyncio.create_task(to_browser())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await upstream.close()  # type: ignore[attr-defined]
        try:
            await browser.close()
        except RuntimeError:
            pass
