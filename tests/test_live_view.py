"""Live view: tickets for a running test sandbox, and the noVNC files and VNC stream proxied through the API.

The sandbox's websockify is played by a real local HTTP server and a real WebSocket echo server.
"""

import asyncio
import http.server
import socket
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import serve

from app.config import settings
from app.services.sandboxService.SandboxService import LiveViewError
from tests.test_run_service import seed_project

RUN_BASE = "/tx-agents/ng-automate/run"
CONTAINER_ID = "0123456789ab" * 4
AuthHeaders = Callable[..., dict[str, str]]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeSandbox:
    def __init__(self, port: int | None, error: Exception | None = None) -> None:
        self.port = port
        self.error = error
        self.tokens: list[tuple[str, bool]] = []

    def live_view_port(self, container_id: str) -> int | None:
        return self.port

    def add_live_view_token(self, container_id: str, interactive: bool) -> str:
        if self.error is not None:
            raise self.error
        token = f"vnc-token-{len(self.tokens) + 1}"
        self.tokens.append((container_id, interactive))
        return token


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeSandbox]:
    def install(port: int | None, error: Exception | None = None) -> FakeSandbox:
        fake = FakeSandbox(port, error)
        monkeypatch.setattr("app.api.endpoints.LiveViewEndpoint.LiveViewEndpoint.live_view_service._sandbox_service", fake)
        return fake

    return install


def _running_run(mongo: MongoClient, org_id: str, container_id: str | None = CONTAINER_ID) -> str:
    run_id = ObjectId()
    now = datetime.now(UTC)
    mongo[settings.MONGO_DATABASE].runs.insert_one({
        "_id": run_id, "org_id": ObjectId(org_id), "project_id": ObjectId(seed_project(mongo, org_id)),
        "created_by": ObjectId(), "mode": "test", "status": "running", "stage": "running_tests", "outputs": {},
        "sandbox_container_id": container_id, "event_seq": 0, "created_at": now, "updated_at": now,
    })
    return str(run_id)


def _ticket(viewer_url: str) -> str:
    return viewer_url.split("/live/")[1].split("/")[0]


def test_ticket_opens_the_run_s_test_sandbox(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox]
) -> None:
    fake = sandbox(31001)
    run_id = _running_run(mongo, org_id)

    response = client.post(f"{RUN_BASE}/{run_id}/live?mode=interactive", headers=auth_headers(org_id))

    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "interactive"
    url = urlsplit(body["viewer_url"])
    ticket = _ticket(body["viewer_url"])
    assert url.path == f"/tx-agents/ng-automate/live/{ticket}/vnc.html"
    assert parse_qs(url.query)["path"] == [f"tx-agents/ng-automate/live/{ticket}/websockify"]
    assert fake.tokens == [(CONTAINER_ID, True)]
    # Only a hash of the ticket is stored, and the VNC token never reaches the browser.
    doc = mongo[settings.MONGO_DATABASE].live_view_tickets.find_one()
    assert doc["_id"] != ticket and len(doc["_id"]) == 64
    assert "vnc-token" not in response.text


def test_view_only_viewer(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox]
) -> None:
    fake = sandbox(31001)

    response = client.post(f"{RUN_BASE}/{_running_run(mongo, org_id)}/live", headers=auth_headers(org_id))

    assert response.json()["mode"] == "view" and "view_only=1" in response.json()["viewer_url"]
    assert fake.tokens == [(CONTAINER_ID, False)]


@pytest.mark.parametrize("case", ["no_container", "no_display_port", "display_down"])
def test_nothing_to_watch_is_a_conflict(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox],
    case: str,
) -> None:
    sandbox(None if case == "no_display_port" else 31001, LiveViewError("down") if case == "display_down" else None)
    run_id = _running_run(mongo, org_id, None if case == "no_container" else CONTAINER_ID)

    response = client.post(f"{RUN_BASE}/{run_id}/live", headers=auth_headers(org_id))

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "LIVE_VIEW_UNAVAILABLE"


def test_other_orgs_and_tokenless_callers_get_no_ticket(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox]
) -> None:
    sandbox(31001)
    run_id = _running_run(mongo, org_id)

    other = client.post(f"{RUN_BASE}/{run_id}/live", headers=auth_headers(str(ObjectId())))
    anonymous = client.post(f"{RUN_BASE}/{run_id}/live")

    assert (other.status_code, anonymous.status_code) == (404, 401)


# ---------- viewer routes: the proxy ----------


class _NoVncFiles(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/vnc.html":
            body = b"<html><title>noVNC</title></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def novnc_files() -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoVncFiles)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture
def websockify() -> Iterator[tuple[int, list[str]]]:
    """An echo server that records the path (with websockify's token) each connection asked for."""
    port = _free_port()
    paths: list[str] = []
    ready = threading.Event()
    stop: list[Any] = []

    async def echo(connection: Any) -> None:
        paths.append(connection.request.path)
        async for message in connection:
            await connection.send(message)

    def run() -> None:
        async def main() -> None:
            async with serve(echo, "127.0.0.1", port, subprotocols=["binary"]):
                loop_stop = asyncio.get_running_loop().create_future()
                stop.append((asyncio.get_running_loop(), loop_stop))
                ready.set()
                await loop_stop

        asyncio.run(main())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(5)
    yield port, paths
    loop, future = stop[0]
    loop.call_soon_threadsafe(future.set_result, None)
    thread.join(5)


def _viewer(client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str,
            sandbox: Callable[..., FakeSandbox], port: int) -> str:
    sandbox(port)
    response = client.post(f"{RUN_BASE}/{_running_run(mongo, org_id)}/live?mode=interactive", headers=auth_headers(org_id))
    return _ticket(response.json()["viewer_url"])


def test_novnc_files_come_through_with_a_sandboxing_policy(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox],
    novnc_files: int,
) -> None:
    ticket = _viewer(client, mongo, auth_headers, org_id, sandbox, novnc_files)

    page = client.get(f"/tx-agents/ng-automate/live/{ticket}/vnc.html")
    missing = client.get(f"/tx-agents/ng-automate/live/{ticket}/nothing.js")
    traversal = client.get(f"/tx-agents/ng-automate/live/{ticket}/..%2F..%2Fetc%2Fpasswd")
    wrong_ticket = client.get("/tx-agents/ng-automate/live/not-a-ticket/vnc.html")

    assert page.status_code == 200 and b"noVNC" in page.content
    assert page.headers["content-security-policy"].startswith("sandbox")
    assert page.headers["x-content-type-options"] == "nosniff"
    assert [r.status_code for r in (missing, traversal, wrong_ticket)] == [404, 404, 404]


def test_vnc_stream_is_proxied_both_ways_with_the_hidden_token(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox],
    websockify: tuple[int, list[str]],
) -> None:
    port, paths = websockify
    ticket = _viewer(client, mongo, auth_headers, org_id, sandbox, port)

    with client.websocket_connect(f"/tx-agents/ng-automate/live/{ticket}/websockify", subprotocols=["binary"]) as ws:
        ws.send_bytes(b"RFB 003.008\n")
        assert ws.receive_bytes() == b"RFB 003.008\n"

    assert paths == ["/websockify?token=vnc-token-1"]


def test_unknown_ticket_cannot_open_the_stream(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/tx-agents/ng-automate/live/forged-ticket/websockify") as ws:
            ws.receive_bytes()

    assert closed.value.code == 4404


def test_expired_ticket_is_refused(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, sandbox: Callable[..., FakeSandbox],
    novnc_files: int,
) -> None:
    ticket = _viewer(client, mongo, auth_headers, org_id, sandbox, novnc_files)
    mongo[settings.MONGO_DATABASE].live_view_tickets.update_many({}, {"$set": {"expires_at": datetime(2020, 1, 1)}})

    assert client.get(f"/tx-agents/ng-automate/live/{ticket}/vnc.html").status_code == 404
