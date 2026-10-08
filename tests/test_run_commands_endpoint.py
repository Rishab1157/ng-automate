"""Steering through the API: POST /run/{id}/commands and the run WebSocket (events out, commands in)."""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_ALL_ORGS
from tests.test_run_service import seed_project

BASE = "/tx-agents/ng-automate/run"
AuthHeaders = Callable[..., dict[str, str]]


def _run(mongo: MongoClient, org_id: str, status: str = "running") -> str:
    run_id = ObjectId()
    now = datetime.now(UTC)
    mongo[settings.MONGO_DATABASE].runs.insert_one({
        "_id": run_id, "org_id": ObjectId(org_id), "project_id": ObjectId(seed_project(mongo, org_id)),
        "created_by": ObjectId(), "mode": "test", "status": status, "stage": "healing", "outputs": {},
        "event_seq": 0, "created_at": now, "updated_at": now,
    })
    return str(run_id)


def _token(headers: dict[str, str]) -> str:
    return headers["Authorization"].removeprefix("Bearer ")


def _events(mongo: MongoClient, run_id: str) -> list[tuple[str, str]]:
    docs = mongo[settings.MONGO_DATABASE].run_events.find({"run_id": ObjectId(run_id)}).sort("seq", 1)
    return [(doc["type"], doc["message"]) for doc in docs]


# ---------- POST /commands ----------

def test_message_is_stored_pending_and_shown_in_the_timeline(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _run(mongo, org_id)

    response = client.post(f"{BASE}/{run_id}/commands", headers=auth_headers(org_id),
                           json={"type": "message", "text": "  use the existing LoginPage  "})

    assert response.status_code == 202
    body = response.json()
    assert (body["type"], body["text"], body["status"]) == ("message", "use the existing LoginPage", "pending")
    assert _events(mongo, run_id) == [("user_message", "You: use the existing LoginPage")]
    doc = mongo[settings.MONGO_DATABASE].run_commands.find_one({"_id": ObjectId(body["id"])})
    assert doc["run_id"] == ObjectId(run_id) and doc["status"] == "pending"


@pytest.mark.parametrize("type", ["pause", "resume", "stop"])
def test_control_commands_ignore_text(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, type: str
) -> None:
    response = client.post(f"{BASE}/{_run(mongo, org_id)}/commands", headers=auth_headers(org_id),
                           json={"type": type, "text": "ignored"})

    assert response.status_code == 202 and response.json()["text"] is None


@pytest.mark.parametrize(
    ("body", "status"),
    [({"type": "message"}, 400), ({"type": "message", "text": "   "}, 400), ({"type": "dance"}, 422),
     ({"type": "message", "text": "x" * 4001}, 422)],
)
def test_bad_commands_are_rejected(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, body: dict, status: int
) -> None:
    response = client.post(f"{BASE}/{_run(mongo, org_id)}/commands", headers=auth_headers(org_id), json=body)

    assert response.status_code == status


def test_finished_or_foreign_runs_take_no_commands(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    finished = _run(mongo, org_id, status="completed")
    running = _run(mongo, org_id)

    late = client.post(f"{BASE}/{finished}/commands", headers=auth_headers(org_id), json={"type": "stop"})
    foreign = client.post(f"{BASE}/{running}/commands", headers=auth_headers(str(ObjectId())), json={"type": "stop"})
    anonymous = client.post(f"{BASE}/{running}/commands", json={"type": "stop"})

    assert (late.status_code, late.json()["error"]["code"]) == (409, "RUN_NOT_ACTIVE")
    assert (foreign.status_code, anonymous.status_code) == (404, 401)


# ---------- WebSocket ----------

def test_socket_needs_a_valid_token_first(client: TestClient, mongo: MongoClient, org_id: str) -> None:
    run_id = _run(mongo, org_id)

    for first_message in ({"type": "hello"}, {"type": "auth", "token": "not-a-jwt"}):
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"{BASE}/{run_id}/ws") as ws:
                ws.send_json(first_message)
                ws.receive_json()
        assert closed.value.code == 4401


def test_socket_refuses_other_orgs_and_unknown_runs(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _run(mongo, org_id)
    cases = [
        (run_id, {"token": _token(auth_headers(str(ObjectId())))}, 4404),  # someone else's run
        (str(ObjectId()), {"token": _token(auth_headers(org_id))}, 4404),  # no such run
        (run_id, {"token": _token(auth_headers(str(ObjectId()))), "target_org_id": org_id}, 4403),  # no all-orgs right
        (run_id, {"token": _token(auth_headers(org_id, permissions=("OTHER:READ",)))}, 4403),  # no NG Automate right
    ]
    for target_run, auth, code in cases:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"{BASE}/{target_run}/ws") as ws:
                ws.send_json({"type": "auth", **auth})
                ws.receive_json()
        assert closed.value.code == code


def test_socket_streams_events_and_takes_commands(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _run(mongo, org_id)

    with client.websocket_connect(f"{BASE}/{run_id}/ws") as ws:
        ws.send_json({"type": "auth", "token": _token(auth_headers(org_id))})
        assert ws.receive_json() == {"type": "ready", "run_id": run_id}
        ws.send_json({"type": "command", "command": {"type": "message", "text": "check the locator first"}})
        frames = [ws.receive_json(), ws.receive_json()]
        ws.send_json({"type": "command", "command": {"type": "dance"}})
        error = ws.receive_json()
        while error["type"] != "error":
            error = ws.receive_json()
        # The run ends: the socket sends the end frame and closes.
        mongo[settings.MONGO_DATABASE].runs.update_one({"_id": ObjectId(run_id)}, {"$set": {"status": "cancelled"}})
        end = ws.receive_json()
        while end["type"] != "end":
            end = ws.receive_json()

    ack = next(frame for frame in frames if frame["type"] == "ack")
    event = next(frame for frame in frames if frame["type"] == "event")
    assert ack["command"]["text"] == "check the locator first" and ack["command"]["status"] == "pending"
    assert event["event"]["type"] == "user_message" and event["event"]["message"] == "You: check the locator first"
    assert "invalid command" in error["message"]
    assert end["status"] == "cancelled"


def test_all_orgs_user_can_watch_another_org_s_run(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _run(mongo, org_id)
    admin = auth_headers(str(ObjectId()), permissions=(NGAUTOMATE_ACCESS_ALL_ORGS,))

    with client.websocket_connect(f"{BASE}/{run_id}/ws") as ws:
        ws.send_json({"type": "auth", "token": _token(admin), "target_org_id": org_id})
        assert ws.receive_json()["type"] == "ready"
