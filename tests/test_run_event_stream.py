import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.config import settings
from app.models.runModel import RunEventType
from app.services.runService import RunEventStreamService, RunService

BASE = "/tx-agents/ng-automate/run"


def _seed_run(mongo: MongoClient, org_id: str, status: str = "completed", events: int = 3) -> str:
    db = mongo[settings.MONGO_DATABASE]
    now = datetime.now(UTC)
    run_id = db.runs.insert_one({
        "org_id": ObjectId(org_id), "project_id": ObjectId(), "created_by": ObjectId(), "model_connection_id": None,
        "status": status, "stage": status if status in ("completed", "failed") else "analyzing", "outputs": {},
        "sandbox_container_id": None, "error": None, "event_seq": events,
        "created_at": now, "updated_at": now, "started_at": now, "finished_at": None,
    }).inserted_id
    for seq in range(1, events + 1):
        db.run_events.insert_one({
            "run_id": run_id, "org_id": ObjectId(org_id), "seq": seq, "type": "agent_action", "level": "info",
            "message": f"step {seq}", "data": {"tool": "glob"}, "created_at": now,
        })
    return str(run_id)


def _frames(body: str) -> list[str]:
    return [frame for frame in body.split("\n\n") if frame]


def test_finished_run_streams_all_events_then_end(
    client: TestClient, mongo: MongoClient, auth_headers: Callable[..., dict[str, str]], org_id: str
) -> None:
    run_id = _seed_run(mongo, org_id)

    with client.stream("GET", f"{BASE}/{run_id}/events/stream", headers=auth_headers(org_id)) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _frames(body)
    assert frames[0] == "retry: 3000"
    assert frames[1].startswith('id: 1\nevent: agent_action\ndata: {"seq":1,')
    assert [f.split("\n")[0] for f in frames[1:4]] == ["id: 1", "id: 2", "id: 3"]
    assert frames[-1] == 'event: end\ndata: {"status": "completed", "error": null}'


def test_resume_from_last_event_id(
    client: TestClient, mongo: MongoClient, auth_headers: Callable[..., dict[str, str]], org_id: str
) -> None:
    run_id = _seed_run(mongo, org_id, events=5)
    headers = {**auth_headers(org_id), "Last-Event-ID": "3"}

    with client.stream("GET", f"{BASE}/{run_id}/events/stream", headers=headers) as response:
        ids = [line for line in "".join(response.iter_text()).splitlines() if line.startswith("id: ")]

    assert ids == ["id: 4", "id: 5"]


def test_other_org_gets_404_not_a_stream(
    client: TestClient, mongo: MongoClient, auth_headers: Callable[..., dict[str, str]], org_id: str
) -> None:
    run_id = _seed_run(mongo, org_id)

    response = client.get(f"{BASE}/{run_id}/events/stream", headers=auth_headers(str(ObjectId())))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


@pytest.mark.anyio
async def test_events_appended_while_following_arrive_live(app_db: MongoClient, org_id: str) -> None:
    run_id = _seed_run(app_db, org_id, status="running", events=1)
    service = RunEventStreamService(poll_seconds=0.05, heartbeat_seconds=0.15)
    run_service = RunService()

    async def write_more() -> None:
        await asyncio.sleep(0.2)
        await run_service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, "late step")
        await asyncio.sleep(0.2)
        await run_service.mark_completed(run_id)

    async def never_disconnected() -> bool:
        return False

    writer = asyncio.create_task(write_more())
    items = [item async for item in service.follow(run_id, org_id, 0, never_disconnected)]
    await writer

    messages = [item.event.message for item in items if item.event]
    assert messages == ["step 1", "late step"]
    assert any(item.is_heartbeat for item in items)
    assert items[-1].run is not None and items[-1].run.status.value == "completed"


@pytest.mark.anyio
async def test_disconnect_stops_following(app_db: MongoClient, org_id: str) -> None:
    run_id = _seed_run(app_db, org_id, status="running", events=0)
    service = RunEventStreamService(poll_seconds=0.01)
    calls = 0

    async def disconnected_after_two_checks() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    items = [item async for item in service.follow(run_id, org_id, 0, disconnected_after_two_checks)]

    assert all(item.run is None for item in items)
    assert calls == 3
