import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import ErrorMessages
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_ALL_ORGS
from tests.test_run_service import seed_model_connection, seed_profile, seed_project

BASE = "/tx-agents/ng-automate/run"
PROJECT_BASE = "/tx-agents/ng-automate/project"
RUN_FIELDS = {
    "id", "org_id", "project_id", "created_by", "model_connection_id", "mode", "test_selector", "test_data_id", "run_scope", "status", "stage", "error", "profile_id",
    "generation", "test_report",
    "created_at", "updated_at", "started_at", "finished_at",
}
EVENT_FIELDS = {"seq", "type", "level", "message", "data", "created_at"}
PROFILE_FIELDS = {"id", "project_id", "run_id", "fact_sheet", "findings", "llm_model", "created_at"}

AuthHeaders = Callable[..., dict[str, str]]


class FakeRunExecutor:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    async def submit(self, run_id: str) -> None:
        self.submitted.append(run_id)


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch) -> FakeRunExecutor:
    fake = FakeRunExecutor()
    monkeypatch.setattr("app.api.endpoints.RunEndpoint.RunEndpoint.run_executor", fake)
    return fake


def _create_run(client: TestClient, headers: dict[str, str], project_id: str, query: str = ""):
    return client.post(f"{BASE}/{query}", headers=headers, json={"project_id": project_id})


def _error(response) -> dict:
    return response.json()["error"]


def _db(mongo: MongoClient):
    return mongo[settings.MONGO_DATABASE]


# ---------- create ----------

def test_create_run_is_accepted_and_submitted(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, org_id)

    response = _create_run(client, auth_headers(org_id), project_id)

    assert response.status_code == 202
    body = response.json()
    assert set(body) == RUN_FIELDS
    assert (body["org_id"], body["project_id"]) == (org_id, project_id)
    assert (body["status"], body["stage"]) == ("queued", "queued")
    assert body["model_connection_id"] is None
    assert body["error"] is None and body["profile_id"] is None
    assert body["started_at"] is None and body["finished_at"] is None
    assert executor.submitted == [body["id"]]
    assert _db(mongo).runs.count_documents({"_id": ObjectId(body["id"])}) == 1


@pytest.mark.parametrize("query", ["", "?model_connection_id=", "?model_connection_id=undefined", "?model_connection_id=null", "?model_connection_id=None"])
def test_blank_model_connection_is_stored_as_none(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str, query: str
) -> None:
    response = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id), query)

    assert response.status_code == 202
    assert response.json()["model_connection_id"] is None
    assert _db(mongo).runs.find_one({"_id": ObjectId(response.json()["id"])})["model_connection_id"] is None


def test_enabled_model_connection_is_stored(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    connection_id = seed_model_connection(mongo, org_id, api_key="sk-very-secret")

    response = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id), f"?model_connection_id={connection_id}")

    assert response.status_code == 202
    body = response.json()
    assert body["model_connection_id"] == connection_id
    assert "sk-very-secret" not in response.text
    doc = _db(mongo).runs.find_one({"_id": ObjectId(body["id"])})
    assert doc["model_connection_id"] == ObjectId(connection_id)
    assert "sk-very-secret" not in str(doc)
    assert executor.submitted == [body["id"]]


def test_model_connection_is_read_from_the_query_never_a_header(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id) | {"model_connection_id": seed_model_connection(mongo, org_id)}

    response = _create_run(client, headers, seed_project(mongo, org_id))

    assert response.status_code == 202
    assert response.json()["model_connection_id"] is None


def test_model_connection_of_another_org_is_not_found(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    connection_id = seed_model_connection(mongo, str(ObjectId()))

    response = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id), f"?model_connection_id={connection_id}")

    assert response.status_code == 404
    assert _error(response)["code"] == "CONNECTION_NOT_FOUND"
    assert executor.submitted == []
    assert _db(mongo).runs.count_documents({}) == 0


def test_malformed_model_connection_id_is_rejected(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    response = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id), "?model_connection_id=abc")

    assert response.status_code == 400
    assert _error(response)["code"] == "INVALID_INPUT"
    assert executor.submitted == []


def test_project_of_another_org_is_not_found(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    response = _create_run(client, auth_headers(org_id), seed_project(mongo, str(ObjectId())))

    assert response.status_code == 404
    assert _error(response)["code"] == "RESOURCE_NOT_FOUND"
    assert _error(response)["message"] == ErrorMessages.PROJECT_NOT_FOUND
    assert executor.submitted == []
    assert _db(mongo).runs.count_documents({}) == 0


@pytest.mark.parametrize("body", [{"project_id": "abc"}, {}])
def test_invalid_body_fails_validation(
    client: TestClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str, body: dict
) -> None:
    response = client.post(f"{BASE}/", headers=auth_headers(org_id), json=body)

    assert response.status_code == 422
    assert _error(response)["code"] == "INVALID_INPUT"
    assert executor.submitted == []


def test_all_orgs_user_can_start_a_run_in_another_org(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    other_org = str(ObjectId())
    headers = auth_headers(org_id, permissions=(NGAUTOMATE_ACCESS_ALL_ORGS,))

    response = _create_run(client, headers, seed_project(mongo, other_org), f"?target_org_id={other_org}")

    assert response.status_code == 202
    assert response.json()["org_id"] == other_org


# ---------- read a run ----------

def test_run_is_visible_only_to_its_org(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]

    own = client.get(f"{BASE}/{run_id}", headers=auth_headers(org_id))
    other = client.get(f"{BASE}/{run_id}", headers=auth_headers(str(ObjectId())))
    malformed = client.get(f"{BASE}/not-an-id", headers=auth_headers(org_id))

    assert own.status_code == 200
    assert set(own.json()) == RUN_FIELDS
    assert own.json()["id"] == run_id
    assert other.status_code == 404
    assert _error(other)["code"] == "RESOURCE_NOT_FOUND"
    assert _error(other)["message"] == ErrorMessages.RUN_NOT_FOUND
    assert malformed.status_code == 404


def test_run_shows_error_and_profile_but_never_internals(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]
    profile_id = ObjectId()
    _db(mongo).runs.update_one({"_id": ObjectId(run_id)}, {"$set": {
        "status": "failed",
        "stage": "failed",
        "error": {"code": "SANDBOX_FAILED", "message": "The sandbox could not be started: timeout"},
        "outputs.profile_id": profile_id,
        "outputs.project_dir": "/srv/secret-host-folder/project",
        "sandbox_container_id": "c0ffee1234",
    }})

    response = client.get(f"{BASE}/{run_id}", headers=auth_headers(org_id))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == {"code": "SANDBOX_FAILED", "message": "The sandbox could not be started: timeout"}
    assert body["profile_id"] == str(profile_id)
    assert "secret-host-folder" not in response.text
    assert "c0ffee1234" not in response.text


# ---------- test mode ----------

def _attempt_doc(number: int, kind: str, heal: dict | None = None) -> dict:
    return {
        "number": number,
        "result": {
            "command": "mvn -B -ntp test", "exit_code": 0 if kind == "passed" else 1, "duration_seconds": 4.0,
            "total": 1, "passed": 1 if kind == "passed" else 0, "failed": 0, "errors": 0, "skipped": 0,
            "cases": [], "report_files": [], "output_tail": "[INFO] BUILD",
            "classification": {"kind": kind, "healable": kind != "passed", "reason": kind, "evidence": []},
        },
        "heal": heal,
    }


def test_test_run_is_created_with_its_selector(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, org_id)
    profile_id = seed_profile(mongo, org_id, project_id, datetime.now(UTC))

    response = client.post(
        f"{BASE}/", headers=auth_headers(org_id), json={"project_id": project_id, "mode": "test", "test_selector": "@smoke"}
    )

    assert response.status_code == 202
    body = response.json()
    assert (body["mode"], body["test_selector"], body["profile_id"]) == ("test", "@smoke", profile_id)
    assert body["test_report"] is None
    assert executor.submitted == [body["id"]]


@pytest.mark.parametrize(
    ("body", "status"),
    [({"mode": "deploy"}, 422), ({"mode": "test", "test_selector": "x" * 201}, 422), ({"mode": "test", "test_selector": "a;b"}, 400)],
)
def test_bad_test_settings_are_rejected(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str,
    body: dict, status: int,
) -> None:
    response = client.post(f"{BASE}/", headers=auth_headers(org_id), json={"project_id": seed_project(mongo, org_id), **body})

    assert response.status_code == status
    assert _error(response)["code"] == "INVALID_INPUT"
    assert executor.submitted == []


def test_tests_endpoint_lists_every_attempt_with_the_healer_diffs(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]
    heal = {"summary": "Set TestNG to 7.10.2", "violations": [], "reverted_files": [],
            "changes": [{"path": "pom.xml", "change": "modified", "diff": "-99.0.0\n+7.10.2"}]}
    report = {"outcome": "passed", "stop_reason": "passed", "detail": None, "runs": 2, "heals": 1, "total": 1,
              "passed": 1, "failed": 0, "errors": 0, "skipped": 0, "changed_files": ["pom.xml"]}
    _db(mongo).runs.update_one({"_id": ObjectId(run_id)}, {"$set": {
        "outputs.test_attempts": [_attempt_doc(1, "dependency_failure", heal), _attempt_doc(2, "passed")],
        "outputs.test_report": report,
    }})

    tests = client.get(f"{BASE}/{run_id}/tests", headers=auth_headers(org_id))
    run = client.get(f"{BASE}/{run_id}", headers=auth_headers(org_id))
    other = client.get(f"{BASE}/{run_id}/tests", headers=auth_headers(str(ObjectId())))

    assert tests.status_code == 200
    [first, second] = tests.json()
    assert first["result"]["classification"]["kind"] == "dependency_failure"
    assert first["heal"]["changes"][0]["diff"] == "-99.0.0\n+7.10.2"
    assert second["heal"] is None
    assert run.json()["test_report"] == report
    assert other.status_code == 404


def test_tests_endpoint_is_empty_before_the_tests_ran(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]

    response = client.get(f"{BASE}/{run_id}/tests", headers=auth_headers(org_id))

    assert (response.status_code, response.json()) == (200, [])


def _upload_test_data(client: TestClient, headers: dict[str, str], project_id: str) -> str:
    cases = [{"id": "LOGIN-1", "title": "Valid login", "steps": [{"action": "open", "target": "https://shop.example"}]}]
    response = client.post(
        f"{PROJECT_BASE}/{project_id}/test-data", headers=headers,
        files={"file": ("cases.json", json.dumps(cases).encode(), "application/json")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_generate_run_is_created_with_its_test_data(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    test_data_id = _upload_test_data(client, headers, project_id)

    response = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "mode": "generate", "test_data_id": test_data_id})

    assert response.status_code == 202
    body = response.json()
    assert (body["mode"], body["test_data_id"], body["generation"]) == ("generate", test_data_id, None)
    assert executor.submitted == [body["id"]]


def test_generate_run_needs_test_data_of_the_same_project(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    other_data = _upload_test_data(client, headers, seed_project(mongo, org_id))

    missing = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "mode": "generate"})
    other = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "mode": "generate", "test_data_id": other_data})
    malformed = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "mode": "generate", "test_data_id": "x"})

    assert (missing.status_code, _error(missing)["message"]) == (400, ErrorMessages.TEST_DATA_REQUIRED)
    assert (other.status_code, _error(other)["message"]) == (404, ErrorMessages.TEST_DATA_NOT_FOUND)
    assert malformed.status_code == 422
    assert executor.submitted == []


def test_generate_run_needs_at_least_one_case_with_its_locators(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    cases = [{"id": "LOGOUT-1", "title": "Logout", "steps": [{"action": "click"}]}]
    test_data_id = client.post(
        f"{PROJECT_BASE}/{project_id}/test-data", headers=headers,
        files={"file": ("cases.json", json.dumps(cases).encode(), "application/json")},
    ).json()["id"]

    response = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "mode": "generate", "test_data_id": test_data_id})

    assert (response.status_code, _error(response)["message"]) == (400, ErrorMessages.NO_RUNNABLE_TEST_CASES)
    assert executor.submitted == []


def test_test_data_is_ignored_outside_generate_mode(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    test_data_id = _upload_test_data(client, headers, project_id)

    response = client.post(f"{BASE}/", headers=headers, json={"project_id": project_id, "test_data_id": test_data_id})

    assert response.status_code == 202
    assert response.json()["test_data_id"] is None


# ---------- events ----------

def _seed_events(mongo: MongoClient, run_id: str, org_id: str, count: int) -> None:
    start = datetime.now(UTC)
    docs = [
        {
            "run_id": ObjectId(run_id),
            "org_id": ObjectId(org_id),
            "seq": seq,
            "type": "agent_action",
            "level": "info" if seq % 2 else "detail",
            "message": f"event {seq}",
            "data": {"tool": "grep", "n": seq},
            "created_at": start + timedelta(milliseconds=seq),
        }
        for seq in range(1, count + 1)
    ]
    random.shuffle(docs)  # stored order must not matter
    _db(mongo).run_events.insert_many(docs)


def test_events_are_paged_in_seq_order(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]
    _seed_events(mongo, run_id, org_id, 5)
    headers = auth_headers(org_id)

    first = client.get(f"{BASE}/{run_id}/events?limit=2", headers=headers)
    second = client.get(f"{BASE}/{run_id}/events?after_seq=2&limit=2", headers=headers)
    rest = client.get(f"{BASE}/{run_id}/events?after_seq=4", headers=headers)
    nothing_new = client.get(f"{BASE}/{run_id}/events?after_seq=5", headers=headers)

    assert first.status_code == 200
    assert [e["seq"] for e in first.json()] == [1, 2]
    assert set(first.json()[0]) == EVENT_FIELDS
    assert first.json()[0] | {"created_at": None} == {
        "seq": 1, "type": "agent_action", "level": "info", "message": "event 1", "data": {"tool": "grep", "n": 1},
        "created_at": None,
    }
    assert [e["seq"] for e in second.json()] == [3, 4]
    assert [e["seq"] for e in rest.json()] == [5]
    assert nothing_new.json() == []


def test_events_are_isolated_by_org(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]
    _seed_events(mongo, run_id, org_id, 2)
    # An event of this run tagged with another org is never shown.
    _db(mongo).run_events.insert_one({
        "run_id": ObjectId(run_id), "org_id": ObjectId(), "seq": 3, "type": "agent_action", "level": "info",
        "message": "foreign", "data": {}, "created_at": datetime.now(UTC),
    })

    own = client.get(f"{BASE}/{run_id}/events", headers=auth_headers(org_id))
    other = client.get(f"{BASE}/{run_id}/events", headers=auth_headers(str(ObjectId())))

    assert [e["seq"] for e in own.json()] == [1, 2]
    assert other.status_code == 404
    assert _error(other)["message"] == ErrorMessages.RUN_NOT_FOUND


@pytest.mark.parametrize("query", ["limit=0", "limit=1001", "after_seq=-1", "limit=abc"])
def test_event_query_bounds_are_validated(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str, query: str
) -> None:
    run_id = _create_run(client, auth_headers(org_id), seed_project(mongo, org_id)).json()["id"]

    response = client.get(f"{BASE}/{run_id}/events?{query}", headers=auth_headers(org_id))

    assert response.status_code == 422
    assert _error(response)["code"] == "INVALID_INPUT"


# ---------- project profile ----------

def test_profile_is_not_found_until_one_exists(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, org_id)
    url = f"{PROJECT_BASE}/{project_id}/profile"

    before = client.get(url, headers=auth_headers(org_id))
    now = datetime.now(UTC)
    seed_profile(mongo, org_id, project_id, now - timedelta(minutes=5), llm_model="ollama/old")
    newest = seed_profile(mongo, org_id, project_id, now, llm_model="openai/gpt-4o")
    after = client.get(url, headers=auth_headers(org_id))

    assert before.status_code == 404
    assert _error(before)["code"] == "RESOURCE_NOT_FOUND"
    assert _error(before)["message"] == ErrorMessages.PROFILE_NOT_FOUND
    assert after.status_code == 200
    body = after.json()
    assert set(body) == PROFILE_FIELDS
    assert (body["id"], body["project_id"], body["llm_model"]) == (newest, project_id, "openai/gpt-4o")
    assert body["fact_sheet"]["build_tool"] == {"value": "Maven", "source": "code", "evidence": ["pom.xml"], "confidence": "high"}
    assert body["findings"]["test_command"]["value"] == "mvn test"
    assert body["findings"]["important_paths"] == [{"path": "src/test/java/pages", "role": "page objects"}]


def test_profile_of_another_orgs_project_is_not_found(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    other_org = str(ObjectId())
    project_id = seed_project(mongo, other_org)
    seed_profile(mongo, other_org, project_id, datetime.now(UTC))

    response = client.get(f"{PROJECT_BASE}/{project_id}/profile", headers=auth_headers(org_id))

    assert response.status_code == 404
    assert _error(response)["message"] == ErrorMessages.PROJECT_NOT_FOUND


# ---------- auth ----------

@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/"),
        ("get", "/0123456789abcdef01234567"),
        ("get", "/0123456789abcdef01234567/events"),
        ("get", "/0123456789abcdef01234567/tests"),
    ],
)
def test_run_endpoints_need_a_token(client: TestClient, executor: FakeRunExecutor, method: str, path: str) -> None:
    response = client.request(method, f"{BASE}{path}", json={"project_id": str(ObjectId())})

    assert response.status_code == 401
    assert _error(response)["code"] == "INVALID_CREDENTIALS"


def test_run_endpoints_need_the_ng_automate_permission(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id, permissions=("SYSTEMADMIN-X:READ:ALL_ORGS",))

    response = _create_run(client, headers, seed_project(mongo, org_id))

    assert response.status_code == 403
    assert _error(response)["code"] == "INSUFFICIENT_PERMISSIONS"
    assert executor.submitted == []


def test_other_org_needs_all_orgs_permission(
    client: TestClient, mongo: MongoClient, executor: FakeRunExecutor, auth_headers: AuthHeaders, org_id: str
) -> None:
    other_org = str(ObjectId())

    response = _create_run(client, auth_headers(org_id), seed_project(mongo, other_org), f"?target_org_id={other_org}")

    assert response.status_code == 403
    assert executor.submitted == []
