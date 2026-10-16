"""Test-data upload and reads: POST/GET /project/{project_id}/test-data."""

import json
from collections.abc import Callable

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import ErrorMessages
from app.utils.TestDataParser import MAX_TEST_DATA_BYTES
from tests.test_run_service import seed_model_connection, seed_project

BASE = "/tx-agents/ng-automate/project"
AuthHeaders = Callable[..., dict[str, str]]

CASES = [
    {
        "id": "LOGIN-1",
        "title": "Valid login",
        "steps": [
            {"action": "open", "target": "https://shop.example/login"},
            {"action": "type", "target": "id=username", "value": "admin"},
            {"action": "click", "target": "css=button[type=submit]"},
            {"action": "verify_text", "target": "css=h1", "expected": "Dashboard"},
        ],
        "tags": ["smoke"],
    },
    {
        "id": "LOGIN-2",
        "title": "Logout",
        "steps": [{"action": "click"}],
    },
]


def _upload(client: TestClient, headers: dict[str, str], project_id: str, content: bytes, filename: str = "cases.json"):
    return client.post(f"{BASE}/{project_id}/test-data", headers=headers, files={"file": (filename, content, "application/json")})


def test_upload_stores_the_parsed_cases_and_warns_about_missing_locators(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, org_id)

    response = _upload(client, auth_headers(org_id), project_id, json.dumps(CASES).encode())

    assert response.status_code == 201
    body = response.json()
    assert (body["project_id"], body["filename"], body["source_format"], body["case_count"]) == (project_id, "cases.json", "json", 2)
    assert body["warnings"] == ["case LOGIN-2 step 1: click needs a locator"]
    assert body["status"] == "ready" and body["error"] is None
    assert body["cases"][0]["steps"][1] == {
        "action": "type", "target": "id=username", "target_hint": None, "alternate_locators": [],
        "value": "admin", "expected": None, "text": None, "needs_locator": False,
    }
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(body["id"])})
    assert doc["org_id"] == ObjectId(org_id) and doc["case_count"] == 2


def test_csv_upload(client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str) -> None:
    csv = (
        "case_id,title,action,target,value,expected,tags,description\n"
        "C1,Search,open,https://shop.example,,,,\n"
        "C1,Search,type,id=q,shoes,,,\n"
    ).encode()

    response = _upload(client, auth_headers(org_id), seed_project(mongo, org_id), csv, "cases.csv")

    assert response.status_code == 201
    assert (response.json()["source_format"], response.json()["case_count"]) == ("csv", 1)


def test_invalid_files_are_rejected_and_nothing_is_stored(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, org_id)
    headers = auth_headers(org_id)

    not_json = _upload(client, headers, project_id, b"{not json")
    wrong_type = _upload(client, headers, project_id, b"hello", "cases.pdf")
    too_big = _upload(client, headers, project_id, b" " * (MAX_TEST_DATA_BYTES + 10))

    assert [r.status_code for r in (not_json, wrong_type, too_big)] == [400, 400, 400]
    assert not_json.json()["error"]["message"].startswith("Test data is not valid JSON")
    assert mongo[settings.MONGO_DATABASE].test_data.count_documents({}) == 0


def test_project_of_another_org_is_not_found(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    project_id = seed_project(mongo, str(ObjectId()))

    response = _upload(client, auth_headers(org_id), project_id, json.dumps(CASES).encode())

    assert response.status_code == 404
    assert response.json()["error"]["message"] == ErrorMessages.PROJECT_NOT_FOUND


def test_list_and_get_are_scoped_to_the_project_and_org(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    other_project = seed_project(mongo, org_id)
    first = _upload(client, headers, project_id, json.dumps(CASES[:1]).encode(), "a.json").json()
    second = _upload(client, headers, project_id, json.dumps(CASES).encode(), "b.json").json()

    listed = client.get(f"{BASE}/{project_id}/test-data", headers=headers)
    one = client.get(f"{BASE}/{project_id}/test-data/{first['id']}", headers=headers)
    wrong_project = client.get(f"{BASE}/{other_project}/test-data/{first['id']}", headers=headers)
    other_org = client.get(f"{BASE}/{project_id}/test-data/{first['id']}", headers=auth_headers(str(ObjectId())))
    malformed = client.get(f"{BASE}/{project_id}/test-data/not-an-id", headers=headers)

    assert [item["id"] for item in listed.json()] == [second["id"], first["id"]]
    assert "cases" not in listed.json()[0]
    assert one.status_code == 200 and one.json()["cases"][0]["id"] == "LOGIN-1"
    assert [r.status_code for r in (wrong_project, other_org, malformed)] == [404, 404, 404]
    assert wrong_project.json()["error"]["message"] == ErrorMessages.TEST_DATA_NOT_FOUND


def test_test_data_endpoints_need_a_token(client: TestClient) -> None:
    project_id = str(ObjectId())

    responses = [
        client.post(f"{BASE}/{project_id}/test-data", files={"file": ("a.json", b"[]", "application/json")}),
        client.get(f"{BASE}/{project_id}/test-data"),
        client.get(f"{BASE}/{project_id}/test-data/{ObjectId()}"),
        client.get(f"{BASE}/{project_id}/test-data/{ObjectId()}/content"),
        client.put(f"{BASE}/{project_id}/test-data/{ObjectId()}/content", json={"content": "[]"}),
    ]

    assert [r.status_code for r in responses] == [401, 401, 401, 401, 401]


class FakeProcessor:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    async def submit(self, test_data_id: str) -> None:
        self.submitted.append(test_data_id)


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> FakeProcessor:
    fake = FakeProcessor()
    monkeypatch.setattr("app.api.endpoints.ProjectEndpoint.ProjectEndpoint.test_data_processor", fake)
    return fake


TEXT_SOURCE = b"TC-101: Valid login\n1. Open https://shop.example\n2. Click the Login button\n"


def test_text_source_is_accepted_and_read_in_the_background(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    project_id = seed_project(mongo, org_id)

    response = _upload(client, auth_headers(org_id), project_id, TEXT_SOURCE, "login-cases.txt")

    assert response.status_code == 202
    body = response.json()
    assert (body["status"], body["source_format"], body["case_count"], body["cases"]) == ("processing", "text", 0, [])
    assert processor.submitted == [body["id"]]
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(body["id"])})
    assert doc["raw_source"].startswith("TC-101") and doc["model_connection_id"] is None


def test_json_in_another_shape_is_a_free_form_source(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    export = {"suite": {"name": "Login", "tests": [{"name": "Valid login", "description": "open and log in"}]}}

    response = _upload(client, auth_headers(org_id), seed_project(mongo, org_id), json.dumps(export).encode(), "x.json")

    assert response.status_code == 202
    assert (response.json()["status"], response.json()["source_format"]) == ("processing", "json")


def test_cases_can_be_reviewed_and_replaced(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    test_data_id = _upload(client, headers, project_id, json.dumps(CASES).encode()).json()["id"]
    edited = [{"id": "LOGIN-2", "title": "Logout", "steps": [{"action": "click", "target_hint": "Logout link"}]}]

    response = client.put(f"{BASE}/{project_id}/test-data/{test_data_id}", headers=headers, json={"cases": edited})

    assert response.status_code == 200
    body = response.json()
    assert body["case_count"] == 1 and body["warnings"] == []
    assert body["cases"][0]["steps"][0]["target_hint"] == "Logout link"
    assert body["cases"][0]["steps"][0]["needs_locator"] is False


def test_processing_source_cannot_be_edited_and_only_failed_ones_are_retried(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    test_data_id = _upload(client, headers, project_id, TEXT_SOURCE, "a.txt").json()["id"]
    cases = [{"id": "A", "title": "a", "steps": [{"action": "open", "target": "https://x"}]}]

    edit = client.put(f"{BASE}/{project_id}/test-data/{test_data_id}", headers=headers, json={"cases": cases})
    early_retry = client.post(f"{BASE}/{project_id}/test-data/{test_data_id}/retry", headers=headers)
    mongo[settings.MONGO_DATABASE].test_data.update_one({"_id": ObjectId(test_data_id)}, {"$set": {"status": "failed"}})
    retry = client.post(f"{BASE}/{project_id}/test-data/{test_data_id}/retry", headers=headers)

    assert edit.status_code == 400 and "not ready" in edit.json()["error"]["message"]
    assert early_retry.status_code == 400
    assert retry.status_code == 202 and retry.json()["status"] == "processing"
    assert processor.submitted == [test_data_id, test_data_id]


# ---------- the editor: a test-data file's text, saved and read again ----------

def _content(client: TestClient, headers: dict[str, str], project_id: str, test_data_id: str):
    return client.get(f"{BASE}/{project_id}/test-data/{test_data_id}/content", headers=headers)


def _save(client: TestClient, headers: dict[str, str], project_id: str, test_data_id: str, text: str):
    return client.put(f"{BASE}/{project_id}/test-data/{test_data_id}/content", headers=headers, json={"content": text})


def test_the_editor_gets_the_file_as_it_was_uploaded(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    text = json.dumps({"source": "txgenie", "test_cases": CASES}, indent=2)
    test_data_id = _upload(client, headers, project_id, text.encode()).json()["id"]

    response = _content(client, headers, project_id, test_data_id)

    assert response.status_code == 200
    assert response.json() == {
        "test_data_id": test_data_id, "filename": "cases.json", "content": text, "editable": True,
        "size_bytes": len(text.encode()), "reason": None,
    }


def test_saving_our_format_replaces_the_file_and_its_cases_at_once(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    test_data_id = _upload(client, headers, project_id, json.dumps(CASES).encode()).json()["id"]
    edited = json.dumps([CASES[0]], indent=2)

    response = _save(client, headers, project_id, test_data_id, edited)

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["case_count"], body["warnings"], body["filename"]) == ("ready", 1, [], "cases.json")
    assert _content(client, headers, project_id, test_data_id).json()["content"] == edited
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(test_data_id)})
    assert doc["revision"] == 1 and doc["raw_source"] is None
    assert processor.submitted == []


def test_saved_free_text_is_read_again_in_the_background(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    test_data_id = _upload(client, headers, project_id, json.dumps(CASES).encode()).json()["id"]
    export = json.dumps({"suite": {"tests": [{"name": "Valid login", "steps": "open the shop and log in"}]}})

    response = _save(client, headers, project_id, test_data_id, export)

    assert response.status_code == 202
    assert (response.json()["status"], response.json()["case_count"], response.json()["cases"]) == ("processing", 0, [])
    assert processor.submitted == [test_data_id]
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(test_data_id)})
    assert doc["raw_source"] == json.dumps(json.loads(export), indent=2) and doc["data_set"] is None


def test_invalid_text_is_rejected_and_nothing_changes(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    original = json.dumps(CASES)
    test_data_id = _upload(client, headers, project_id, original.encode()).json()["id"]

    responses = [_save(client, headers, project_id, test_data_id, text) for text in ("   ", "[]")]

    assert [r.status_code for r in responses] == [400, 400]
    assert _content(client, headers, project_id, test_data_id).json()["content"] == original
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(test_data_id)})
    assert doc["case_count"] == 2 and doc.get("revision", 0) == 0


def test_the_file_cannot_be_saved_while_a_run_uses_the_project(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    original = json.dumps(CASES)
    test_data_id = _upload(client, headers, project_id, original.encode()).json()["id"]
    mongo[settings.MONGO_DATABASE].runs.insert_one(
        {"project_id": ObjectId(project_id), "org_id": ObjectId(org_id), "status": "running"}
    )

    response = _save(client, headers, project_id, test_data_id, json.dumps([CASES[0]]))

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROJECT_BUSY"
    assert _content(client, headers, project_id, test_data_id).json()["content"] == original


def test_a_file_too_large_to_edit_comes_without_its_text(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    test_data_id = _upload(client, headers, project_id, json.dumps(CASES).encode()).json()["id"]
    stored = settings.DATA_DIR / "projects" / project_id / "test-data" / "cases.json"
    stored.write_bytes(b" " * (1024 * 1024 + 1))

    body = _content(client, headers, project_id, test_data_id).json()

    assert (body["content"], body["editable"], body["size_bytes"]) == (None, False, 1024 * 1024 + 1)
    assert "too large" in body["reason"]


@pytest.mark.usefixtures("model_connections_only")
def test_without_the_local_model_free_text_needs_a_model_connection_and_our_format_does_not(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)

    text = _upload(client, headers, project_id, TEXT_SOURCE, "a.txt")
    ours = _upload(client, headers, project_id, json.dumps(CASES).encode())
    as_text = _save(client, headers, project_id, ours.json()["id"], json.dumps({"suite": "login"}))

    assert text.status_code == 400 and text.json()["error"]["code"] == "MODEL_CONNECTION_REQUIRED"
    assert ours.status_code == 201
    assert as_text.status_code == 400 and as_text.json()["error"]["code"] == "MODEL_CONNECTION_REQUIRED"
    assert processor.submitted == []


@pytest.mark.usefixtures("model_connections_only")
def test_a_retry_can_use_another_model_connection(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str, processor: FakeProcessor
) -> None:
    headers, project_id = auth_headers(org_id), seed_project(mongo, org_id)
    first, second = seed_model_connection(mongo, org_id), seed_model_connection(mongo, org_id)
    upload = client.post(
        f"{BASE}/{project_id}/test-data?model_connection_id={first}",
        headers=headers,
        files={"file": ("a.txt", TEXT_SOURCE, "text/plain")},
    )
    test_data_id = upload.json()["id"]
    mongo[settings.MONGO_DATABASE].test_data.update_one({"_id": ObjectId(test_data_id)}, {"$set": {"status": "failed"}})

    retry = client.post(f"{BASE}/{project_id}/test-data/{test_data_id}/retry?model_connection_id={second}", headers=headers)

    assert upload.status_code == 202 and retry.status_code == 202
    doc = mongo[settings.MONGO_DATABASE].test_data.find_one({"_id": ObjectId(test_data_id)})
    assert doc["model_connection_id"] == ObjectId(second)
