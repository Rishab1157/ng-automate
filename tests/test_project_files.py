"""The editor's API: the project's editable files beside its test-data files, saving, path safety, the busy lock;
and TXGENIE's test cases as test data (kept as written, ready at once)."""

import json
from collections.abc import Callable

from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.config import settings
from app.utils.TestDataParser import read_test_data
from tests.conftest import make_zip

BASE = "/tx-agents/ng-automate/project"
AuthHeaders = Callable[..., dict[str, str]]
BINARY = bytes(range(256))
PROJECT_ZIP = make_zip({
    "pom.xml": "<project/>",
    "src/test/java/LoginTest.java": "class LoginTest {}",
    "drivers/chromedriver": BINARY,
})
TXGENIE_CASES = {
    "test_cases": [
        {
            "id": "TC001",
            "title": "Valid login",
            "description": "A registered user logs in",
            "preconditions": "User is registered\n- Browser is open",
            "priority": "High",
            "test_type": "Functional",
            "module": "Login",
            "test_data": {"username": "tomsmith"},
            "steps": [
                {"step_number": 2, "action": "Click the Login button", "test_data": "", "expected_result": "The dashboard is shown"},
                {"step_number": 1, "action": "Enter the username and password", "test_data": "tomsmith / secret", "expected_result": ""},
            ],
        }
    ]
}


def _create_project(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(f"{BASE}/upload", headers=headers, files={"file": ("shop.zip", PROJECT_ZIP, "application/zip")})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _names(nodes: list[dict]) -> list[str]:
    return [node["name"] for node in nodes]


# ---------- tree and reading ----------


def test_the_tree_shows_the_project_and_its_test_data_side_by_side(
    client: TestClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)
    client.post(f"{BASE}/{project_id}/test-data", headers=headers,
                files={"file": ("txgenie.json", json.dumps(TXGENIE_CASES).encode(), "application/json")})

    tree = client.get(f"{BASE}/{project_id}/files", headers=headers).json()

    assert _names(tree["project"]) == ["drivers", "src", "pom.xml"]  # folders first
    assert tree["project"][1]["children"][0]["path"] == "src/test"
    assert _names(tree["test_data"]) == ["txgenie.json"]
    assert (tree["truncated"], tree["busy"]) == (False, False)


def test_a_text_file_can_be_read_and_a_binary_one_downloaded(
    client: TestClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)

    text = client.get(f"{BASE}/{project_id}/files/content", headers=headers, params={"path": "pom.xml"}).json()
    binary = client.get(f"{BASE}/{project_id}/files/content", headers=headers, params={"path": "drivers/chromedriver"}).json()
    download = client.get(f"{BASE}/{project_id}/files/download", headers=headers, params={"path": "drivers/chromedriver"})

    assert (text["content"], text["editable"], text["binary"]) == ("<project/>", True, False)
    assert (binary["content"], binary["editable"], binary["binary"], binary["size"]) == (None, False, True, 256)
    assert download.status_code == 200 and download.content == BINARY


def test_test_data_files_are_read_only(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)
    client.post(f"{BASE}/{project_id}/test-data", headers=headers, files={"file": ("cases.txt", b"Open the login page", "text/plain")})

    file = client.get(f"{BASE}/{project_id}/files/content", headers=headers,
                      params={"root": "test-data", "path": "cases.txt"}).json()

    assert (file["content"], file["editable"]) == ("Open the login page", False)


# ---------- saving ----------


def test_saving_changes_the_projects_editable_copy(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)

    saved = client.put(f"{BASE}/{project_id}/files/content", headers=headers,
                       json={"path": "pom.xml", "content": "<project>edited</project>"})
    created = client.put(f"{BASE}/{project_id}/files/content", headers=headers,
                         json={"path": "src/test/java/pages/HomePage.java", "content": "class HomePage {}"})

    assert saved.status_code == 200 and saved.json()["content"] == "<project>edited</project>"
    assert created.status_code == 200
    working = settings.DATA_DIR.resolve() / "projects" / project_id / "project"
    assert (working / "pom.xml").read_text(encoding="utf-8") == "<project>edited</project>"
    assert (working / "src/test/java/pages/HomePage.java").read_text(encoding="utf-8") == "class HomePage {}"


def test_paths_never_leave_the_project(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)

    for path in ("../evil.txt", "src/../../evil.txt", ".git/config", "/etc/passwd", "C:/Windows/evil.txt"):
        response = client.put(f"{BASE}/{project_id}/files/content", headers=headers, json={"path": path, "content": "x"})
        assert response.status_code == 400, path
    for path in ("../source.zip", "../../projects"):
        response = client.get(f"{BASE}/{project_id}/files/content", headers=headers, params={"path": path})
        assert response.status_code == 400, path
    assert client.get(f"{BASE}/{project_id}/files/content", headers=headers, params={"path": "nope.txt"}).status_code == 404


def test_files_cannot_be_saved_while_a_run_uses_the_project(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)
    mongo[settings.MONGO_DATABASE].runs.insert_one(
        {"project_id": ObjectId(project_id), "org_id": ObjectId(org_id), "status": "running"}
    )

    response = client.put(f"{BASE}/{project_id}/files/content", headers=headers, json={"path": "pom.xml", "content": "x"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROJECT_BUSY"
    assert client.get(f"{BASE}/{project_id}/files", headers=headers).json()["busy"] is True


def test_another_organization_sees_nothing(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    project_id = _create_project(client, auth_headers(org_id))
    other = auth_headers(str(ObjectId()))

    assert client.get(f"{BASE}/{project_id}/files", headers=other).status_code == 404
    assert client.get(f"{BASE}/{project_id}/files/download", headers=other, params={"path": "pom.xml"}).status_code == 404
    response = client.put(f"{BASE}/{project_id}/files/content", headers=other, json={"path": "pom.xml", "content": "x"})
    assert response.status_code == 404


# ---------- TXGENIE test cases ----------


def test_txgenie_cases_are_kept_as_written_and_ready_at_once() -> None:
    data_set = read_test_data(json.dumps(TXGENIE_CASES).encode(), "txgenie.json")

    [case] = data_set.cases  # type: ignore[union-attr]
    assert (case.id, case.title, case.description) == ("TC001", "Valid login", "A registered user logs in")
    assert case.preconditions == ["User is registered", "Browser is open"]
    assert [(step.action, step.value, step.expected) for step in case.steps] == [
        ("Enter the username and password", "tomsmith / secret", None),  # step_number order
        ("Click the Login button", None, "The dashboard is shown"),
    ]
    assert case.test_data == {"username": "tomsmith"}
    assert case.tags == ["High", "Functional", "Login"]


def test_a_plain_list_of_txgenie_cases_works_too(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)

    response = client.post(f"{BASE}/{project_id}/test-data", headers=headers,
                           files={"file": ("cases.json", json.dumps(TXGENIE_CASES["test_cases"]).encode(), "application/json")})

    assert response.status_code == 201  # ready at once: no LLM needed
    assert response.json()["status"] == "ready" and response.json()["case_count"] == 1


def test_uploaded_test_data_names_never_clash(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _create_project(client, headers)
    for _ in range(2):
        client.post(f"{BASE}/{project_id}/test-data", headers=headers, files={"file": ("cases.txt", b"Open the page", "text/plain")})

    tree = client.get(f"{BASE}/{project_id}/files", headers=headers).json()

    assert _names(tree["test_data"]) == ["cases (2).txt", "cases.txt"]
