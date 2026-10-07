import os
from collections.abc import Callable

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_ALL_ORGS
from tests.conftest import make_zip

BASE = "/tx-agents/ng-automate/project"
SAMPLE_ZIP = make_zip({"pom.xml": "<project/>", "src/test/java/LoginTest.java": "class LoginTest {}"})

AuthHeaders = Callable[..., dict[str, str]]


def _upload(client: TestClient, headers: dict[str, str], data: bytes = SAMPLE_ZIP, query: str = ""):
    return client.post(f"{BASE}/upload{query}", headers=headers, files={"file": ("demo.zip", data, "application/zip")})


def _error(response) -> dict:
    return response.json()["error"]


# ---------- upload ----------

def test_upload_creates_a_project(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = _upload(client, auth_headers(org_id))

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "demo"
    assert body["source"] == "upload"
    assert body["status"] == "ready"
    assert body["org_id"] == org_id
    assert body["archive"]["file_count"] == 2
    assert body["git"] is None


def test_unsafe_zip_is_rejected(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = _upload(client, auth_headers(org_id), make_zip({"../evil.txt": "x"}))

    assert response.status_code == 400
    assert _error(response)["code"] == "INVALID_ARCHIVE"


def test_upload_over_the_size_limit_is_rejected(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = _upload(client, auth_headers(org_id), os.urandom(6 * 1024 * 1024))  # limit is 5 MB in tests

    assert response.status_code == 413
    assert _error(response)["code"] == "PAYLOAD_TOO_LARGE"


# ---------- read ----------

def test_project_is_visible_only_to_its_org(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    project_id = _upload(client, auth_headers(org_id)).json()["id"]

    own = client.get(f"{BASE}/{project_id}", headers=auth_headers(org_id))
    other = client.get(f"{BASE}/{project_id}", headers=auth_headers(str(ObjectId())))

    assert own.status_code == 200
    assert other.status_code == 404
    assert _error(other)["code"] == "RESOURCE_NOT_FOUND"


def test_list_returns_only_own_projects(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    other_org = str(ObjectId())
    _upload(client, auth_headers(org_id))
    _upload(client, auth_headers(other_org))

    response = client.get(f"{BASE}/", headers=auth_headers(org_id))

    assert response.status_code == 200
    assert [p["org_id"] for p in response.json()] == [org_id]


def test_malformed_project_id_is_not_found(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    assert client.get(f"{BASE}/not-an-id", headers=auth_headers(org_id)).status_code == 404


# ---------- auth ----------

def test_missing_token_is_rejected(client: TestClient) -> None:
    response = client.get(f"{BASE}/")

    assert response.status_code == 401
    assert _error(response)["code"] == "INVALID_CREDENTIALS"


def test_expired_token_is_rejected(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = client.get(f"{BASE}/", headers=auth_headers(org_id, expires_in=-10))

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_user_without_ng_automate_permission_is_forbidden(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = client.get(f"{BASE}/", headers=auth_headers(org_id, permissions=("SYSTEMADMIN-X:READ:ALL_ORGS",)))

    assert response.status_code == 403
    assert _error(response)["code"] == "INSUFFICIENT_PERMISSIONS"


def test_other_org_needs_all_orgs_permission(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    other_org = str(ObjectId())

    denied = client.get(f"{BASE}/?target_org_id={other_org}", headers=auth_headers(org_id))
    allowed = _upload(client, auth_headers(org_id, permissions=(NGAUTOMATE_ACCESS_ALL_ORGS,)), query=f"?target_org_id={other_org}")

    assert denied.status_code == 403
    assert allowed.status_code == 201
    assert allowed.json()["org_id"] == other_org


def test_malformed_target_org_id_is_rejected(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id, permissions=(NGAUTOMATE_ACCESS_ALL_ORGS,))

    assert client.get(f"{BASE}/?target_org_id=abc", headers=headers).status_code == 400


# ---------- git ----------

def _create_from_git(client: TestClient, headers: dict[str, str], connection_id: str, **body: str):
    return client.post(f"{BASE}/git", headers=headers, json={"git_connection_id": connection_id, **body})


def test_unknown_git_connection_is_not_found(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = _create_from_git(client, auth_headers(org_id), str(ObjectId()))

    assert response.status_code == 404
    assert _error(response)["code"] == "CONNECTION_NOT_FOUND"


@pytest.mark.parametrize("variant", ["other_org", "not_enabled_for_ng_automate", "inactive"])
def test_git_connection_must_belong_to_org_and_be_enabled(
    client: TestClient, auth_headers: AuthHeaders, add_git_connection: Callable[..., str], org_id: str, variant: str
) -> None:
    connection_id = add_git_connection(
        org_id=str(ObjectId()) if variant == "other_org" else org_id,
        enabled_for_ng_automate=variant != "not_enabled_for_ng_automate",
        is_active=variant != "inactive",
    )

    assert _create_from_git(client, auth_headers(org_id), connection_id).status_code == 404


@pytest.mark.parametrize("repo_url", ["http://github.com/a/b.git", "file:///etc", "https://user:pass@github.com/a/b.git"])
def test_unsafe_repo_urls_are_refused(
    client: TestClient, auth_headers: AuthHeaders, add_git_connection: Callable[..., str], org_id: str, repo_url: str
) -> None:
    response = _create_from_git(client, auth_headers(org_id), add_git_connection(org_id, repo_url=repo_url))

    assert response.status_code == 422
    assert _error(response)["code"] == "GIT_FETCH_FAILED"


def test_branch_that_looks_like_an_option_is_refused(
    client: TestClient, auth_headers: AuthHeaders, add_git_connection: Callable[..., str], org_id: str
) -> None:
    response = _create_from_git(client, auth_headers(org_id), add_git_connection(org_id), branch="--upload-pack=x")

    assert response.status_code == 422


def test_malformed_git_connection_id_fails_validation(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    response = _create_from_git(client, auth_headers(org_id), "abc")

    assert response.status_code == 422
    assert _error(response)["code"] == "INVALID_INPUT"


@pytest.mark.network
def test_public_repo_becomes_a_project(
    client: TestClient, auth_headers: AuthHeaders, add_git_connection: Callable[..., str], org_id: str
) -> None:
    connection_id = add_git_connection(org_id)

    response = _create_from_git(client, auth_headers(org_id), connection_id)

    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "git"
    assert body["git"]["connection_id"] == connection_id
    assert body["git"]["branch"] == "main"
    assert len(body["git"]["commit_sha"]) == 40
    assert body["archive"]["file_count"] > 0


@pytest.mark.network
def test_missing_branch_is_reported(
    client: TestClient, auth_headers: AuthHeaders, add_git_connection: Callable[..., str], org_id: str
) -> None:
    response = _create_from_git(client, auth_headers(org_id), add_git_connection(org_id), branch="no-such-branch")

    assert response.status_code == 422
    assert "was not found" in _error(response)["message"]
