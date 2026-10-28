"""GET /project/{project_id}/download: the whole project as a zip."""

import io
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.api.endpoints.ProjectEndpoint.ProjectEndpoint import _DownloadResponse
from app.config import settings
from app.models.projectFileModel import ProjectDownloadModel
from app.models.projectModel import ArchiveModel, GitOriginModel, ProjectMapper, ProjectSource
from app.services.projectService.ProjectFileService import download_name
from tests.conftest import make_zip

BASE = "/tx-agents/ng-automate/project"
AuthHeaders = Callable[..., dict[str, str]]
PROJECT_FILES = {
    "pom.xml": "<project/>",
    "src/test/java/LoginTest.java": "class LoginTest {}",
    "src/test/java/My Test ünï.java": "class MyTest {}",
    "docs/日本語.md": "テスト",
    "README.md": "# demo",
}


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The server's temporary folder, to see that downloads leave nothing behind."""
    root = tmp_path / "server-temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _upload(client: TestClient, headers: dict[str, str], name: str = "demo") -> str:
    response = client.post(
        f"{BASE}/upload", headers=headers, data={"name": name},
        files={"file": ("demo.zip", make_zip(PROJECT_FILES), "application/zip")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _working_copy(project_id: str) -> Path:
    return Path(settings.DATA_DIR) / "projects" / project_id / "project"


def _write(folder: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)


def _names(content: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return set(archive.namelist())


def test_download_is_the_editable_copy_without_tool_folders_and_results(
    client: TestClient, auth_headers: AuthHeaders, org_id: str, temp_root: Path
) -> None:
    headers = auth_headers(org_id)
    project_id = _upload(client, headers)
    working = _working_copy(project_id)
    _write(working, {
        "src/test/java/CartTest.java": "class CartTest {}",  # an agent's new test
        "ngauto-results/junit.xml": "<testsuite/>",
        "test-results/trace.zip": b"PK",
        "playwright-report/index.html": "<html/>",
        "allure-results/x.json": "{}",
        "target/surefire-reports/TEST-x.xml": "<testsuite/>",
        "node_modules/left-pad/index.js": "module.exports = 1",
        ".git/config": "[core]",
        "sub/.venv/lib.py": "",
    })
    (working / "README.md").unlink()  # deleted in the project

    response = client.get(f"{BASE}/{project_id}/download", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert int(response.headers["content-length"]) == len(response.content)  # the UI shows a percentage
    assert _names(response.content) == (set(PROJECT_FILES) - {"README.md"}) | {"src/test/java/CartTest.java"}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("docs/日本語.md").decode() == "テスト"
    assert list(temp_root.iterdir()) == []  # the temporary zip is gone once sent


def test_download_name_is_the_project_name_and_the_time(
    client: TestClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    headers = auth_headers(org_id)
    project_id = _upload(client, headers, name="Shop tests")
    before = datetime.now(UTC).replace(microsecond=0)

    response = client.get(f"{BASE}/{project_id}/download", headers=headers)

    disposition = response.headers["content-disposition"]
    match = re.fullmatch(r"attachment; filename\*=utf-8''Shop%20tests-(\d{8}-\d{6})\.zip", disposition)
    assert match, disposition
    stamp = datetime.strptime(match[1], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    assert before <= stamp <= datetime.now(UTC)


@pytest.mark.parametrize(
    ("project_name", "expected"),
    [
        ("demo", "demo-20261028-120000.zip"),
        ('a/b\\c:d*e?f"g<h>i|j', "a_b_c_d_e_f_g_h_i_j-20261028-120000.zip"),
        ("  ..Ünïcode tests..  ", "Ünïcode tests-20261028-120000.zip"),
        ("...", "project-20261028-120000.zip"),
        ("x" * 300, f"{'x' * 100}-20261028-120000.zip"),
    ],
)
def test_download_name_is_safe_on_every_system(project_name: str, expected: str) -> None:
    assert download_name(project_name, datetime(2026, 10, 28, 12, 0, tzinfo=UTC)) == expected


def test_download_is_scoped_to_the_organization(
    client: TestClient, auth_headers: AuthHeaders, org_id: str, temp_root: Path
) -> None:
    project_id = _upload(client, auth_headers(org_id))

    other_org = client.get(f"{BASE}/{project_id}/download", headers=auth_headers(str(ObjectId())))
    unknown = client.get(f"{BASE}/{ObjectId()}/download", headers=auth_headers(org_id))
    no_token = client.get(f"{BASE}/{project_id}/download")

    assert [other_org.status_code, unknown.status_code, no_token.status_code] == [404, 404, 401]
    assert list(temp_root.iterdir()) == []


def test_ui_origin_can_read_the_file_name(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    project_id = _upload(client, auth_headers(org_id))

    response = client.get(
        f"{BASE}/{project_id}/download", headers={**auth_headers(org_id), "Origin": "https://qxcel.example.com"}
    )

    assert response.status_code == 200
    exposed = [name.strip().lower() for name in response.headers["access-control-expose-headers"].split(",")]
    assert "content-disposition" in exposed


def test_git_project_response_keeps_its_connection_id(
    client: TestClient, mongo: MongoClient, auth_headers: AuthHeaders, org_id: str
) -> None:
    """The UI reads git.connection_id to offer the push."""
    connection_id = str(ObjectId())
    db_model = ProjectMapper.to_create_db_model(
        project_id=ObjectId(), org_id=org_id, user_id=str(ObjectId()), name="shop", source=ProjectSource.GIT,
        archive=ArchiveModel(size_bytes=1, sha256="0" * 64, file_count=1),
        git=GitOriginModel(connection_id=connection_id, repo_url="https://github.com/a/b.git", branch="main", commit_sha="a" * 40),
    )
    mongo[settings.MONGO_DATABASE].projects.insert_one(db_model.model_dump(by_alias=True))

    response = client.get(f"{BASE}/{db_model.id}", headers=auth_headers(org_id))

    assert response.status_code == 200
    assert response.json()["git"]["connection_id"] == connection_id


# ---------- break it ----------

def test_empty_project_downloads_as_an_empty_zip(
    client: TestClient, auth_headers: AuthHeaders, org_id: str, temp_root: Path
) -> None:
    headers = auth_headers(org_id)
    project_id = _upload(client, headers)
    working = _working_copy(project_id)
    for child in working.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    response = client.get(f"{BASE}/{project_id}/download", headers=headers)

    assert response.status_code == 200
    assert _names(response.content) == set()
    assert list(temp_root.iterdir()) == []


def test_very_large_file_is_downloaded_whole(client: TestClient, auth_headers: AuthHeaders, org_id: str) -> None:
    headers = auth_headers(org_id)
    project_id = _upload(client, headers)
    big = os.urandom(8 * 1024 * 1024)  # over the 5 MB project limit these tests use: agents can grow a project
    _write(_working_copy(project_id), {"data/big.bin": big})

    response = client.get(f"{BASE}/{project_id}/download", headers=headers)

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("data/big.bin") == big


def test_links_in_the_project_are_never_followed(
    client: TestClient, auth_headers: AuthHeaders, org_id: str, tmp_path: Path
) -> None:
    headers = auth_headers(org_id)
    project_id = _upload(client, headers)
    outside = tmp_path / "server-secrets"
    _write(outside, {"secret.txt": "never in a download"})
    if not _make_dir_link(_working_copy(project_id) / "linked", outside):
        pytest.skip("this machine cannot create links")

    response = client.get(f"{BASE}/{project_id}/download", headers=headers)

    assert response.status_code == 200
    assert not any(name.startswith("linked") for name in _names(response.content))
    assert (outside / "secret.txt").is_file()


@pytest.mark.anyio
async def test_temporary_zip_is_removed_when_the_client_goes_away(tmp_path: Path) -> None:
    folder = tmp_path / "ngauto-download-x"
    folder.mkdir()
    path = folder / "project.zip"
    path.write_bytes(b"PK" * 1000)
    response = _DownloadResponse(ProjectDownloadModel(folder=folder, path=path, filename="demo.zip"))

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "GET", "headers": [], "extensions": {}}
    with pytest.raises(OSError):
        await response(scope, receive, send)

    assert not folder.exists()


def _make_dir_link(link: Path, target: Path) -> bool:
    """A symbolic link, or on Windows (no symlink privilege) a junction."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        if sys.platform != "win32":
            return False
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return True
    except OSError:
        return False
