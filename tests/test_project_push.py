"""POST /project/{project_id}/git/push: a git project's files back to its repository.

A bare repository on disk stands in for GitHub (no network): the https-only rule is relaxed for file:// URLs in these
tests only. Git runs with an empty global config, so the developer's own git settings never change the results.
"""

import base64
import importlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import jwt
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import ErrorMessages, GitFetchError
from app.models.authModel import CurrentUserModel
from app.models.projectModel import ArchiveModel, GitOriginModel, ProjectMapper, ProjectSource
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_OWN_ORG
from app.services.gitConnectionService.GitPushService import describe_push_failure, new_branch_name
from app.services.projectService import ProjectStorageService
from app.dto.projectDto.requestDto.PushProjectDTO import PushProjectDTO
from app.services.projectService.ProjectGitService import DEFAULT_COMMIT_MESSAGE, commit_author
from app.utils.ArchiveUtils import sha256_file
from tests.conftest import GITHUB_PROVIDER_ID, NG_MODULE_ID, make_zip

# The modules (the package exports classes of the same names).
fetch_module = importlib.import_module("app.services.gitConnectionService.GitFetchService")
push_module = importlib.import_module("app.services.gitConnectionService.GitPushService")

BASE = "/tx-agents/ng-automate/project"
TOKEN = "tok-SECRET-9f8e7d6c5b4a"
NOW = "20261028-120000"
_real_run = subprocess.run  # the helpers below run git for real, also while a test spies on subprocess.run

INITIAL_FILES = {
    "pom.xml": "<project/>\n",
    "README.md": "# Shop tests\n",
    "src/test/java/LoginTest.java": "class LoginTest {}\n",
    "docs/guide.md": "How to run\n",
    ".gitignore": "*.log\n",
}


# ---------- a local "GitHub" ----------

class Remote:
    """A bare repository standing in for the remote. A teammate's commits go straight into it (git fast-import: one
    git process per commit, which keeps these tests fast on Windows)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.url = path.as_uri()

    def commit(self, files: dict[str, str | bytes | None], branch: str = "main", start: str | None = None) -> str:
        """A teammate's commit on `branch` (None deletes a file); a new branch starts from `start`. Returns its sha."""
        stream = [f"commit refs/heads/{branch}\n".encode(), b"committer Teammate <teammate@example.com> 1761650000 +0000\n"]
        stream.append(_data(b"teammate"))
        stream.append(f"from refs/heads/{start or branch}^0\n".encode())
        for name, content in files.items():
            if content is None:
                stream.append(f"D {name}\n".encode())
            else:
                stream.append(f"M 100644 inline {name}\n".encode())
                stream.append(_data(content.encode() if isinstance(content, str) else content))
        _fast_import(self.path, b"".join(stream))
        return self.head(branch)

    def files(self, ref: str = "main") -> set[str]:
        return {name for name in git("-C", str(self.path), "ls-tree", "-r", "-z", "--name-only", ref).split("\0") if name}

    def read(self, path: str, ref: str = "main") -> bytes:
        return _real_run(["git", "-C", str(self.path), "show", f"{ref}:{path}"], capture_output=True, check=True).stdout

    def head(self, ref: str = "main") -> str:
        return git("-C", str(self.path), "rev-parse", ref).strip()

    def branches(self) -> set[str]:
        return set(git("-C", str(self.path), "for-each-ref", "--format=%(refname:short)", "refs/heads").split())

    def author(self, ref: str = "main") -> str:
        return git("-C", str(self.path), "log", "-1", "--format=%an <%ae>|%s", ref).strip()

    def reject_pushes(self) -> None:
        hook = self.path / "hooks" / "pre-receive"
        hook.write_bytes(b"#!/bin/sh\necho 'push blocked by policy' >&2\nexit 1\n")
        hook.chmod(0o755)


def _data(content: bytes) -> bytes:
    return b"data %d\n" % len(content) + content + b"\n"


def _fast_import(repo: Path, stream: bytes) -> None:
    result = _real_run(["git", "-C", str(repo), "fast-import", "--quiet"], input=stream, capture_output=True)
    assert result.returncode == 0, result.stderr


def git(*args: str) -> str:
    result = _real_run(["git", *args], capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _hermetic_git(patch: pytest.MonkeyPatch, folder: Path) -> None:
    """Git (the services' and the tests' own) without the developer's settings: an empty global config, no system
    config."""
    config = folder / "empty.gitconfig"
    config.write_text("")
    patch.setenv("GIT_CONFIG_GLOBAL", str(config))
    patch.setenv("GIT_CONFIG_NOSYSTEM", "1")


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def remote_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The remote as every test starts it (INITIAL_FILES on main), made once and copied for each test."""
    root = tmp_path_factory.mktemp("remote-template")
    with pytest.MonkeyPatch.context() as patch:
        _hermetic_git(patch, root)
        git("init", "-q", "--bare", "-b", "main", str(root / "remote.git"))
        stream = [b"commit refs/heads/main\n", b"committer Teammate <teammate@example.com> 1761640000 +0000\n"]
        stream.append(_data(b"initial"))
        for name, content in INITIAL_FILES.items():
            stream += [f"M 100644 inline {name}\n".encode(), _data(content.encode())]
        _fast_import(root / "remote.git", b"".join(stream))
    return root / "remote.git"


@pytest.fixture(autouse=True)
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Git without the developer's settings; file:// remotes allowed (these tests only); temporary folders in a folder
    of the test's own, to see that pushes leave nothing behind."""
    _hermetic_git(monkeypatch, tmp_path)
    monkeypatch.setattr(fetch_module, "check_repo_url", _allow_local_repo)
    monkeypatch.setattr(push_module, "check_repo_url", _allow_local_repo)
    monkeypatch.setattr(push_module, "_timestamp", lambda: NOW)
    temp_root = tmp_path / "server-temp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    return temp_root


def _allow_local_repo(repo_url: str, error: Any = GitFetchError) -> None:
    if not repo_url.startswith("file:///"):
        raise error(ErrorMessages.GIT_ONLY_HTTPS)


@pytest.fixture
def remote(tmp_path: Path, remote_template: Path) -> Remote:
    return Remote(Path(shutil.copytree(remote_template, tmp_path / "remote.git")))


class GitSpy:
    """Records every command (and its environment) the services give git."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(command), dict(kwargs.get("env") or {})))
        return _real_run(command, **kwargs)


@pytest.fixture
def git_spy(monkeypatch: pytest.MonkeyPatch) -> GitSpy:
    spy = GitSpy()
    monkeypatch.setattr(subprocess, "run", spy)
    return spy


def _headers(org_id: str, **claims: str) -> dict[str, str]:
    payload = {
        "user_id": str(ObjectId()),
        "org_id": org_id,
        "permissions": [NGAUTOMATE_ACCESS_OWN_ORG],
        "exp": int(time.time()) + 300,
        **claims,
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, settings.SECRET_KEY.get_secret_value(), algorithm='HS256')}"}


def _user_headers(org_id: str) -> dict[str, str]:
    return _headers(org_id, firstname="Asha", lastname="Rao", email="asha.rao@example.com")


def _add_connection(mongo: MongoClient, org_id: str, repo_url: str, name: str = "Test connection") -> str:
    return str(mongo[settings.QXCEL_DATABASE].git_connections.insert_one({
        "org_id": ObjectId(org_id),
        "userdefined_name": name,
        "repo_url": repo_url,
        "branch": "main",
        "token": TOKEN,
        "git_provider_id": GITHUB_PROVIDER_ID,
        "is_active": True,
        "available_for_module_ids": [NG_MODULE_ID],
        "active_for_module_ids": [NG_MODULE_ID],
    }).inserted_id)


def _create_project(client: TestClient, headers: dict[str, str], connection_id: str) -> dict[str, Any]:
    response = client.post(f"{BASE}/git", headers=headers, json={"git_connection_id": connection_id})
    assert response.status_code == 201, response.text
    return response.json()


def _working_copy(project_id: str) -> Path:
    return Path(settings.DATA_DIR) / "projects" / project_id / "project"


def _push(client: TestClient, headers: dict[str, str], project_id: str, target: str, message: str | None = None):
    body = {"target": target} if message is None else {"target": target, "message": message}
    return client.post(f"{BASE}/{project_id}/git/push", headers=headers, json=body)


def _write(folder: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)


def _seed_git_project(mongo: MongoClient, org_id: str, remote: Remote, connection_id: str) -> dict[str, Any]:
    """A git project as POST /project/git leaves it (snapshot, editable copy, document), without cloning again."""
    project_id = str(ObjectId())
    storage = ProjectStorageService()
    snapshot = storage.archive_path(project_id)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(make_zip(INITIAL_FILES))
    storage.ensure_working_copy(project_id)
    commit_sha = remote.head()
    db_model = ProjectMapper.to_create_db_model(
        project_id=ObjectId(project_id), org_id=org_id, user_id=str(ObjectId()), name="shop", source=ProjectSource.GIT,
        archive=ArchiveModel(size_bytes=snapshot.stat().st_size, sha256=sha256_file(snapshot), file_count=len(INITIAL_FILES)),
        git=GitOriginModel(connection_id=connection_id, repo_url=remote.url, branch="main", commit_sha=commit_sha),
    )
    mongo[settings.MONGO_DATABASE].projects.insert_one(db_model.model_dump(by_alias=True))
    return {"id": project_id, "git": {"commit_sha": commit_sha}, "connection_id": connection_id}


@pytest.fixture
def git_project(client: TestClient, mongo: MongoClient, remote: Remote, org_id: str) -> dict[str, Any]:
    """A project of the remote's main branch."""
    return _seed_git_project(mongo, org_id, remote, _add_connection(mongo, org_id, remote.url))


# ---------- target current ----------

def test_current_branch_gets_our_files_and_keeps_a_teammates_new_ones(
    client: TestClient, mongo: MongoClient, remote: Remote, org_id: str, isolated_git: Path
) -> None:
    # Created through the API like a user does: the real fetch, its snapshot and recorded commit.
    git_project = _create_project(client, _user_headers(org_id), _add_connection(mongo, org_id, remote.url))
    working = _working_copy(git_project["id"])
    _write(working, {
        "README.md": "# Shop tests (healed by NG Automate)\n",
        "src/test/java/CartTest.java": "class CartTest {}\n",
        "debug.log": "ignored by the repo's .gitignore\n",
        "ngauto-results/junit.xml": "<testsuite/>",
        "test-results/trace.zip": b"PK",
        "playwright-report/index.html": "<html/>",
        "allure-results/x.json": "{}",
        "target/surefire-reports/TEST-x.xml": "<testsuite/>",
        "node_modules/left-pad/index.js": "module.exports = 1",
    })
    (working / "docs" / "guide.md").unlink()
    remote.commit({"README.md": "# Shop tests (teammate's edit)\n", "CONTRIBUTING.md": "Be nice\n"})

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "target": "current",
        "branch": "main",
        "repo_url": remote.url,
        "web_url": None,  # not GitHub/GitLab/Bitbucket
        "commit_sha": remote.head("main"),
        "files_changed": 3,  # README.md changed, CartTest.java added, docs/guide.md deleted
        "nothing_to_push": False,
        "based_on": "branch_head",
    }
    assert remote.read("README.md") == b"# Shop tests (healed by NG Automate)\n"  # ours wins over the teammate's
    assert remote.files() == {
        "pom.xml", "README.md", ".gitignore", "src/test/java/LoginTest.java", "src/test/java/CartTest.java",
        "CONTRIBUTING.md",  # the teammate's new file stays
    }
    assert remote.author() == f"Asha Rao <asha.rao@example.com>|{DEFAULT_COMMIT_MESSAGE}"
    assert list(isolated_git.iterdir()) == []  # no temporary folder left

    again = _push(client, _user_headers(org_id), git_project["id"], "current")  # the same files once more

    assert again.status_code == 200
    second = again.json()
    assert (second["nothing_to_push"], second["commit_sha"], second["files_changed"], second["branch"]) == (True, None, 0, "main")
    assert remote.head() == body["commit_sha"]


# ---------- target new_branch ----------

def test_new_branch_starts_at_the_recorded_commit_and_the_token_stays_in_the_environment(
    client: TestClient, remote: Remote, git_project: dict[str, Any], org_id: str, git_spy: "GitSpy",
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorded = git_project["git"]["commit_sha"]
    _write(_working_copy(git_project["id"]), {"src/test/java/CartTest.java": "class CartTest {}\n"})
    remote.commit({"CONTRIBUTING.md": "Be nice\n"})  # main moves on after the project was created

    with caplog.at_level(logging.DEBUG):
        response = _push(client, _user_headers(org_id), git_project["id"], "new_branch", message="Heal the login test")

    assert response.status_code == 200, response.text
    body = response.json()
    branch = f"test-connection-{NOW}"
    assert (body["target"], body["branch"], body["based_on"], body["files_changed"]) == ("new_branch", branch, "recorded_commit", 1)
    assert body["commit_sha"] == remote.head(branch)
    assert remote.head(f"{branch}~1") == recorded
    assert "CONTRIBUTING.md" not in remote.files(branch)
    assert "src/test/java/CartTest.java" in remote.files(branch)
    assert "src/test/java/CartTest.java" not in remote.files("main")  # main is untouched
    assert remote.author(branch) == "Asha Rao <asha.rao@example.com>|Heal the login test"
    assert TOKEN not in response.text and TOKEN not in caplog.text
    _assert_token_only_in_env(git_spy)
    assert any("push" in command for command, _ in git_spy.calls)


def test_new_branch_of_an_odd_connection_name_gets_a_suffix_when_it_exists_twice(
    client: TestClient, mongo: MongoClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    mongo[settings.QXCEL_DATABASE].git_connections.update_one(
        {"_id": ObjectId(git_project["connection_id"])}, {"$set": {"userdefined_name": " ..Ünïcode / Tests.. "}}
    )
    taken = f"unicode-tests-{NOW}"
    remote.commit({"a.txt": "a"}, branch=taken, start="main")
    remote.commit({"b.txt": "b"}, branch=f"{taken}-2", start="main")
    _write(_working_copy(git_project["id"]), {"README.md": "changed\n"})

    response = _push(client, _headers(org_id), git_project["id"], "new_branch", message="   ")

    assert response.status_code == 200, response.text
    assert response.json()["branch"] == f"{taken}-3"
    assert {taken, f"{taken}-2", f"{taken}-3"} <= remote.branches()
    # No names in the token and a blank message: NG Automate's own.
    assert remote.author(f"{taken}-3") == f"NG Automate <ngautomate@noreply.local>|{DEFAULT_COMMIT_MESSAGE}"


def test_new_branch_falls_back_to_the_branch_head_when_the_recorded_commit_is_gone(
    client: TestClient, mongo: MongoClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    mongo[settings.MONGO_DATABASE].projects.update_one(
        {"_id": ObjectId(git_project["id"])}, {"$set": {"git.commit_sha": "0123456789abcdef0123456789abcdef01234567"}}
    )
    head = remote.commit({"CONTRIBUTING.md": "Be nice\n"})
    _write(_working_copy(git_project["id"]), {"README.md": "changed\n"})

    response = _push(client, _user_headers(org_id), git_project["id"], "new_branch")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["based_on"] == "branch_head"
    assert remote.head(f"{body['branch']}~1") == head


def test_new_branch_with_nothing_changed_creates_no_branch(
    client: TestClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    response = _push(client, _user_headers(org_id), git_project["id"], "new_branch")

    assert response.status_code == 200
    body = response.json()
    assert (body["nothing_to_push"], body["branch"], body["web_url"], body["commit_sha"]) == (True, None, None, None)
    assert remote.branches() == {"main"}


@pytest.mark.parametrize(
    "name",
    ["Test connection", "  ..Café / Ünïcode..lock -- Tests!! ", "テスト", "---", "a" * 300, "x.lock", "@{weird}~^:?*[\\"],
)
def test_any_connection_name_gives_a_valid_branch_name(name: str) -> None:
    branch = new_branch_name(name, NOW)

    assert git("check-ref-format", "--branch", branch).strip() == branch
    assert branch == branch.lower() and not branch.startswith((".", "-")) and branch.endswith(NOW)
    assert set(branch) <= set("abcdefghijklmnopqrstuvwxyz0123456789._-")


def test_git_project_downloads_like_any_project(
    client: TestClient, git_project: dict[str, Any], org_id: str
) -> None:
    _write(_working_copy(git_project["id"]), {"src/test/java/CartTest.java": "class CartTest {}\n", "ngauto-results/a.xml": "<x/>"})

    response = client.get(f"{BASE}/{git_project['id']}/download", headers=_user_headers(org_id))

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == set(INITIAL_FILES) | {"src/test/java/CartTest.java"}


# ---------- refusals and failures ----------

def test_uploaded_project_can_only_be_downloaded(client: TestClient, org_id: str) -> None:
    headers = _user_headers(org_id)
    upload = client.post(f"{BASE}/upload", headers=headers, files={"file": ("demo.zip", make_zip({"a.txt": "a"}), "application/zip")})

    response = _push(client, headers, upload.json()["id"], "current")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "NOT_A_GIT_PROJECT"
    assert response.json()["error"]["message"] == ErrorMessages.NOT_A_GIT_PROJECT


def test_busy_project_is_not_pushed(
    client: TestClient, mongo: MongoClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    mongo[settings.MONGO_DATABASE].runs.insert_one(
        {"project_id": ObjectId(git_project["id"]), "org_id": ObjectId(org_id), "status": "running"}
    )
    before = remote.head()

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROJECT_BUSY"
    assert remote.head() == before


@pytest.mark.parametrize("change", ["deleted", "disabled"])
def test_missing_connection_is_not_found(
    client: TestClient, mongo: MongoClient, git_project: dict[str, Any], org_id: str, change: str
) -> None:
    connections = mongo[settings.QXCEL_DATABASE].git_connections
    query = {"_id": ObjectId(git_project["connection_id"])}
    if change == "deleted":
        connections.delete_one(query)
    else:
        connections.update_one(query, {"$set": {"is_active": False}})

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CONNECTION_NOT_FOUND"


def test_another_orgs_project_is_not_found(client: TestClient, git_project: dict[str, Any]) -> None:
    response = _push(client, _user_headers(str(ObjectId())), git_project["id"], "current")

    assert response.status_code == 404


def test_rejected_push_is_explained_without_the_token(
    client: TestClient,
    remote: Remote,
    git_project: dict[str, Any],
    org_id: str,
    isolated_git: Path,
    git_spy: GitSpy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write(_working_copy(git_project["id"]), {"README.md": "changed\n"})
    remote.reject_pushes()
    before = remote.head()

    with caplog.at_level(logging.DEBUG):
        response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "GIT_PUSH_FAILED"
    assert error["message"] == ErrorMessages.GIT_PUSH_REJECTED.format(reason="push blocked by policy")
    assert remote.head() == before
    assert list(isolated_git.iterdir()) == []  # the temporary clone is removed after a failure too
    assert TOKEN not in response.text and TOKEN not in caplog.text
    _assert_token_only_in_env(git_spy)


def _assert_token_only_in_env(spy: GitSpy) -> None:
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert spy.calls
    for command, env in spy.calls:
        assert not any(TOKEN in part or basic in part for part in command)
        assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {basic}"


def test_empty_project_is_not_pushed(
    client: TestClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    working = _working_copy(git_project["id"])
    for child in working.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    _write(working, {"node_modules/x.js": "tool folders do not count"})

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 400
    assert response.json()["error"]["message"] == ErrorMessages.PROJECT_NOTHING_TO_PUSH
    assert remote.files() == set(INITIAL_FILES)


def test_target_must_be_known(client: TestClient, git_project: dict[str, Any], org_id: str) -> None:
    response = _push(client, _user_headers(org_id), git_project["id"], "force")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            f"remote: Permission to acme/shop.git denied to bot.\nfatal: unable to access 'https://x-access-token:{TOKEN}"
            "@github.com/acme/shop.git/': The requested URL returned error: 403\n",
            ErrorMessages.GIT_NO_WRITE_ACCESS,
        ),
        ("remote: Write access to repository not granted.\nfatal: unable to access '...': The requested URL returned "
         "error: 403\n", ErrorMessages.GIT_NO_WRITE_ACCESS),
        ("remote: You are not allowed to push code to this project.\n", ErrorMessages.GIT_NO_WRITE_ACCESS),
        ("remote: Invalid username or password.\nfatal: Authentication failed for 'https://github.com/a/b.git/'\n",
         ErrorMessages.GIT_AUTH_FAILED),
        (
            "remote: error: GH006: Protected branch update failed for refs/heads/main.\n"
            "remote: error: Changes must be made through a pull request.\n"
            "To https://github.com/acme/shop.git\n ! [remote rejected] main -> main (protected branch hook declined)\n",
            ErrorMessages.GIT_PUSH_REJECTED.format(
                reason="GH006: Protected branch update failed for refs/heads/main. Changes must be made through a pull "
                "request."
            ),
        ),
        (
            "To https://github.com/acme/shop.git\n ! [rejected]        main -> main (fetch first)\n",
            ErrorMessages.GIT_PUSH_OUTDATED.format(branch="main"),
        ),
        (
            "To https://github.com/acme/shop.git\n ! [remote rejected] main -> main (pre-receive hook declined)\n",
            ErrorMessages.GIT_PUSH_REJECTED.format(reason="pre-receive hook declined"),
        ),
    ],
)
def test_push_failures_are_explained_in_plain_words(stderr: str, expected: str) -> None:
    message = describe_push_failure(stderr, TOKEN, "main")

    assert message == expected
    assert TOKEN not in message


def test_unknown_failure_text_is_scrubbed_of_the_token_also_when_encoded() -> None:
    basic = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()

    message = describe_push_failure(f"fatal: header Authorization: Basic {basic} {TOKEN} broke\n", TOKEN, "main")

    assert TOKEN not in message and basic not in message
    assert message.startswith("Git failed:")


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        ({"firstname": "Asha", "lastname": "Rao", "email": "asha@example.com"}, ("Asha Rao", "asha@example.com")),
        ({"firstname": " <Evil>\n", "lastname": "", "email": "not an email"}, ("Evil", "ngautomate@noreply.local")),
        ({}, ("NG Automate", "ngautomate@noreply.local")),
        ({"firstname": "...", "email": "a@b.c"}, ("NG Automate", "a@b.c")),
    ],
)
def test_commit_author_comes_from_the_token(claims: dict[str, str], expected: tuple[str, str]) -> None:
    user = CurrentUserModel(user_id=str(ObjectId()), org_id=str(ObjectId()), **claims)

    author = commit_author(user)

    assert (author.name, author.email) == expected


def test_the_default_commit_message_claims_nothing_the_push_may_not_hold() -> None:
    # It said "generated and healed tests" also for a project nothing was healed in (or nothing generated).
    assert "heal" not in DEFAULT_COMMIT_MESSAGE.lower() and "generat" not in DEFAULT_COMMIT_MESSAGE.lower()
    assert f"'{DEFAULT_COMMIT_MESSAGE}'" in (PushProjectDTO.model_fields["message"].description or "")


# ---------- break it ----------

def test_weird_file_names_large_files_and_their_deletion(
    client: TestClient, mongo: MongoClient, remote: Remote, org_id: str
) -> None:
    remote.commit({"docs/old notes ü.md": "old"})
    project = _create_project(client, _user_headers(org_id), _add_connection(mongo, org_id, remote.url))
    working = _working_copy(project["id"])
    big = os.urandom(8 * 1024 * 1024)  # over the 5 MB project limit these tests use: a push has no such limit
    _write(working, {"src/test/java/My Test ü.java": "class T {}\n", "src/日本語.txt": "テスト\n", "data/big.bin": big})
    (working / "docs" / "old notes ü.md").unlink()

    response = _push(client, _user_headers(org_id), project["id"], "current")

    assert response.status_code == 200, response.text
    files = remote.files()
    assert {"src/test/java/My Test ü.java", "src/日本語.txt", "data/big.bin"} <= files
    assert "docs/old notes ü.md" not in files
    assert remote.read("data/big.bin") == big


def test_links_and_git_folders_in_the_project_never_reach_the_repository(
    client: TestClient, remote: Remote, git_project: dict[str, Any], org_id: str, tmp_path: Path
) -> None:
    outside = tmp_path / "server-secrets"
    _write(outside, {"secret.txt": "do not push"})
    working = _working_copy(git_project["id"])
    linked = _make_dir_link(working / "linked", outside)
    _write(working, {
        ".GIT/config": "[core]\n\tfsmonitor = calc.exe\n",  # on Windows this would be the clone's own .git/config
        "sub/.git": "gitdir: ../elsewhere\n",
        "README.md": "changed\n",
    })

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 200, response.text
    assert response.json()["files_changed"] == 1  # README.md only
    assert remote.files() == set(INITIAL_FILES)
    assert linked or sys.platform != "win32"  # Windows can always make a junction: the link case really ran there


def test_a_file_where_the_remote_has_a_folder_wins(
    client: TestClient, remote: Remote, git_project: dict[str, Any], org_id: str
) -> None:
    remote.commit({"reports/summary.md": "teammate's folder"})
    _write(_working_copy(git_project["id"]), {"reports": "now a file\n"})

    response = _push(client, _user_headers(org_id), git_project["id"], "current")

    assert response.status_code == 200, response.text
    assert remote.read("reports") == b"now a file\n"
    assert "reports/summary.md" not in remote.files()


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
