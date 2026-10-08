"""Test setup. Tests use their own databases and data folder, never the developer's."""

import io
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import AsyncIterator, Callable, Iterator

# Must run before `app` is imported: settings are read once at import time.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="ngauto-test-data-")
_TEST_MONGO_URI = os.environ.get("TEST_MONGO_URI", "mongodb://localhost:27017")
os.environ.update({
    "MONGO_URI": _TEST_MONGO_URI,
    # Never the developer's QXcel server from .env: the test QXcel database lives next to the test database.
    "QXCEL_MONGO_URI": _TEST_MONGO_URI,
    # The healer's memory (Qdrant, embeddings) is off; its tests use an in-memory Qdrant and fake embeddings.
    "HEAL_MEMORY_ENABLED": "false",
    # Per-process names: several test runs at once never share (or drop) each other's data.
    "MONGO_DATABASE": f"ng_automate_test_{os.getpid()}",
    "QXCEL_DATABASE": f"ng_automate_test_qxcel_{os.getpid()}",
    "DATA_DIR": _TEST_DATA_DIR,
    "SECRET_KEY": "test-secret-key-that-is-at-least-32-bytes-long",
    "LLM_MODEL": "test-model",
    "MAX_PROJECT_MB": "5",
})

import jwt  # noqa: E402
import pytest  # noqa: E402
from bson import ObjectId  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.mongo import close_client  # noqa: E402
from app.main import app  # noqa: E402
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_OWN_ORG  # noqa: E402

NG_MODULE_ID = ObjectId()
GITHUB_PROVIDER_ID = ObjectId()
PUBLIC_REPO_URL = "https://github.com/anhtester/AutomationFrameworkCucumberTestNG.git"


@pytest.fixture(scope="session")
def mongo() -> Iterator[MongoClient]:
    client: MongoClient = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("MongoDB is not running")

    for name in (settings.MONGO_DATABASE, settings.QXCEL_DATABASE):
        client.drop_database(name)
    qxcel = client[settings.QXCEL_DATABASE]
    qxcel.modules.insert_one({"_id": NG_MODULE_ID, "module_code": "NGAUTOMATE", "module_name": "NG Automate"})
    qxcel.git_providers.insert_one({"_id": GITHUB_PROVIDER_ID, "provider_code": "GITHUB", "is_active": True})

    yield client

    for name in (settings.MONGO_DATABASE, settings.QXCEL_DATABASE):
        client.drop_database(name)
    client.close()
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture
def anyio_backend() -> str:
    """Async tests (`@pytest.mark.anyio`) run on asyncio, like the app."""
    return "asyncio"


@pytest.fixture
async def app_db(mongo: MongoClient) -> AsyncIterator[MongoClient]:
    """For async service tests: the app's Mongo client is bound to one event loop, so close it after each test.

    Yields the sync client for seeding and assertions; cleans our collections afterwards.
    """
    yield mongo
    await close_client()
    for name in ("projects", "runs", "run_events", "project_profiles", "test_data", "live_view_tickets", "run_commands"):
        mongo[settings.MONGO_DATABASE][name].delete_many({})
    for name in ("git_connections", "model_connections", "model_providers", "model_types"):
        mongo[settings.QXCEL_DATABASE][name].delete_many({})


@pytest.fixture
def client(mongo: MongoClient) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    for name in ("projects", "runs", "run_events", "project_profiles", "test_data", "live_view_tickets", "run_commands"):
        mongo[settings.MONGO_DATABASE][name].delete_many({})
    for name in ("git_connections", "model_connections", "model_providers", "model_types"):
        mongo[settings.QXCEL_DATABASE][name].delete_many({})


@pytest.fixture
def org_id() -> str:
    return str(ObjectId())


@pytest.fixture
def auth_headers() -> Callable[..., dict[str, str]]:
    def make(
        org_id: str,
        permissions: tuple[str, ...] = (NGAUTOMATE_ACCESS_OWN_ORG,),
        expires_in: int = 300,
        secret: str = settings.SECRET_KEY.get_secret_value(),
    ) -> dict[str, str]:
        payload = {
            "user_id": str(ObjectId()),
            "org_id": org_id,
            "permissions": list(permissions),
            "exp": int(time.time()) + expires_in,
        }
        return {"Authorization": f"Bearer {jwt.encode(payload, secret, algorithm='HS256')}"}

    return make


@pytest.fixture
def add_git_connection(mongo: MongoClient) -> Callable[..., str]:
    def add(
        org_id: str,
        repo_url: str = PUBLIC_REPO_URL,
        branch: str = "main",
        enabled_for_ng_automate: bool = True,
        is_active: bool = True,
    ) -> str:
        module_ids = [NG_MODULE_ID] if enabled_for_ng_automate else []
        result = mongo[settings.QXCEL_DATABASE].git_connections.insert_one({
            "org_id": ObjectId(org_id),
            "userdefined_name": "Test connection",
            "repo_url": repo_url,
            "branch": branch,
            "git_provider_id": GITHUB_PROVIDER_ID,
            "is_active": is_active,
            "available_for_module_ids": module_ids,
            "active_for_module_ids": module_ids,
        })
        return str(result.inserted_id)

    return add


def make_zip(files: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
