"""Shared async MongoDB clients for the whole app.

- get_db():       our database (projects, runs, events), on MONGO_URI
- get_qxcel_db(): QXcel's database, on QXCEL_MONGO_URI (MONGO_URI when not set). Read-only for us: git and model
                  connections, modules.
"""

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.config import settings

_client: AsyncMongoClient | None = None
_qxcel_client: AsyncMongoClient | None = None


def get_client() -> AsyncMongoClient:
    global _client
    if _client is None:
        _client = _new_client(settings.MONGO_URI)
    return _client


def get_qxcel_client() -> AsyncMongoClient:
    global _qxcel_client
    if not settings.QXCEL_MONGO_URI or settings.QXCEL_MONGO_URI == settings.MONGO_URI:
        return get_client()
    if _qxcel_client is None:
        _qxcel_client = _new_client(settings.QXCEL_MONGO_URI)
    return _qxcel_client


def _new_client(uri: str) -> AsyncMongoClient:
    # serverSelectionTimeoutMS: fail fast when MongoDB is down instead of hanging.
    return AsyncMongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)


def get_db() -> AsyncDatabase:
    return get_client()[settings.MONGO_DATABASE]


def get_qxcel_db() -> AsyncDatabase:
    return get_qxcel_client()[settings.QXCEL_DATABASE]


async def ping() -> bool:
    try:
        result = await get_client().admin.command("ping")
        return bool(result.get("ok"))
    except Exception:
        return False


async def close_client() -> None:
    global _client, _qxcel_client
    for client in (_client, _qxcel_client):
        if client is not None:
            await client.close()
    _client = _qxcel_client = None
