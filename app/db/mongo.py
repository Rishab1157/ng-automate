"""One shared async MongoDB client for the whole app.

- get_db():       our database (projects, runs, events)
- get_qxcel_db(): QXcel's database. Read-only for us: git and model connections, modules.
"""

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.config import settings

_client: AsyncMongoClient | None = None


def get_client() -> AsyncMongoClient:
    global _client
    if _client is None:
        # serverSelectionTimeoutMS: fail fast when MongoDB is down instead of hanging.
        _client = AsyncMongoClient(settings.MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> AsyncDatabase:
    return get_client()[settings.MONGO_DATABASE]


def get_qxcel_db() -> AsyncDatabase:
    return get_client()[settings.QXCEL_DATABASE]


async def ping() -> bool:
    try:
        result = await get_client().admin.command("ping")
        return bool(result.get("ok"))
    except Exception:
        return False


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
