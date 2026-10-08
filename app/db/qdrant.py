"""One shared async Qdrant client: the vector database of the healer's memory (MongoDB keeps all other data).

The client library loads on first use, so the API starts without it.
"""

from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

_client: "AsyncQdrantClient | None" = None


def get_qdrant_client() -> "AsyncQdrantClient":
    global _client
    if _client is None:
        if not settings.QDRANT_URL:
            raise RuntimeError("QDRANT_URL is not set")
        from qdrant_client import AsyncQdrantClient

        api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
        _client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=api_key, timeout=settings.QDRANT_TIMEOUT_SECONDS)
    return _client


async def close_qdrant_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
