"""Healing memories in Qdrant: one collection, one point per heal, every search scoped to one organization.

The collection is made for many tenants the way Qdrant recommends: org_id is a tenant index and the HNSW graph is
built per organization (payload_m), not across all of them (m=0). A search without an organization is refused.
"""

import logging
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.db.qdrant import get_qdrant_client

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

TENANT_FIELD = "org_id"
KEYWORD_FIELDS = ("run_id", "embedding_model")
PAYLOAD_M = 16


class HealMemoryRepository:
    def __init__(self, client: "AsyncQdrantClient | None" = None, collection_name: str | None = None) -> None:
        self._client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self._ready_size: int | None = None

    @property
    def client(self) -> "AsyncQdrantClient":
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the collection and its indexes if missing; refuse one made for vectors of another size."""
        if self._ready_size == vector_size:
            return
        from qdrant_client import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        if await self.client.collection_exists(self.collection_name):
            info = await self.client.get_collection(self.collection_name)
            size = getattr(info.config.params.vectors, "size", None)
            if size != vector_size:
                raise ValueError(
                    f"Qdrant collection {self.collection_name!r} holds vectors of size {size}, the embedding model "
                    f"makes {vector_size}: use another QDRANT_COLLECTION_NAME for this model"
                )
        else:
            try:
                await self.client.create_collection(
                    self.collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                    hnsw_config=models.HnswConfigDiff(payload_m=PAYLOAD_M, m=0),
                )
            except UnexpectedResponse as error:
                if error.status_code != 409:  # 409: another worker created it a moment ago
                    raise
            await self.client.create_payload_index(
                self.collection_name,
                TENANT_FIELD,
                field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            )
            for field in KEYWORD_FIELDS:
                await self.client.create_payload_index(
                    self.collection_name, field, field_schema=models.PayloadSchemaType.KEYWORD
                )
            logger.info("Created the Qdrant collection %s (vector size %d)", self.collection_name, vector_size)
        self._ready_size = vector_size

    async def upsert(self, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        from qdrant_client import models

        if not payload.get(TENANT_FIELD):
            raise ValueError("A healing memory needs an org_id")
        await self.client.upsert(
            self.collection_name, points=[models.PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    async def search(
        self,
        *,
        org_id: str,
        vector: list[float],
        embedding_model: str,
        limit: int,
        min_score: float,
        exclude_run_id: str | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """(point id, payload, similarity) of the organization's most similar memories, best first."""
        from qdrant_client import models

        if not org_id:
            raise ValueError("A memory search needs an org_id: memories are never searched across organizations")
        must = [
            models.FieldCondition(key=TENANT_FIELD, match=models.MatchValue(value=org_id)),
            models.FieldCondition(key="embedding_model", match=models.MatchValue(value=embedding_model)),
        ]
        must_not = (
            [models.FieldCondition(key="run_id", match=models.MatchValue(value=exclude_run_id))] if exclude_run_id else []
        )
        response = await self.client.query_points(
            self.collection_name,
            query=vector,
            query_filter=models.Filter(must=must, must_not=must_not),
            limit=limit,
            score_threshold=min_score,
            with_payload=True,
        )
        return [(str(point.id), point.payload or {}, point.score) for point in response.points]
