"""The healer's memory: earlier heals of the same organization, found by similarity of the problem.

recall() runs before a heal and returns the most similar earlier heals (what was tried and what came of it), for
the healer's prompt. remember() runs once the next test run, or the end of the test phase, shows what a heal did;
successful and failed heals are both kept. The memories are context, not instructions: the healer decides.

Never raises and never holds up healing for long: when memory is off, or Qdrant or the embedding server is down,
recall() returns no memories and remember() keeps nothing.
"""

import asyncio
import logging
from typing import Protocol

from app.config import settings
from app.models.healMemoryModel import HealMemoryDbModel, HealMemoryMapper, HealMemoryModel, describe_problem
from app.models.testRunModel import TestRunResultModel
from app.repositories.healMemoryRepository import HealMemoryRepository

logger = logging.getLogger(__name__)


class Embeddings(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HealMemoryService:
    def __init__(
        self,
        repository: HealMemoryRepository | None = None,
        embeddings: Embeddings | None = None,
        enabled: bool | None = None,
    ) -> None:
        if enabled is None:
            enabled = settings.HEAL_MEMORY_ENABLED and bool(settings.QDRANT_URL) and bool(settings.EMBEDDING_BASE_URL)
        self.enabled = enabled
        self._repository = repository
        self._embeddings = embeddings
        # Longest one call may take: an embedding and up to three Qdrant requests (first use creates the collection).
        self.timeout_seconds = settings.EMBEDDING_TIMEOUT_SECONDS + 3 * settings.QDRANT_TIMEOUT_SECONDS

    @property
    def repository(self) -> HealMemoryRepository:
        if self._repository is None:
            self._repository = HealMemoryRepository()
        return self._repository

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from app.utils.EmbeddingClient import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings()
        return self._embeddings

    async def recall(self, *, org_id: str, run_id: str, result: TestRunResultModel) -> list[HealMemoryModel]:
        """The organization's earlier heals most similar to this failure, best first (other runs only)."""
        if not self.enabled:
            return []
        try:
            async with asyncio.timeout(self.timeout_seconds):
                [vector] = await self.embeddings.embed([settings.EMBEDDING_QUERY_PREFIX + describe_problem(result)])
                await self.repository.ensure_collection(len(vector))
                hits = await self.repository.search(
                    org_id=org_id,
                    vector=vector,
                    embedding_model=settings.EMBEDDING_MODEL,
                    limit=settings.HEAL_MEMORY_LIMIT,
                    min_score=settings.HEAL_MEMORY_MIN_SCORE,
                    exclude_run_id=run_id,
                )
        except Exception:
            logger.warning("Healer memory search failed; the healer goes on without memories", exc_info=True)
            return []
        memories: list[HealMemoryModel] = []
        for point_id, payload, score in hits:
            if payload.get("org_id") != org_id:  # defense in depth: the search already filters on it
                logger.error("Qdrant returned memory %s of another organization; it is dropped", point_id)
                continue
            try:
                memories.append(HealMemoryMapper.to_model(point_id, payload, score))
            except Exception:
                logger.warning("Skipping unreadable healer memory %s", point_id, exc_info=True)
        return memories

    async def remember(self, point_id: str, memory: HealMemoryDbModel) -> bool:
        """Keep one heal (replacing an earlier save of the same heal). False when it could not be kept."""
        if not self.enabled:
            return False
        try:
            async with asyncio.timeout(self.timeout_seconds):
                [vector] = await self.embeddings.embed([memory.problem])
                await self.repository.ensure_collection(len(vector))
                await self.repository.upsert(point_id, vector, memory.model_dump(mode="json"))
        except Exception:
            logger.warning("Could not save the healer memory of run %s", memory.run_id, exc_info=True)
            return False
        logger.info("Saved healer memory %s (run %s, attempt %d: %s)", point_id, memory.run_id, memory.attempt, memory.result)
        return True
