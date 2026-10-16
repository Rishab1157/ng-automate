"""The healer's memory: earlier heals of the same project (never another project's, or organization's), found by
similarity of the problem.

recall() runs before a heal and returns the most similar earlier heals (what was tried and what came of it), for
the healer's prompt. remember() runs once the next test run, or the end of the test phase, shows what a heal did;
successful and failed heals are both kept. The memories are context, not instructions: the healer decides.

Never raises and never holds up healing for long: when memory is off, or Qdrant or the embedding server is down,
recall() returns no memories and remember() keeps nothing. Such a failure is not silent: its reason is logged once
(again only after memory has worked in between) and kept in `problem`, which the run's timeline shows at each heal.
"""

import asyncio
import logging
from typing import Protocol

from app.config import settings
from app.models.healMemoryModel import HealMemoryDbModel, HealMemoryMapper, HealMemoryModel, describe_problem
from app.models.testRunModel import TestRunResultModel
from app.repositories.healMemoryRepository import HealMemoryRepository

logger = logging.getLogger(__name__)

EMBEDDING_FAILED = (
    "Healer memory is not working: the embedding server {url} gave no vector for model {model} ({error}). "
    "The healer goes on without earlier heals; check EMBEDDING_BASE_URL and EMBEDDING_MODEL."
)
MEMORY_FAILED = "Healer memory is not working ({error}). The healer goes on without earlier heals."
MAX_ERROR_CHARS = 200


class _EmbeddingFailed(Exception):
    """The embedding server gave no vector; the message says so in words a user can act on."""


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
        # Why the last use of memory failed; None while it works (or is off).
        self.problem: str | None = None

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

    async def recall(
        self, *, org_id: str, project_id: str, run_id: str, result: TestRunResultModel
    ) -> list[HealMemoryModel]:
        """The project's earlier heals most similar to this failure, best first (other runs only)."""
        if not self.enabled:
            return []
        try:
            async with asyncio.timeout(self.timeout_seconds):
                vector = await self._embed(settings.EMBEDDING_QUERY_PREFIX + describe_problem(result))
                await self.repository.ensure_collection(len(vector))
                hits = await self.repository.search(
                    org_id=org_id,
                    project_id=project_id,
                    vector=vector,
                    embedding_model=settings.EMBEDDING_MODEL,
                    limit=settings.HEAL_MEMORY_LIMIT,
                    min_score=settings.HEAL_MEMORY_MIN_SCORE,
                    exclude_run_id=run_id,
                )
        except Exception as error:
            self._failed(error)
            return []
        self._worked()
        memories: list[HealMemoryModel] = []
        for point_id, payload, score in hits:
            # Defense in depth: the search already filters on both.
            if payload.get("org_id") != org_id or payload.get("project_id") != project_id:
                logger.error("Qdrant returned memory %s of another organization or project; it is dropped", point_id)
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
                vector = await self._embed(memory.problem)
                await self.repository.ensure_collection(len(vector))
                await self.repository.upsert(point_id, vector, memory.model_dump(mode="json"))
        except Exception as error:
            self._failed(error)
            logger.info("The healer memory of run %s was not saved", memory.run_id)
            return False
        self._worked()
        logger.info("Saved healer memory %s (run %s, attempt %d: %s)", point_id, memory.run_id, memory.attempt, memory.result)
        return True

    async def _embed(self, text: str) -> list[float]:
        try:
            [vector] = await self.embeddings.embed([text])
        except Exception as error:
            raise _EmbeddingFailed(EMBEDDING_FAILED.format(
                url=settings.EMBEDDING_BASE_URL, model=settings.EMBEDDING_MODEL, error=_short(error)
            )) from error
        return vector

    def _failed(self, error: Exception) -> None:
        """Keeps and logs why memory failed: a warning the first time, not again until memory has worked in between
        (every heal of every run would repeat it)."""
        embedding = isinstance(error, _EmbeddingFailed)
        problem = str(error) if embedding else MEMORY_FAILED.format(error=_short(error))
        if problem != self.problem:
            logger.warning("%s", problem, exc_info=not embedding)
        self.problem = problem

    def _worked(self) -> None:
        if self.problem is not None:
            logger.info("Healer memory works again")
        self.problem = None


def _short(error: Exception) -> str:
    """'ConnectError: [Errno 111] Connection refused', or just the type when the error has no text."""
    text = next((line.strip() for line in str(error).splitlines() if line.strip()), "")
    return f"{type(error).__name__}: {text}"[:MAX_ERROR_CHARS] if text else type(error).__name__
