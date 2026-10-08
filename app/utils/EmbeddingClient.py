"""Turns text into vectors with an embedding model on an Ollama server (the healer's memory). No reasoning here."""

import httpx

from app.config import settings

# A server that does not answer at all is given up on quickly; a slow answer gets the full timeout.
CONNECT_TIMEOUT_SECONDS = 5.0


class OllamaEmbeddings:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base_url = base_url or settings.EMBEDDING_BASE_URL
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL is not set")
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model or settings.EMBEDDING_MODEL
        timeout = timeout_seconds if timeout_seconds is not None else settings.EMBEDDING_TIMEOUT_SECONDS
        self.timeout = httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT_SECONDS))
        self.transport = transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order. Raises on any failure: the caller decides what that means."""
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            response = await client.post(self.url, json={"model": self.model, "input": texts, "truncate": True})
            response.raise_for_status()
            vectors = response.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts) or not all(vectors):
            raise ValueError(f"The embedding server returned {len(vectors or [])} vector(s) for {len(texts)} text(s)")
        return vectors
