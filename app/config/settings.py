"""All settings, typed and validated once at startup. Values come from the environment or .env."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=True,
        extra="ignore",
    )

    # API server
    HOST: str = "127.0.0.1"
    PORT: int = 4207
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["*"]

    # Auth: tokens are issued by QXcel and signed with the shared secret
    SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"

    # MongoDB: our own database, and QXcel's (read-only) for git and model connections
    MONGO_URI: str
    MONGO_DATABASE: str = "ng_automate"
    # QXcel's server, when it is not MONGO_URI's (ours runs in docker-compose, QXcel's stays where QXcel has it).
    QXCEL_MONGO_URI: str | None = None
    QXCEL_DATABASE: str = "qxcel"

    # Project storage
    DATA_DIR: Path = Path("data")
    MAX_PROJECT_MB: int = 200
    GIT_CLONE_TIMEOUT_SECONDS: int = 300
    # Cloned when a git connection has no branch. A repo without it falls back to its own default branch.
    GIT_DEFAULT_BRANCH: str = "main"

    # Sandbox: one Docker container per agent session, running the OpenHands agent server
    SANDBOX_IMAGE: str = "ghcr.io/openhands/agent-server:1.53.0-python"
    SANDBOX_CPUS: float = 2.0
    SANDBOX_MEMORY_MB: int = 4096
    SANDBOX_STARTUP_TIMEOUT_SECONDS: int = 180
    # Writable sandboxes (runner, healer, test generator) need build tools: docker/sandbox-test/Dockerfile.
    SANDBOX_TEST_IMAGE: str = "ng-automate/sandbox-test:1.53.0"
    # Keep Maven/npm/pip downloads between runs, one cache per organization (never shared across orgs).
    SANDBOX_BUILD_CACHES: bool = True
    # Docker network for every sandbox, created with inter-container traffic turned off: a sandbox reaches the
    # internet and the host reaches its published ports, but no sandbox can reach another one.
    SANDBOX_NETWORK: str = "ngauto-sandboxes"
    # How long a live-view ticket opens new connections (an open connection stays open).
    LIVE_VIEW_TICKET_SECONDS: int = 600

    # Runs (the master agent's pipelines)
    MAX_CONCURRENT_RUNS: int = 2
    # How often a running run looks for the user's commands (message, pause, resume, stop).
    COMMAND_POLL_SECONDS: float = 1.0
    # A run paused longer than this is stopped (its timeout does not count paused time).
    MAX_PAUSE_SECONDS: int = 1800
    # The whole run (analyze, generate, tests and healing). Keep it above HEALER_TIME_BUDGET_SECONDS.
    RUN_TIMEOUT_SECONDS: int = 7200

    # Analyzer agent
    ANALYZER_MAX_ITERATIONS: int = 40
    # Rounds in which the agent corrects an answer that failed validation, before the strict-JSON formatter.
    ANALYZER_REPAIR_ATTEMPTS: int = 2
    # Tries of the strict-JSON formatter (schema-constrained LLM call), the last resort.
    STRUCTURED_OUTPUT_ATTEMPTS: int = 2

    # Runner and healer
    TEST_RUN_TIMEOUT_SECONDS: int = 1800
    # Agent steps in one heal; then its changes are checked, the tests run again and the next heal goes on.
    HEALER_MAX_ITERATIONS: int = 60
    # All healing in one run shares this time (paused time does not count). When it is used up, the test phase
    # stops and reports that the healer timed out.
    HEALER_TIME_BUDGET_SECONDS: int = 3600
    # The same failure this many runs in a row, although the healer worked on it in between: no progress, stop.
    HEALER_SAME_FAILURE_LIMIT: int = 3
    GENERATOR_MAX_ITERATIONS: int = 60
    GENERATOR_TIMEOUT_SECONDS: int = 1500
    # Times the generator is told to continue when it stops before writing any file.
    GENERATOR_NUDGES: int = 2

    # Healer memory: earlier heals of the same organization (the problem, what was tried, what came of it), found by
    # similarity in Qdrant and given to the healer as context; the healer decides what to do with them. Healing works
    # without it: when it is off, or Qdrant or the embedding server is down, the healer just gets no memories.
    HEAL_MEMORY_ENABLED: bool = True
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: SecretStr | None = None  # only when Qdrant has authentication turned on
    QDRANT_COLLECTION_NAME: str = "ngauto_heal_memories"
    QDRANT_TIMEOUT_SECONDS: int = 10
    # An Ollama server whose model only turns text into vectors. The healer's reasoning model stays LLM_MODEL (or the
    # run's model connection).
    EMBEDDING_BASE_URL: str | None = None
    EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    # Text put before a search (some models want an instruction there). Problems are compared with problems, and
    # qwen3-embedding:0.6b separated them best without one.
    EMBEDDING_QUERY_PREFIX: str = ""
    EMBEDDING_TIMEOUT_SECONDS: float = 30
    # Memories given to the healer per heal, and how similar a memory must be (cosine, 0..1). Measured with
    # qwen3-embedding:0.6b on real failures: the same problem scored 0.82-0.95, different problems 0.36-0.71.
    # Another embedding model needs its own value.
    HEAL_MEMORY_LIMIT: int = 3
    HEAL_MEMORY_MIN_SCORE: float = 0.75
    # One LLM call that reads a part of a free-form test-data source (a small local model needs minutes).
    TEST_DATA_LLM_TIMEOUT_SECONDS: int = 1200

    # Default LLM, used when a request has no model connection
    LLM_MODEL: str
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: SecretStr | None = None
    LLM_API_MODE: Literal["auto", "chat", "responses"] = "auto"
    LLM_NUM_CTX: int | None = None
    LLM_THINKING: bool = True

    @property
    def MAX_PROJECT_BYTES(self) -> int:
        return self.MAX_PROJECT_MB * 1024 * 1024


settings = Settings()  # type: ignore[call-arg]  # required values come from the environment
