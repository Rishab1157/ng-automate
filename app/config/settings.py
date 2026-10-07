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
    QXCEL_DATABASE: str = "qxcel"

    # Project storage
    DATA_DIR: Path = Path("data")
    MAX_PROJECT_MB: int = 200
    GIT_CLONE_TIMEOUT_SECONDS: int = 300

    # Sandbox: one Docker container per agent session, running the OpenHands agent server
    SANDBOX_IMAGE: str = "ghcr.io/openhands/agent-server:1.53.0-python"
    SANDBOX_CPUS: float = 2.0
    SANDBOX_MEMORY_MB: int = 4096
    SANDBOX_STARTUP_TIMEOUT_SECONDS: int = 180

    # Runs (the master agent's pipelines)
    MAX_CONCURRENT_RUNS: int = 2
    RUN_TIMEOUT_SECONDS: int = 3600

    # Analyzer agent
    ANALYZER_MAX_ITERATIONS: int = 40

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
