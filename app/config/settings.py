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

    # Sandbox: one Docker container per agent session, running OUR OpenHands agent server, built from the vendored SDK
    # in third_party/ (docker compose --profile build build agent-server): every change made there runs in sandboxes.
    SANDBOX_IMAGE: str = "ng-automate/agent-server:1.53.0-source"
    SANDBOX_CPUS: float = 2.0
    SANDBOX_MEMORY_MB: int = 4096
    SANDBOX_STARTUP_TIMEOUT_SECONDS: int = 180
    # Writable sandboxes (runner, healer, test generator) need build tools: docker/sandbox-test/Dockerfile, on the
    # image above. (ng-automate/sandbox-test:1.53.0 is the earlier one, on OpenHands' prebuilt agent server.)
    SANDBOX_TEST_IMAGE: str = "ng-automate/sandbox-test:1.53.0-source"
    # Keep Maven/npm/pip downloads between runs, one cache per organization (never shared across orgs).
    SANDBOX_BUILD_CACHES: bool = True
    # Docker network for every sandbox, created with inter-container traffic turned off: a sandbox reaches the
    # internet and the host reaches its published ports, but no sandbox can reach another one.
    SANDBOX_NETWORK: str = "ngauto-sandboxes"
    # Agents run inside a sandbox container, where 127.0.0.1 is the container itself: a model server on the
    # machine that runs the API is only reachable through this host alias (Docker Desktop gives every container this
    # name). A loopback LLM base URL is rewritten to it for sandboxed agents.
    SANDBOX_HOST_ALIAS: str = "host.docker.internal"
    # Sandboxes reach the internet (package downloads) and the model server, but nothing on a private address: not this
    # machine's services (QXcel's MongoDB, Qdrant, the API: all reachable through the host alias, also when they listen
    # on 127.0.0.1 only), not the LAN, not cloud metadata. A helper container sets the rules from outside the sandbox
    # (docker/sandbox-firewall; build it once: docker compose --profile build build sandbox-firewall) and the sandbox
    # has no NET_ADMIN, so not even root in it can lift them.
    SANDBOX_EGRESS_FIREWALL: bool = True
    SANDBOX_FIREWALL_IMAGE: str = "ng-automate/sandbox-firewall:1"
    # Private "host:port" endpoints sandboxes may still reach (JSON list), e.g. a model connection's server or a package
    # registry on the LAN. The local model's server (LLM_BASE_URL, when USE_LOCAL_LLM) is allowed without being listed.
    SANDBOX_ALLOWED_HOSTS: list[str] = []
    # How long a live-view ticket opens new connections (an open connection stays open).
    LIVE_VIEW_TICKET_SECONDS: int = 600

    # Runs (the master agent's pipelines)
    MAX_CONCURRENT_RUNS: int = 2
    # How often a running run looks for the user's commands (message, pause, resume, stop).
    COMMAND_POLL_SECONDS: float = 1.0
    # A run paused longer than this is stopped (its timeout does not count paused time).
    MAX_PAUSE_SECONDS: int = 1800
    # Run ownership: the server executing a run marks it and refreshes the mark every RUN_HEARTBEAT_SECONDS.
    # Another server (or this one after a restart) takes a run over only when the mark is older than
    # RUN_LEASE_SECONDS, i.e. its server stopped; it looks for such runs every RUN_ORPHAN_SCAN_SECONDS.
    RUN_HEARTBEAT_SECONDS: float = 10
    RUN_LEASE_SECONDS: float = 45
    RUN_ORPHAN_SCAN_SECONDS: float = 30
    # The whole run (analyze, generate, tests and healing). Keep it above HEALER_TIME_BUDGET_SECONDS.
    RUN_TIMEOUT_SECONDS: int = 7200

    # Analyzer agent
    ANALYZER_MAX_ITERATIONS: int = 40
    # Rounds in which the agent corrects an answer that failed validation, before the strict-JSON formatter.
    ANALYZER_REPAIR_ATTEMPTS: int = 2
    # Tries of the strict-JSON formatter (schema-constrained LLM call), the last resort.
    STRUCTURED_OUTPUT_ATTEMPTS: int = 2
    # Longest answer of one schema-constrained call (the test-data reader's, the analyzer's formatter's). Without a cap
    # Ollama writes until its context is full: a model that repeated itself took 20 min per try (the timeout).
    STRUCTURED_OUTPUT_MAX_TOKENS: int = 8192
    # Longest string in such an answer from an Ollama model, which keeps to it while writing (JSON schema maxLength): a
    # model that made up "the 31 letters" a step asked for wrote 1234567890... until it was cut off. Now the string ends
    # here and the answer goes on (the test-data reader then drops the made-up value: it is not in the source).
    STRUCTURED_OUTPUT_MAX_STRING_CHARS: int = 2000

    # Runner and healer
    TEST_RUN_TIMEOUT_SECONDS: int = 1800
    # Agent steps in one heal; then its changes are checked, the tests run again and the next heal goes on.
    HEALER_MAX_ITERATIONS: int = 60
    # All healing in one run shares this time (paused time does not count). When it is used up, the test phase
    # stops and reports that the healer timed out.
    HEALER_TIME_BUDGET_SECONDS: int = 3600
    # The same failure this many runs in a row, although the healer worked on it in between: no progress, stop.
    HEALER_SAME_FAILURE_LIMIT: int = 3
    # Locator failures (an element not found) go to the healer too: it looks the element up on the live page and
    # marks every locator it changes for review. False: they are reported to the user, like failed assertions.
    HEAL_LOCATOR_FAILURES: bool = True
    GENERATOR_MAX_ITERATIONS: int = 60
    # Time for one batch of cases (see GENERATOR_BATCH_SIZE), paused time not counted.
    GENERATOR_TIMEOUT_SECONDS: int = 1500
    # Cases the generator gets at a time, in one conversation. A small local model given all cases at once tries to
    # write them in one huge reply and runs out of time; a few at a time it writes steadily.
    GENERATOR_BATCH_SIZE: int = 3
    # After each batch the tests are built without running them (the stack's compile/collect step); while they do not
    # build, the compiler's errors go back to the generator, this many times per batch.
    GENERATOR_FIX_ROUNDS: int = 3
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

    # true: every agent uses the local model below (LLM_*), and a request's model_connection_id is ignored. false:
    # every request whose work needs a model must name one of the organization's QXcel model connections
    # (the model_connection_id query parameter), and the local model is never used.
    USE_LOCAL_LLM: bool = True

    # The local LLM (USE_LOCAL_LLM)
    LLM_MODEL: str
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: SecretStr | None = None
    LLM_API_MODE: Literal["auto", "chat", "responses"] = "auto"
    LLM_NUM_CTX: int | None = None
    LLM_THINKING: bool = True
    LLM_NATIVE_TOOL_CALLING: bool = True
    # Limits on every model call the agents make. Without a length limit a local model caught in a repetition loop
    # writes until its context is full, for minutes, and holds the model server (Ollama answers one request at a time)
    # while every other agent waits silently.
    LLM_MAX_OUTPUT_TOKENS: int = 4096
    # Seconds one model call may take, and how many times a failing call is made in all (the SDK counts the first
    # call). The SDK's own defaults (300 s, 5 tries) kept an agent silent for many minutes when the model server was
    # unreachable.
    LLM_TIMEOUT_SECONDS: int = 300
    LLM_NUM_RETRIES: int = 3
    # The pause before each new try: it doubles from the first value up to the second. A model gateway that rate-limits
    # (HTTP 429) usually lets calls through again within a minute; with 2 tries and the SDK's 8 s pause, the healer
    # stopped at once.
    LLM_RETRY_MIN_WAIT_SECONDS: int = 20
    LLM_RETRY_MAX_WAIT_SECONDS: int = 60
    # Seconds a self-hosted model may take to answer a 1-token request, asked before an agent starts or test data is
    # read with it. Its server answers HTTP at once also when generation is stuck: a real analysis then waited the full
    # LLM_TIMEOUT_SECONDS on every try (991 s) and a text source stayed "processing" for 20+ minutes. 0: no check.
    MODEL_ANSWER_CHECK_SECONDS: int = 90

    @property
    def MAX_PROJECT_BYTES(self) -> int:
        return self.MAX_PROJECT_MB * 1024 * 1024


settings = Settings()  # type: ignore[call-arg]  # required values come from the environment
