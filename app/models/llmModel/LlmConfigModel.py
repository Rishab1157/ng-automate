from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr


class LlmConfigModel(BaseModel):
    """Everything an agent needs to talk to one model."""

    model_config = ConfigDict(frozen=True)

    model: str
    base_url: str | None = None
    api_key: SecretStr | None = None
    # "chat" = Chat Completions API, "responses" = OpenAI Responses API, "auto" = the SDK decides.
    # Gateways that only allow /chat/completions need "chat".
    api_mode: Literal["auto", "chat", "responses"] = "auto"
    # Ollama only: context window in tokens. Ollama's default (4096) is too small for agents.
    num_ctx: int | None = None
    # False for models without a "thinking" mode (e.g. Devstral); the SDK asks for it by default.
    thinking: bool = True
