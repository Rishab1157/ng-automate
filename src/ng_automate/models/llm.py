from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMConfig:
    """Everything needed to talk to one model."""

    model: str
    base_url: str | None = None
    # Local models (Ollama) need no key. repr=False keeps it out of prints and logs.
    api_key: str | None = field(default=None, repr=False)
    # "chat" = Chat Completions API, "responses" = OpenAI Responses API, "auto" = SDK decides.
    # Gateways that only allow /chat/completions need "chat".
    api_mode: str = "auto"
    # Ollama only: context window in tokens. Ollama's default (4096) is too small for agents.
    num_ctx: int | None = None
    # False for models without a "thinking" mode (e.g. Devstral); the SDK asks for it by default.
    thinking: bool = True
