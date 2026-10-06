from openhands.sdk import LLM

from ng_automate.config import LLMConfig


def build_llm(cfg: LLMConfig) -> LLM:
    """The one place that turns our config into an OpenHands LLM (LiteLLM inside)."""
    return LLM(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        api_mode=cfg.api_mode,
    )
