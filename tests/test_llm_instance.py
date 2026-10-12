"""The LLM an agent gets: settings that must travel with it into the sandbox's agent server."""

import pytest

from app.models.llmModel import LlmConfigModel
from app.utils.LlmInstance import build_llm


@pytest.mark.parametrize("native", [True, False])
def test_tool_calling_mode_travels_with_the_agent(native: bool) -> None:
    llm = build_llm(LlmConfigModel(model="ollama_chat/qwen2.5-coder:14b", native_tool_calling=native))

    # The agent runs in the sandbox's agent server: what it gets is the serialized LLM.
    assert llm.model_dump()["native_tool_calling"] is native


def test_native_tool_calls_are_the_default() -> None:
    assert build_llm(LlmConfigModel(model="openai/gpt-4o")).native_tool_calling is True
