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


def test_a_loopback_model_url_is_rewritten_for_the_sandbox() -> None:
    # The sandbox is a container, so 127.0.0.1 there is the container itself: the model server on the machine
    # that runs the API is reached through the host-gateway alias the sandbox is started with.
    llm = build_llm(
        LlmConfigModel(model="ollama_chat/qwen2.5-coder:14b", base_url="http://127.0.0.1:11435")
    )

    assert llm.model_dump()["base_url"] == "http://host.docker.internal:11435"


def test_a_hosted_model_url_is_left_alone() -> None:
    llm = build_llm(LlmConfigModel(model="openai/gpt-4o", base_url="https://agenticqe.ai/v1"))

    assert llm.model_dump()["base_url"] == "https://agenticqe.ai/v1"
