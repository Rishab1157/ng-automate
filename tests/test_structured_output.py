import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.utils import StructuredOutput
from app.utils.StructuredOutput import complete_json

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
CONFIG = LlmConfigModel(model="openai/gpt-4o", base_url="https://gw.example/v1", api_key=SecretStr("sk-very-secret-key"), num_ctx=32768)


def _parse(reply: str) -> int:
    try:
        return int(json.loads(reply)["x"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"bad reply: {type(error).__name__}") from None


class FakeLiteLLM:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(*replies: Any) -> FakeLiteLLM:
        llm = FakeLiteLLM(*replies)
        monkeypatch.setattr(StructuredOutput, "litellm", llm)
        return llm
    return install


def test_sends_the_schema_and_returns_the_parsed_reply(fake) -> None:
    llm = fake('{"x": 7}')

    assert complete_json(CONFIG, "instructions", "text", "thing", SCHEMA, _parse) == 7

    call = llm.calls[0]
    assert call["response_format"] == {"type": "json_schema", "json_schema": {"name": "thing", "schema": SCHEMA, "strict": True}}
    assert call["model"] == "openai/gpt-4o"
    assert call["api_base"] == "https://gw.example/v1"
    assert call["api_key"] == "sk-very-secret-key"
    assert call["num_ctx"] == 32768
    assert call["temperature"] == 0


def test_retry_tells_the_model_the_exact_error(fake) -> None:
    llm = fake('{"y": 1}', '{"x": 3}')

    assert complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2) == 3

    retry_messages = llm.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"y": 1}'}
    assert "bad reply: KeyError" in retry_messages[-1]["content"]


def test_gives_up_with_the_last_reason(fake) -> None:
    fake("nope", "still nope")

    with pytest.raises(ValueError, match="bad reply"):
        complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2)


def test_provider_errors_never_leak_the_key(fake, caplog: pytest.LogCaptureFixture) -> None:
    fake(RuntimeError("401 for key sk-very-secret-key at https://gw.example/v1"), '{"x": 1}')

    with caplog.at_level(logging.DEBUG):
        assert complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2) == 1

    assert "sk-very-secret-key" not in caplog.text


def test_all_provider_errors_raise_without_the_key(fake) -> None:
    fake(RuntimeError("bad key sk-very-secret-key"))

    with pytest.raises(ValueError) as raised:
        complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=1)

    assert "sk-very-secret-key" not in str(raised.value)
    assert "RuntimeError" in str(raised.value)


# ---------- rate limits (HTTP 429) ----------

def _rate_limit(message: str = "rate limit reached for requests") -> Exception:
    from litellm.exceptions import RateLimitError

    return RateLimitError(message, llm_provider="openai", model="gpt-4o")


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The pauses complete_json makes, without waiting. Only this module's sleep: never time.sleep, which the threads
    of other tests use too."""
    made: list[float] = []
    monkeypatch.setattr(StructuredOutput, "sleep", made.append)
    return made


def test_a_rate_limited_call_is_tried_again_after_the_agents_pauses(fake, pauses: list[float]) -> None:
    # A gateway's 429 was tried again at once, so it failed again: it now waits as the agents' model calls do.
    llm = fake(_rate_limit(), _rate_limit(), '{"x": 4}')

    assert complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=1) == 4

    assert len(llm.calls) == 3
    assert pauses == [
        min(settings.LLM_RETRY_MIN_WAIT_SECONDS * 2 ** n, settings.LLM_RETRY_MAX_WAIT_SECONDS) for n in range(2)
    ]


def test_a_rate_limit_that_lasts_gives_up_after_the_agents_tries(fake, pauses: list[float]) -> None:
    llm = fake(*[_rate_limit() for _ in range(2 * settings.LLM_NUM_RETRIES)])

    with pytest.raises(ValueError, match="RateLimitError"):
        complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2)

    # No second round right after the last try: that would be the immediate retry again.
    assert len(llm.calls) == settings.LLM_NUM_RETRIES
    assert len(pauses) == settings.LLM_NUM_RETRIES - 1 and max(pauses) <= settings.LLM_RETRY_MAX_WAIT_SECONDS


def test_other_errors_and_a_used_up_quota_are_not_waited_for(fake, pauses: list[float]) -> None:
    fake(RuntimeError("connection reset"), '{"x": 1}')
    assert complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2) == 1

    llm = fake(_rate_limit("You exceeded your current quota: insufficient_quota"), '{"x": 2}')
    with pytest.raises(ValueError, match="RateLimitError"):
        complete_json(CONFIG, "i", "t", "thing", SCHEMA, _parse, attempts=2)

    assert pauses == [] and len(llm.calls) == 1  # waiting cannot bring a used-up quota back
