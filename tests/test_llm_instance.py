"""The LLM an agent gets: settings that must travel with it into the sandbox's agent server, and the check that a
self-hosted model answers before an agent or the test-data reader starts."""

import http.server
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import SecretStr

from app.config import settings
from app.models.llmModel import LlmConfigModel
from app.utils import LlmInstance
from app.utils.LlmInstance import MODEL_NOT_ANSWERING, build_llm, check_model_answers, is_self_hosted


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


def test_a_rate_limited_model_call_is_tried_again_after_growing_pauses() -> None:
    # A real run: the model gateway answered 429 and the healer stopped at once ("the model is rate limited"). Its one
    # retry came 8 s later, while such a limit usually lasts up to a minute.
    from litellm.exceptions import RateLimitError

    llm = build_llm(LlmConfigModel(model="openai/gpt-4o"))
    calls: list[int] = []

    @llm._make_retry_decorator()
    def call() -> None:
        calls.append(1)
        raise RateLimitError("rate limit reached for requests", llm_provider="openai", model="gpt-4o")

    # Only this call's own pauses are recorded. Never patch time.sleep: threads left by other tests sleep in loops,
    # and a whole test run once grew to 14 GB while they appended to this list.
    pauses: list[float] = []
    call.retry.sleep = pauses.append  # type: ignore[attr-defined]

    with pytest.raises(RateLimitError):
        call()

    assert len(calls) == settings.LLM_NUM_RETRIES >= 3
    assert pauses == sorted(pauses) and max(pauses) <= settings.LLM_RETRY_MAX_WAIT_SECONDS  # growing, and bounded
    assert sum(pauses) >= 60
    # The agent runs in the sandbox's agent server: its retry settings travel with the serialized LLM.
    dumped = llm.model_dump()
    assert (dumped["retry_min_wait"], dumped["retry_max_wait"]) == (
        settings.LLM_RETRY_MIN_WAIT_SECONDS, settings.LLM_RETRY_MAX_WAIT_SECONDS
    )


# ---------- the 1-token check: does a self-hosted model answer? ----------

@pytest.mark.parametrize(("base_url", "self_hosted"), [
    ("http://192.168.3.90:11435", True),  # the free model on the local network
    ("http://127.0.0.1:11434", True),
    ("http://localhost:11434", True),
    ("http://host.docker.internal:11434", True),
    ("http://ollama:11434", True),  # a container's name
    ("http://10.1.2.3/v1", True),
    ("https://agenticqe.ai/v1", False),  # a hosted gateway: every request is paid for
    ("https://api.openai.com/v1", False),
    ("http://20.30.40.50/v1", False),
    (None, False),  # the provider's own hosted endpoint
])
def test_only_a_model_on_this_machine_or_network_counts_as_self_hosted(base_url: str | None, self_hosted: bool) -> None:
    assert is_self_hosted(base_url) is self_hosted


@contextmanager
def _model_server(mode: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """An OpenAI-compatible model server on this machine that answers, never answers, or fails. Yields its base URL
    and the request bodies it got."""
    bodies: list[dict[str, Any]] = []
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if mode == "hangs":
                release.wait(60)  # up, but no answer: generation is stuck
                return
            if mode == "fails":
                self.send_error(500, "model crashed")
                return
            body = json.dumps({
                "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", bodies
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def _self_hosted(base_url: str) -> LlmConfigModel:
    return LlmConfigModel(model="openai/gpt-4o-mini", base_url=base_url, api_key=SecretStr("sk-local"), api_mode="chat")


def test_a_self_hosted_model_that_answers_passes_with_a_one_token_request() -> None:
    with _model_server("answers") as (url, bodies):
        assert check_model_answers(_self_hosted(url)) is None

    assert bodies and all(1 in (body.get("max_tokens"), body.get("max_completion_tokens")) for body in bodies)


def test_a_self_hosted_model_that_does_not_answer_fails_the_check_in_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real run: the model server answered HTTP at once (the sandbox's check passed), generation was stuck, and the
    # analyzer waited the full timeout on every try: 991 s, then "the model could not be reached".
    monkeypatch.setattr(settings, "MODEL_ANSWER_CHECK_SECONDS", 2)

    with _model_server("hangs") as (url, bodies):
        started = time.monotonic()
        reason = check_model_answers(_self_hosted(url))
        elapsed = time.monotonic() - started

    assert reason == MODEL_NOT_ANSWERING.format(seconds=2)
    assert bodies and elapsed < 15  # one short try, not LLM_TIMEOUT_SECONDS on every try


def test_a_self_hosted_model_that_fails_is_left_to_the_real_calls() -> None:
    # Only "no answer in time" is the check's business: any other error, the agent's own calls report it.
    with _model_server("fails") as (url, bodies):
        assert check_model_answers(_self_hosted(url)) is None

    assert bodies


def test_a_hosted_model_is_never_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_build(config: LlmConfigModel) -> None:
        raise AssertionError("a hosted model was asked: every request to it is paid for")

    monkeypatch.setattr(LlmInstance, "build_llm", must_not_build)

    assert check_model_answers(LlmConfigModel(model="openai/gpt-4o", base_url="https://agenticqe.ai/v1")) is None
    assert check_model_answers(LlmConfigModel(model="anthropic/claude-sonnet-4-5")) is None
