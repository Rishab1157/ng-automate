"""Token usage: the model, pricing input and output apart, reading it from an agent conversation and a host-side call,
counting it per stage of a run, the end-of-run event, the API response and the test-data record. A failure to read
or save the numbers never fails a run. No LLM: prices come from a fake price list unless a test says otherwise.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bson import ObjectId
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.llm.utils.metrics import Metrics
from pydantic import SecretStr

from app.core.exceptions import AnalysisError
from app.dto.runDto.responseDto import RunResponseDTO
from app.models.llmModel import LlmConfigModel
from app.models.runModel import RunEventType, RunMapper, RunOutputsModel, RunStatus, RunTokenUsageModel
from app.models.testDataModel import TestDataMapper
from app.models.tokenUsageModel import TokenUsageModel
from app.services.testDataService import TestDataProcessor
from app.utils import StructuredOutput, TokenCost
from app.utils.AgentConversationUtils import close_conversation, read_token_usage
from app.utils.TokenCost import priced
from tests.test_master_graph import FakeAnalyzer, _analysis_state
from tests.test_master_graph import _setup as _analysis_setup
from tests.test_master_test_phase import DEPENDENCY, FIX, PASSED, _setup, _test_state
from tests.test_run_executor import FakeGraph, FakeRunService, make_executor, make_run, settle
from tests.test_test_data_processor import DATA_SET, FakeConnections, FakeExtractor, FakeService, _upload

MODEL = "openai/gpt-test"


class FakePrices:
    """LiteLLM's cost_per_token with a fixed price list: input $1/M (cached $0.50/M), output $2/M."""

    def __init__(self, known: bool = True) -> None:
        self.known = known
        self.calls: list[dict[str, Any]] = []

    def cost_per_token(self, **kwargs: Any) -> tuple[float, float]:
        self.calls.append(kwargs)
        if not self.known:
            raise Exception("This model isn't mapped yet")
        cached = kwargs["cache_read_input_tokens"]
        return (kwargs["prompt_tokens"] - cached) * 1e-6 + cached * 0.5e-6, kwargs["completion_tokens"] * 2e-6


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> FakePrices:
    fake = FakePrices()
    monkeypatch.setattr(TokenCost, "litellm", fake)
    return fake


def _usage(input_tokens: int, output_tokens: int, cost: bool = True, **extra: Any) -> TokenUsageModel:
    costs = {"input_cost_usd": input_tokens * 1e-6, "output_cost_usd": output_tokens * 2e-6} if cost else {}
    return TokenUsageModel(input_tokens=input_tokens, output_tokens=output_tokens, calls=1, **costs, **extra)


# --------------------------------------------------------------------------- the model


def test_usages_add_up_field_by_field() -> None:
    first = TokenUsageModel(input_tokens=1000, cached_input_tokens=400, cache_write_tokens=5, output_tokens=50,
                            reasoning_tokens=7, calls=2, input_cost_usd=0.5, output_cost_usd=0.25)
    second = TokenUsageModel(input_tokens=10, output_tokens=1, calls=1, input_cost_usd=0.25, output_cost_usd=0.125)

    total = first.add(second)

    assert total.model_dump() == {
        "input_tokens": 1010, "cached_input_tokens": 400, "cache_write_tokens": 5, "output_tokens": 51,
        "reasoning_tokens": 7, "calls": 3, "input_cost_usd": 0.75, "output_cost_usd": 0.375, "total_cost_usd": 1.125,
    }
    assert first.add(None) is first


def test_an_unknown_count_or_price_does_not_hide_the_known_one() -> None:
    priced_part = TokenUsageModel(input_tokens=10, input_cost_usd=0.2, output_cost_usd=0.1)
    unpriced_part = TokenUsageModel(input_tokens=5, calls=1)

    total = priced_part.add(unpriced_part)

    assert (total.calls, total.input_cost_usd, total.output_cost_usd) == (1, 0.2, 0.1)
    assert TokenUsageModel().total_cost_usd is None


def test_description_keeps_input_and_output_apart() -> None:
    usage = TokenUsageModel(input_tokens=1_234_567, cached_input_tokens=300_000, output_tokens=45_000,
                            input_cost_usd=2.4, output_cost_usd=0.45)

    assert usage.describe() == "input 1.2M (cached 300k) ~$2.40, output 45k ~$0.45, total ~$2.85"
    assert TokenUsageModel(input_tokens=950, output_tokens=12).describe() == "input 950, output 12"
    free = TokenUsageModel(input_tokens=2000, output_tokens=10, input_cost_usd=0.0, output_cost_usd=0.0)
    assert free.describe() == "input 2k $0, output 10 $0, total $0"
    assert TokenUsageModel(input_tokens=1500, input_cost_usd=0.0042).describe().startswith("input 1.5k ~$0.0042")


def test_run_usage_is_kept_per_stage_and_in_all() -> None:
    usage = RunTokenUsageModel()

    usage = usage.add("analyze", _usage(100, 10)).add("analyze", _usage(50, 5))  # a resumed analysis counts again
    usage = usage.add("heal", _usage(1000, 100)).add("heal", _usage(2000, 200))

    assert usage.analyze is not None and (usage.analyze.input_tokens, usage.analyze.calls) == (150, 2)
    assert usage.generate is None
    assert [heal.input_tokens for heal in usage.heals] == [1000, 2000]
    assert (usage.total.input_tokens, usage.total.output_tokens, usage.total.calls) == (3150, 315, 4)
    assert usage.total.input_cost_usd == pytest.approx(3150e-6)
    assert usage.total.output_cost_usd == pytest.approx(630e-6)


# --------------------------------------------------------------------------- pricing


def test_input_and_output_are_priced_apart_and_cached_input_is_cheaper(prices: FakePrices) -> None:
    usage = priced(TokenUsageModel(input_tokens=1000, cached_input_tokens=400, output_tokens=100), MODEL)

    assert usage.input_cost_usd == pytest.approx(600e-6 + 200e-6)
    assert usage.output_cost_usd == pytest.approx(200e-6)
    assert prices.calls == [{"model": MODEL, "prompt_tokens": 1000, "completion_tokens": 100,
                             "cache_read_input_tokens": 400, "cache_creation_input_tokens": 0}]


def test_a_model_without_a_price_keeps_its_counts_and_has_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TokenCost, "litellm", FakePrices(known=False))

    usage = priced(TokenUsageModel(input_tokens=1000, output_tokens=100), "openai/local-model")

    assert (usage.input_tokens, usage.input_cost_usd, usage.output_cost_usd, usage.total_cost_usd) == (1000, None, None, None)
    assert "$" not in usage.describe()


def test_litellm_prices_a_known_model_and_not_an_unknown_one() -> None:
    known = priced(TokenUsageModel(input_tokens=1000, output_tokens=100), "openai/gpt-4o")
    unknown = priced(TokenUsageModel(input_tokens=1000, output_tokens=100), "openai/no-such-model-for-ng-automate")

    assert known.input_cost_usd and known.output_cost_usd and known.output_cost_usd != known.input_cost_usd
    assert unknown.input_cost_usd is None and unknown.output_cost_usd is None


# --------------------------------------------------------------------------- reading it from a conversation


class FakeStatsConversation:
    def __init__(self, stats: Any = None, stats_error: Exception | None = None, refresh_error: Exception | None = None) -> None:
        self._stats, self.stats_error, self.refresh_error = stats, stats_error, refresh_error
        self.refreshes = 0
        self.closed = False
        self.state = SimpleNamespace(refresh_from_server=self._refresh)

    def _refresh(self) -> None:
        self.refreshes += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    @property
    def conversation_stats(self) -> Any:
        if self.stats_error is not None:
            raise self.stats_error
        return self._stats

    def close(self) -> None:
        self.closed = True


def _stats(*calls: tuple[int, int, int]) -> ConversationStats:
    """An agent and a condenser LLM, the calls (input, cached, output) split between them."""
    agent, condenser = Metrics(model_name=MODEL), Metrics(model_name=MODEL)
    for number, (prompt, cached, completion) in enumerate(calls):
        (agent if number % 2 == 0 else condenser).add_token_usage(
            prompt_tokens=prompt, completion_tokens=completion, cache_read_tokens=cached, cache_write_tokens=0,
            context_window=128_000, response_id=f"r{number}", reasoning_tokens=1,
        )
    return ConversationStats(usage_to_metrics={"agent": agent, "condenser": condenser})


def test_a_conversation_is_read_call_by_call_before_it_closes(prices: FakePrices) -> None:
    conversation = FakeStatsConversation(_stats((1000, 400, 100), (2000, 0, 50), (500, 500, 10)))

    usage = close_conversation(conversation, MODEL)  # type: ignore[arg-type]

    assert conversation.closed and conversation.refreshes == 1
    assert usage is not None
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.reasoning_tokens, usage.calls) == (
        3500, 900, 160, 3, 3
    )
    assert len(prices.calls) == 3  # each call priced on its own
    assert usage.input_cost_usd == pytest.approx((600 + 200 + 2000 + 250) * 1e-6)
    assert usage.output_cost_usd == pytest.approx(320e-6)


def test_totals_only_are_priced_as_they_are(prices: FakePrices) -> None:
    metrics = Metrics(model_name=MODEL)
    metrics.add_token_usage(prompt_tokens=800, completion_tokens=20, cache_read_tokens=0, cache_write_tokens=0,
                            context_window=0, response_id="r")
    snapshot = SimpleNamespace(get_combined_metrics=lambda: SimpleNamespace(
        token_usages=[], accumulated_token_usage=metrics.accumulated_token_usage
    ))

    usage = read_token_usage(FakeStatsConversation(snapshot), MODEL)  # type: ignore[arg-type]

    assert usage is not None and (usage.input_tokens, usage.output_tokens, usage.calls) == (800, 20, None)
    assert usage.input_cost_usd == pytest.approx(800e-6)


def test_unreadable_numbers_never_stop_the_close(prices: FakePrices, caplog: pytest.LogCaptureFixture) -> None:
    conversation = FakeStatsConversation(stats_error=ConnectionError("agent server is gone"),
                                         refresh_error=ConnectionError("agent server is gone"))

    assert close_conversation(conversation, MODEL) is None  # type: ignore[arg-type]
    assert conversation.closed
    assert "Could not read the agent's token usage" in caplog.text
    assert read_token_usage(SimpleNamespace(), MODEL) is None  # type: ignore[arg-type]  # no stats at all


# --------------------------------------------------------------------------- host-side calls


class FakeLiteLLM:
    def completion(self, **kwargs: Any) -> Any:
        usage = SimpleNamespace(prompt_tokens=300, completion_tokens=40,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=100), completion_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"x": 1}'))], usage=usage)


def test_a_structured_output_call_reports_what_it_used(monkeypatch: pytest.MonkeyPatch, prices: FakePrices) -> None:
    monkeypatch.setattr(StructuredOutput, "litellm", FakeLiteLLM())
    used: list[TokenUsageModel] = []
    config = LlmConfigModel(model=MODEL, api_key=SecretStr("sk-test-key-123"))

    StructuredOutput.complete_json(config, "i", "t", "thing", {"type": "object"}, json.loads, on_usage=used.append)

    [usage] = used
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.calls) == (300, 100, 40, 1)
    assert usage.input_cost_usd == pytest.approx(250e-6) and usage.output_cost_usd == pytest.approx(80e-6)


# --------------------------------------------------------------------------- per stage of a run


@pytest.mark.anyio
async def test_each_heal_is_added_to_the_runs_usage(tmp_path: Path) -> None:
    heal_usage = _usage(5000, 300)
    setup = _setup(tmp_path, [DEPENDENCY, PASSED], [FIX.model_copy(update={"token_usage": heal_usage})])
    earlier = RunTokenUsageModel().add("analyze", _usage(1000, 50))

    update = await setup.nodes.test_and_heal(_test_state(tmp_path, token_usage=earlier.model_dump(mode="json")))

    saved = RunTokenUsageModel.model_validate(setup.runs.outputs["token_usage"])
    assert saved.analyze == earlier.analyze and saved.heals == [heal_usage]
    assert (saved.total.input_tokens, saved.total.output_tokens, saved.total.calls) == (6000, 350, 2)
    assert update["token_usage"] == saved.model_dump(mode="json")
    # Each heal keeps its own numbers too (GET /run/{id}/tests).
    assert setup.runs.outputs["test_attempts"][0].heal.token_usage == heal_usage


@pytest.mark.anyio
async def test_a_failed_save_of_the_usage_does_not_stop_the_heal_loop(tmp_path: Path) -> None:
    setup = _setup(tmp_path, [DEPENDENCY, PASSED], [FIX.model_copy(update={"token_usage": _usage(10, 1)})])
    save_outputs = setup.runs.save_outputs

    async def failing_save(run_id: str, **outputs: Any) -> None:
        if "token_usage" in outputs:
            raise RuntimeError("database is down")
        await save_outputs(run_id, **outputs)

    setup.runs.save_outputs = failing_save  # type: ignore[method-assign]

    update = await setup.nodes.test_and_heal(_test_state(tmp_path))

    assert update["test_report"]["stop_reason"] == "passed"
    assert "token_usage" not in update and "token_usage" not in setup.runs.outputs


@dataclass
class CountingAnalyzer(FakeAnalyzer):
    token_usage: TokenUsageModel | None = None


@pytest.mark.anyio
async def test_the_analysis_usage_is_counted_also_when_it_fails(tmp_path: Path) -> None:
    usage = _usage(700, 70)
    done = _analysis_setup(tmp_path / "ok", analyzer=CountingAnalyzer(events=1, token_usage=usage))
    failed = _analysis_setup(tmp_path / "failed", analyzer=CountingAnalyzer(
        events=1, error=AnalysisError("The analysis failed: no answer"), token_usage=usage
    ))

    update = await done.nodes.analyze(_analysis_state(tmp_path / "ok"))
    with pytest.raises(AnalysisError):
        await failed.nodes.analyze(_analysis_state(tmp_path / "failed"))

    for setup in (done, failed):
        saved = RunTokenUsageModel.model_validate(setup.runs.outputs["token_usage"])
        assert saved.analyze == usage and saved.total.input_tokens == 700
    assert update["token_usage"]["analyze"]["input_tokens"] == 700


# --------------------------------------------------------------------------- end of the run, API response


@pytest.mark.anyio
async def test_the_run_ends_with_one_token_usage_event_before_it_is_marked_finished() -> None:
    usage = RunTokenUsageModel().add("analyze", TokenUsageModel(
        input_tokens=1_200_000, cached_input_tokens=300_000, output_tokens=45_000, input_cost_usd=2.4,
        output_cost_usd=0.45,
    ))
    run = make_run(outputs=RunOutputsModel(token_usage=usage))
    runs = FakeRunService(run)
    seen_status: list[RunStatus] = []
    append_event = runs.append_event

    async def recording_append(run_id: str, org_id: str, type: RunEventType, message: str, **kwargs: Any) -> None:
        seen_status.append(runs.runs[run_id].status)
        await append_event(run_id, org_id, type, message, **kwargs)

    runs.append_event = recording_append  # type: ignore[method-assign]
    executor, _ = make_executor(runs, FakeGraph())

    await executor.submit(run.id)
    await settle(executor)

    assert [event[1] for event in runs.events] == [RunEventType.TOKEN_USAGE, RunEventType.RUN_COMPLETED]
    _, _, message, data = runs.events[0]
    assert message == "Tokens: input 1.2M (cached 300k) ~$2.40, output 45k ~$0.45, total ~$2.85"
    assert data["total"]["total_cost_usd"] == pytest.approx(2.85) and data["analyze"]["input_tokens"] == 1_200_000
    assert seen_status[0] == RunStatus.RUNNING  # still running: a client reading the timeline gets it


@pytest.mark.anyio
async def test_a_failed_run_reports_its_usage_and_one_without_usage_reports_nothing() -> None:
    async def fail(state: dict[str, Any]) -> dict[str, Any]:
        raise AnalysisError("The analysis failed: no answer")

    counted = make_run(outputs=RunOutputsModel(token_usage=RunTokenUsageModel().add("analyze", _usage(10, 1))))
    uncounted = make_run()
    runs = FakeRunService(counted, uncounted)
    executor, _ = make_executor(runs, FakeGraph(fail))

    await executor.submit(counted.id)
    await executor.submit(uncounted.id)
    await settle(executor)

    assert [e[1] for e in runs.events if e[0] == counted.id] == [RunEventType.TOKEN_USAGE, RunEventType.RUN_FAILED]
    assert [e[1] for e in runs.events if e[0] == uncounted.id] == [RunEventType.RUN_FAILED]


@pytest.mark.anyio
async def test_a_usage_report_that_fails_does_not_fail_the_run() -> None:
    run = make_run(outputs=RunOutputsModel(token_usage=RunTokenUsageModel().add("heal", _usage(10, 1))))
    runs = FakeRunService(run)
    append_event = runs.append_event

    async def failing_append(run_id: str, org_id: str, type: RunEventType, message: str, **kwargs: Any) -> None:
        if type == RunEventType.TOKEN_USAGE:
            raise RuntimeError("database is down")
        await append_event(run_id, org_id, type, message, **kwargs)

    runs.append_event = failing_append  # type: ignore[method-assign]
    executor, _ = make_executor(runs, FakeGraph())

    await executor.submit(run.id)
    await settle(executor)

    assert runs.runs[run.id].status == RunStatus.COMPLETED
    assert [event[1] for event in runs.events] == [RunEventType.RUN_COMPLETED]


def test_the_run_api_returns_the_saved_usage() -> None:
    usage = RunTokenUsageModel().add("generate", _usage(4000, 400)).add("heal", _usage(100, 10))
    now = datetime.now(UTC)
    doc = {
        "_id": ObjectId(), "org_id": ObjectId(), "project_id": ObjectId(), "created_by": ObjectId(), "mode": "generate",
        "status": "completed", "stage": "completed", "created_at": now, "updated_at": now,
        "outputs": {"token_usage": usage.model_dump(mode="json")},  # as stored, with the computed totals
    }

    body = RunResponseDTO.from_model(RunMapper.to_model(doc)).model_dump(mode="json")

    assert body["token_usage"]["generate"]["input_tokens"] == 4000
    assert body["token_usage"]["heals"][0]["output_cost_usd"] == pytest.approx(20e-6)
    assert body["token_usage"]["total"]["total_cost_usd"] == pytest.approx(4100e-6 + 820e-6)
    assert RunResponseDTO.from_model(RunMapper.to_model({**doc, "outputs": {}})).token_usage is None


# --------------------------------------------------------------------------- test-data reading


class CountingExtractor(FakeExtractor):
    token_usage = TokenUsageModel(input_tokens=900, output_tokens=90, calls=2)


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", [DATA_SET, ValueError("no test cases were found in the source")])
async def test_a_test_data_reading_stores_its_usage_ready_or_failed(outcome: Any) -> None:
    service = FakeService(_upload("t1"))
    processor = TestDataProcessor(
        test_data_service=service,  # type: ignore[arg-type]
        model_connection_service=FakeConnections(),  # type: ignore[arg-type]
        extractor_factory=CountingExtractor(outcome),  # type: ignore[arg-type]
    )

    await processor.submit("t1")
    await asyncio.gather(*list(processor._tasks.values()), return_exceptions=True)

    assert service.token_usage == {"t1": CountingExtractor.token_usage}
    assert ("t1" in service.ready) == (outcome is DATA_SET)


def test_the_test_data_record_reads_its_usage() -> None:
    now = datetime.now(UTC)
    doc = {"_id": ObjectId(), "org_id": ObjectId(), "project_id": ObjectId(), "created_by": ObjectId(),
           "filename": "cases.txt", "source_format": "text", "status": "ready", "created_at": now,
           "token_usage": TokenUsageModel(input_tokens=5, calls=1).model_dump(mode="json")}

    assert TestDataMapper.to_model(doc).token_usage == TokenUsageModel(input_tokens=5, calls=1)
    assert TestDataMapper.to_model({**doc, "token_usage": None}).token_usage is None
