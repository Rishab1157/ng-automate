"""The healer's memory: the problem text, judging a heal, saving and finding memories, organization isolation, and
healing without memory when its servers are down.

Qdrant runs in memory (qdrant-client's local mode: real filters and scores, no server) and the embeddings are
bag-of-words vectors: texts that share words are similar, like real embeddings. No network.
"""

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from app.agents.HealerAgent.HealerPrompts import build_heal_prompt
from app.agents.HealerAgent.HealLoopPolicy import judge_heal
from app.config import settings
from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource
from app.models.healerModel import FileChangeModel, HealOutcomeModel
from app.models.healMemoryModel import (
    HealMemoryDbModel,
    HealMemoryMapper,
    HealMemoryResult,
    describe_problem,
    describe_run,
    memory_point_id,
)
from app.models.runModel import TestAttemptModel
from app.models.testRunModel import (
    HEALABLE_KINDS,
    FailureClassificationModel,
    FailureKind,
    TestCommandModel,
    TestRunResultModel,
)
from app.repositories.healMemoryRepository import HealMemoryRepository
from app.services.healMemoryService import HealMemoryService
from app.utils.EmbeddingClient import OllamaEmbeddings

# The local mode has no payload indexes (the server has them); filters work the same.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

ORG_A, ORG_B, ORG_C = "6a0d959bdc11660be6b2ae63", "6a0d959bdc11660be6b2ae64", "6a0d959bdc11660be6b2ae65"
STACK = "Java · Maven · TestNG · Selenium"
COMMAND = TestCommandModel(tool="maven", command="mvn -B -ntp test", report_globs=["target/surefire-reports/TEST-*.xml"])

# Copied from a real `mvn -B -ntp test` run in the test sandbox.
MAVEN_TESTNG_OUTPUT = """[INFO] Building shop-ui-tests 1.0.0
[WARNING] The POM for org.testng:testng:jar:99.0.0 is missing, no dependency information available
[INFO] BUILD FAILURE
[INFO] Total time:  0.418 s
[ERROR] Failed to execute goal on project shop-ui-tests: Could not resolve dependencies for project com.demo:shop-ui-tests:jar:1.0.0
[ERROR] dependency: org.testng:testng:jar:99.0.0 (test)
[ERROR] 	org.testng:testng:jar:99.0.0 was not found in https://repo.maven.apache.org/maven2 during a previous attempt.
[ERROR] -> [Help 1]"""
MAVEN_SELENIUM_OUTPUT = """[INFO] Building web-tests 2.1
[WARNING] The POM for org.seleniumhq.selenium:selenium-java:jar:9.9.9 is missing, no dependency information available
[INFO] BUILD FAILURE
[INFO] Total time:  1.734 s
[ERROR] Failed to execute goal on project web-tests: Could not resolve dependencies for project com.acme:web-tests:jar:2.1
[ERROR] dependency: org.seleniumhq.selenium:selenium-java:jar:9.9.9 (test)
[ERROR] 	org.seleniumhq.selenium:selenium-java:jar:9.9.9 was not found in https://repo.maven.apache.org/maven2 during a previous attempt.
[ERROR] -> [Help 1]"""
CHROMEDRIVER_OUTPUT = """org.openqa.selenium.SessionNotCreatedException: Could not start a new session.
Response code 500. Message: session not created: This version of ChromeDriver only supports Chrome version 114
Current browser version is 129.0.6668.58 with binary path /usr/bin/chromium"""


def _result(kind: FailureKind, evidence: list[str], output: str = "", passed: int = 0) -> TestRunResultModel:
    return TestRunResultModel(
        command=COMMAND.command, exit_code=0 if kind == FailureKind.PASSED else 1, duration_seconds=3.0,
        total=passed, passed=passed, failed=0, errors=0, skipped=0, output_tail=output,
        classification=FailureClassificationModel(
            kind=kind, healable=kind in HEALABLE_KINDS, reason=f"{kind.value} reason", evidence=evidence
        ),
    )


MAVEN_TESTNG = _result(
    FailureKind.DEPENDENCY_FAILURE,
    ["[ERROR] Failed to execute goal on project shop-ui-tests: Could not resolve dependencies for project com.demo:shop-ui-tests:jar:1.0.0"],
    MAVEN_TESTNG_OUTPUT,
)
MAVEN_SELENIUM = _result(
    FailureKind.DEPENDENCY_FAILURE,
    ["[ERROR] Failed to execute goal on project web-tests: Could not resolve dependencies for project com.acme:web-tests:jar:2.1"],
    MAVEN_SELENIUM_OUTPUT,
)
CHROMEDRIVER = _result(
    FailureKind.ENVIRONMENT_FAILURE,
    ["org.openqa.selenium.SessionNotCreatedException: Could not start a new session."],
    CHROMEDRIVER_OUTPUT,
)
PASSED = _result(FailureKind.PASSED, [], passed=2)
BUILD = _result(FailureKind.BUILD_FAILURE, ["LoginPage.java:[12,8] cannot find symbol"])
FIX = HealOutcomeModel(
    summary="TestNG 99.0.0 does not exist: set it to 7.10.2 in pom.xml.",
    changes=[FileChangeModel(path="pom.xml", change="modified", diff="-<version>99.0.0</version>\n+<version>7.10.2</version>")],
)


class FakeEmbeddings:
    """Bag-of-words vectors. An instruction before "Query:" is ignored, as an instruction-aware model does."""

    DIMENSIONS = 512

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise httpx.ConnectError("connection refused")
        return [self._vector(text.split("Query:", 1)[-1]) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for word in re.findall(r"[a-z]{2,}", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@pytest.fixture(autouse=True)
def memory_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # The tests' own values, whatever .env says (fake embeddings score differently from a real model).
    monkeypatch.setattr(settings, "HEAL_MEMORY_MIN_SCORE", 0.5)
    monkeypatch.setattr(settings, "HEAL_MEMORY_LIMIT", 3)
    monkeypatch.setattr(settings, "EMBEDDING_QUERY_PREFIX", "Instruct: find the same problem\nQuery:")


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def memory(qdrant: AsyncQdrantClient, embeddings: FakeEmbeddings) -> HealMemoryService:
    return HealMemoryService(HealMemoryRepository(qdrant, "test_heal_memories"), embeddings, enabled=True)


def _memory(org_id: str, run_id: str, failure: TestRunResultModel = MAVEN_TESTNG,
            after: TestRunResultModel | None = PASSED, heal: HealOutcomeModel = FIX,
            embedding_model: str | None = None) -> tuple[str, HealMemoryDbModel]:
    attempt = TestAttemptModel(number=1, result=failure, heal=heal)
    next_attempt = TestAttemptModel(number=2, result=after) if after is not None else None
    memory = HealMemoryMapper.to_db_model(
        org_id=org_id, project_id="6a0d959bdc11660be6b2ae70", run_id=run_id, attempt=attempt, next_attempt=next_attempt,
        result=judge_heal(attempt, next_attempt), stack=STACK, test_command=COMMAND.command,
        embedding_model=embedding_model or settings.EMBEDDING_MODEL,
    )
    return memory_point_id(run_id, attempt.number), memory


# ---------------------------------------------------------------- the problem text and the judgement


def test_problem_text_is_the_kind_and_the_lines_that_decided_it() -> None:
    assert describe_problem(MAVEN_TESTNG).splitlines() == [
        "dependency_failure",
        "[ERROR] Failed to execute goal on project shop-ui-tests: Could not resolve dependencies for project "
        "com.demo:shop-ui-tests:jar:1.0.0",
    ]


def test_without_deciding_lines_the_first_error_lines_of_the_output_are_used() -> None:
    output = "[INFO] Building\n" + "\n".join(f"ERROR number {n}" for n in range(50))

    text = describe_problem(_result(FailureKind.UNKNOWN, [], output))

    assert text.splitlines()[0] == "unknown"
    assert "ERROR number 7" in text and "ERROR number 8" not in text and "[INFO] Building" not in text


@pytest.mark.parametrize(
    ("after", "heal", "expected"),
    [
        (PASSED, FIX, HealMemoryResult.FIXED),
        (MAVEN_TESTNG, FIX, HealMemoryResult.NOT_FIXED),
        (BUILD, FIX, HealMemoryResult.CHANGED),
        (None, FIX.model_copy(update={"blocker": "no network"}), HealMemoryResult.BLOCKED),
        (None, FIX, HealMemoryResult.UNFINISHED),
    ],
)
def test_what_a_heal_achieved_comes_from_the_run_after_it(
    after: TestRunResultModel | None, heal: HealOutcomeModel, expected: HealMemoryResult
) -> None:
    attempt = TestAttemptModel(number=1, result=MAVEN_TESTNG, heal=heal)
    next_attempt = TestAttemptModel(number=2, result=after) if after is not None else None

    assert judge_heal(attempt, next_attempt) == expected


def test_the_stored_payload_reads_back_as_the_same_memory() -> None:
    point_id, stored = _memory(ORG_A, "run-1")

    memory = HealMemoryMapper.to_model(point_id, json.loads(stored.model_dump_json()), 0.9)

    assert (memory.id, memory.org_id, memory.result, memory.succeeded, memory.score) == (
        point_id, ORG_A, HealMemoryResult.FIXED, True, 0.9
    )
    assert memory.changed_files == ["pom.xml"] and "+<version>7.10.2</version>" in memory.diff
    assert memory.next_run == describe_run(PASSED) == "the tests passed (2/2)"
    assert memory_point_id("run-1", 1) == point_id  # the same heal always gets the same id


# ---------------------------------------------------------------- saving and finding


@pytest.mark.anyio
async def test_a_saved_heal_is_found_for_a_similar_problem(memory: HealMemoryService) -> None:
    assert await memory.remember(*_memory(ORG_A, "run-1"))

    [found] = await memory.recall(org_id=ORG_A, run_id="run-2", result=MAVEN_SELENIUM)

    assert found.result == HealMemoryResult.FIXED and found.succeeded
    assert found.fix_summary == FIX.summary and found.changed_files == ["pom.xml"]
    assert found.score is not None and found.score >= settings.HEAL_MEMORY_MIN_SCORE


@pytest.mark.anyio
async def test_an_unrelated_problem_finds_nothing(memory: HealMemoryService) -> None:
    await memory.remember(*_memory(ORG_A, "run-1"))

    assert await memory.recall(org_id=ORG_A, run_id="run-2", result=CHROMEDRIVER) == []


@pytest.mark.anyio
async def test_memories_never_cross_organizations(memory: HealMemoryService) -> None:
    await memory.remember(*_memory(ORG_A, "run-a"))
    await memory.remember(*_memory(ORG_B, "run-b"))

    found_by_a = await memory.recall(org_id=ORG_A, run_id="run-x", result=MAVEN_TESTNG)
    found_by_c = await memory.recall(org_id=ORG_C, run_id="run-x", result=MAVEN_TESTNG)

    assert [(m.org_id, m.run_id) for m in found_by_a] == [(ORG_A, "run-a")]
    assert found_by_c == []


@pytest.mark.anyio
async def test_the_current_run_is_left_out(memory: HealMemoryService) -> None:
    # Its own earlier fixes are already in the healer's prompt.
    await memory.remember(*_memory(ORG_A, "run-1"))

    assert await memory.recall(org_id=ORG_A, run_id="run-1", result=MAVEN_TESTNG) == []
    assert len(await memory.recall(org_id=ORG_A, run_id="run-2", result=MAVEN_TESTNG)) == 1


@pytest.mark.anyio
async def test_successful_and_failed_heals_are_both_kept(memory: HealMemoryService) -> None:
    failed = HealOutcomeModel(summary="Cleared the local Maven cache.", changes=[])
    await memory.remember(*_memory(ORG_A, "run-1", after=MAVEN_TESTNG, heal=failed))
    await memory.remember(*_memory(ORG_A, "run-2", after=PASSED))

    found = await memory.recall(org_id=ORG_A, run_id="run-3", result=MAVEN_SELENIUM)

    assert sorted((m.result, m.succeeded) for m in found) == [
        (HealMemoryResult.FIXED, True), (HealMemoryResult.NOT_FIXED, False)
    ]


@pytest.mark.anyio
async def test_at_most_the_configured_number_of_memories_comes_back(
    memory: HealMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "HEAL_MEMORY_LIMIT", 2)
    for number in range(4):
        await memory.remember(*_memory(ORG_A, f"run-{number}"))

    assert len(await memory.recall(org_id=ORG_A, run_id="run-x", result=MAVEN_TESTNG)) == 2


@pytest.mark.anyio
async def test_vectors_of_another_embedding_model_are_not_compared(memory: HealMemoryService) -> None:
    await memory.remember(*_memory(ORG_A, "run-1", embedding_model="another-embedding-model"))

    assert await memory.recall(org_id=ORG_A, run_id="run-2", result=MAVEN_TESTNG) == []


@pytest.mark.anyio
async def test_saving_the_same_heal_again_replaces_it(memory: HealMemoryService, qdrant: AsyncQdrantClient) -> None:
    await memory.remember(*_memory(ORG_A, "run-1", after=MAVEN_TESTNG))
    await memory.remember(*_memory(ORG_A, "run-1", after=PASSED))

    assert (await qdrant.count("test_heal_memories")).count == 1
    [found] = await memory.recall(org_id=ORG_A, run_id="run-2", result=MAVEN_TESTNG)
    assert found.result == HealMemoryResult.FIXED


@pytest.mark.anyio
async def test_a_query_carries_the_instruction_and_a_stored_problem_does_not(
    memory: HealMemoryService, embeddings: FakeEmbeddings
) -> None:
    point_id, stored = _memory(ORG_A, "run-1")
    await memory.remember(point_id, stored)
    await memory.recall(org_id=ORG_A, run_id="run-2", result=MAVEN_SELENIUM)

    [[saved_text], [query_text]] = embeddings.calls
    assert saved_text == stored.problem
    assert query_text == settings.EMBEDDING_QUERY_PREFIX + describe_problem(MAVEN_SELENIUM)


# ---------------------------------------------------------------- isolation and failures


@pytest.mark.anyio
async def test_a_search_without_an_organization_is_refused(qdrant: AsyncQdrantClient, memory: HealMemoryService) -> None:
    repository = HealMemoryRepository(qdrant, "test_heal_memories")
    await repository.ensure_collection(FakeEmbeddings.DIMENSIONS)

    with pytest.raises(ValueError, match="org_id"):
        await repository.search(org_id="", vector=[0.1] * FakeEmbeddings.DIMENSIONS, embedding_model="m", limit=3, min_score=0.0)
    assert await memory.recall(org_id="", run_id="run-1", result=MAVEN_TESTNG) == []


@pytest.mark.anyio
async def test_a_memory_without_an_organization_is_not_saved(memory: HealMemoryService) -> None:
    point_id, stored = _memory("", "run-1")

    assert await memory.remember(point_id, stored) is False


@pytest.mark.anyio
async def test_another_organizations_memory_from_qdrant_is_dropped() -> None:
    # Defense in depth: even if a search returned another organization's point, it never reaches the healer.
    _, stored = _memory(ORG_B, "run-b")

    class LeakyRepository:
        async def ensure_collection(self, vector_size: int) -> None: ...

        async def search(self, **kwargs: Any) -> list[tuple[str, dict[str, Any], float]]:
            return [("id-b", stored.model_dump(mode="json"), 0.99)]

    service = HealMemoryService(LeakyRepository(), FakeEmbeddings(), enabled=True)  # type: ignore[arg-type]

    assert await service.recall(org_id=ORG_A, run_id="run-a", result=MAVEN_TESTNG) == []


@pytest.mark.anyio
async def test_down_servers_mean_no_memories_not_a_failed_heal(qdrant: AsyncQdrantClient) -> None:
    service = HealMemoryService(HealMemoryRepository(qdrant, "test_heal_memories"), FakeEmbeddings(fail=True), enabled=True)

    assert await service.recall(org_id=ORG_A, run_id="run-1", result=MAVEN_TESTNG) == []
    assert await service.remember(*_memory(ORG_A, "run-1")) is False


@pytest.mark.anyio
async def test_memory_that_is_off_does_nothing(embeddings: FakeEmbeddings) -> None:
    service = HealMemoryService(embeddings=embeddings, enabled=False)

    assert await service.recall(org_id=ORG_A, run_id="run-1", result=MAVEN_TESTNG) == []
    assert await service.remember(*_memory(ORG_A, "run-1")) is False
    assert embeddings.calls == []


@pytest.mark.anyio
async def test_a_collection_made_for_another_vector_size_is_refused(qdrant: AsyncQdrantClient) -> None:
    await HealMemoryRepository(qdrant, "test_heal_memories").ensure_collection(256)

    with pytest.raises(ValueError, match="QDRANT_COLLECTION_NAME"):
        await HealMemoryRepository(qdrant, "test_heal_memories").ensure_collection(1024)


def test_memory_is_off_in_the_test_settings() -> None:
    assert HealMemoryService().enabled is False


# ---------------------------------------------------------------- the embedding server


@pytest.mark.anyio
async def test_ollama_is_asked_for_one_vector_per_text() -> None:
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"model": "qwen3-embedding:0.6b", "embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = OllamaEmbeddings("http://ollama.local:11435/", "qwen3-embedding:0.6b", 5, httpx.MockTransport(answer))

    assert await client.embed(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]
    [request] = requests
    assert str(request.url) == "http://ollama.local:11435/api/embed"
    assert json.loads(request.content) == {"model": "qwen3-embedding:0.6b", "input": ["first", "second"], "truncate": True}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, json={"error": "model not found"}), httpx.Response(200, json={"embeddings": [[0.1]]})],
)
async def test_a_bad_answer_from_ollama_raises(response: httpx.Response) -> None:
    client = OllamaEmbeddings("http://ollama.local:11435", "m", 5, httpx.MockTransport(lambda request: response))

    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        await client.embed(["first", "second"])


# ---------------------------------------------------------------- the healer's prompt


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=["pom.xml"], confidence=Confidence.HIGH)


FACT_SHEET = FactSheetModel(
    total_files=3, language_files={"Java": 2}, primary_language=_fact("Java"), build_tool=_fact("Maven"),
    test_frameworks=_fact(["TestNG"]), automation_tools=_fact(["Selenium"]), bdd_tool=_fact(None),
    marker_files=["pom.xml"], test_dirs=["src/test"], feature_file_count=0, top_level_tree=["pom.xml"],
)


def test_memories_reach_the_healer_as_context_not_orders() -> None:
    worked = HealMemoryMapper.to_model(*_memory_parts(ORG_A, "run-1", PASSED, FIX))
    failed = HealMemoryMapper.to_model(
        *_memory_parts(ORG_A, "run-2", MAVEN_TESTNG, HealOutcomeModel(summary="Cleared the local Maven cache."))
    )

    prompt = build_heal_prompt(MAVEN_SELENIUM, COMMAND, FACT_SHEET, memories=[worked, failed])

    assert "Similar problems from earlier runs of this organization" in prompt
    assert "they may not apply here" in prompt
    assert f"- worked: the tests passed afterwards ({STACK}): dependency_failure" in prompt
    assert "Error: [ERROR] Failed to execute goal on project shop-ui-tests" in prompt
    assert "Tried: TestNG 99.0.0 does not exist: set it to 7.10.2 in pom.xml." in prompt
    assert "+<version>7.10.2</version>" in prompt
    assert "- did not work: the same failure came back" in prompt and "Tried: Cleared the local Maven cache." in prompt


def test_no_memories_no_section() -> None:
    assert "Similar problems" not in build_heal_prompt(MAVEN_SELENIUM, COMMAND, FACT_SHEET)


def _memory_parts(org_id: str, run_id: str, after: TestRunResultModel, heal: HealOutcomeModel) -> tuple[str, dict[str, Any]]:
    point_id, stored = _memory(org_id, run_id, after=after, heal=heal)
    return point_id, stored.model_dump(mode="json")
