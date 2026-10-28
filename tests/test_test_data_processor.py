"""TestDataProcessor: reads free-form sources in the background, one at a time, and records the outcome."""

import asyncio
import importlib
from collections.abc import Collection
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from bson import ObjectId
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import NotFoundError
from app.models.testDataModel import (
    TestCaseSpecModel,
    TestDataFormat,
    TestDataMapper,
    TestDataSetModel,
    TestDataStatus,
    TestDataUploadModel,
    TestStepModel,
)
from app.models.tokenUsageModel import TokenUsageModel
from app.services.testDataService import TestDataProcessor, TestDataService
from app.utils.LlmInstance import MODEL_NOT_ANSWERING

DATA_SET = TestDataSetModel(
    source_format=TestDataFormat.TEXT,
    cases=[TestCaseSpecModel(id="TD-001", title="Login", steps=[TestStepModel(action="open", target="https://x")])],
)


def _upload(test_data_id: str, status: TestDataStatus = TestDataStatus.PROCESSING) -> TestDataUploadModel:
    now = datetime.now(UTC)
    return TestDataUploadModel(
        id=test_data_id, org_id="org-1", project_id="p-1", created_by="u-1", filename="cases.txt",
        source_format=TestDataFormat.TEXT, status=status, raw_source="TC-1: Login\n1. Open https://x\n",
        model_connection_id=None, case_count=0, created_at=now, updated_at=now,
    )


class FakeService:
    def __init__(self, *uploads: TestDataUploadModel) -> None:
        self.uploads = {upload.id: upload for upload in uploads}
        self.ready: dict[str, TestDataSetModel] = {}
        self.failed: dict[str, str] = {}
        self.token_usage: dict[str, TokenUsageModel | None] = {}

    async def get_by_id(self, test_data_id: str) -> TestDataUploadModel:
        if test_data_id not in self.uploads:
            raise NotFoundError("gone")
        return self.uploads[test_data_id]

    async def find_processing_ids(self) -> list[str]:
        return [i for i, u in self.uploads.items() if u.status == TestDataStatus.PROCESSING]

    async def find_case_ids(self, project_id: str, exclude_id: str) -> set[str]:
        data_sets = [self.ready.get(i) or u.data_set for i, u in self.uploads.items()
                     if u.project_id == project_id and i != exclude_id]
        return {case.id for data_set in data_sets if data_set is not None for case in data_set.cases}

    async def mark_ready(self, test_data_id: str, data_set: TestDataSetModel, revision: int | None = None,
                         token_usage: TokenUsageModel | None = None) -> bool:
        if revision is not None and revision != self.uploads[test_data_id].revision:
            return False  # the source was edited while it was read
        self.ready[test_data_id] = data_set
        self.token_usage[test_data_id] = token_usage
        return True

    async def mark_failed(self, test_data_id: str, reason: str, revision: int | None = None,
                          token_usage: TokenUsageModel | None = None) -> bool:
        if revision is not None and revision != self.uploads[test_data_id].revision:
            return False
        self.failed[test_data_id] = reason
        self.token_usage[test_data_id] = token_usage
        return True


class FakeConnections:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str]] = []

    async def get_llm_config(self, model_connection_id: str | None, org_id: str) -> Any:
        self.calls.append((model_connection_id, org_id))
        return SimpleNamespace(model="ollama_chat/devstral")


class FakeExtractor:
    active = 0
    most_active = 0

    def __init__(self, outcome: TestDataSetModel | Exception) -> None:
        self.outcome = outcome
        self.taken: list[set[str]] = []  # the project's case ids each reading was given

    def __call__(self, llm_config: Any) -> "FakeExtractor":
        return self

    def extract(self, text: str, source_format: TestDataFormat, taken_ids: Collection[str] = ()) -> TestDataSetModel:
        import time

        self.taken.append(set(taken_ids))
        FakeExtractor.active += 1
        FakeExtractor.most_active = max(FakeExtractor.most_active, FakeExtractor.active)
        time.sleep(0.05)
        FakeExtractor.active -= 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _processor(service: FakeService, outcome: TestDataSetModel | Exception = DATA_SET) -> tuple[TestDataProcessor, FakeConnections]:
    connections = FakeConnections()
    processor = TestDataProcessor(
        test_data_service=service,  # type: ignore[arg-type]
        model_connection_service=connections,  # type: ignore[arg-type]
        extractor_factory=FakeExtractor(outcome),  # type: ignore[arg-type]
    )
    return processor, connections


async def _settle(processor: TestDataProcessor) -> None:
    await asyncio.gather(*list(processor._tasks.values()), return_exceptions=True)


@pytest.mark.anyio
async def test_source_is_read_with_the_default_model_and_marked_ready() -> None:
    service = FakeService(_upload("t1"))
    processor, connections = _processor(service)

    await processor.submit("t1")
    await _settle(processor)

    assert service.ready == {"t1": DATA_SET}
    assert connections.calls == [(None, "org-1")]  # no connection id -> the free default model


@pytest.mark.anyio
async def test_a_source_edited_while_it_is_read_is_read_again_and_only_the_new_version_counts() -> None:
    service = FakeService(_upload("t1"))
    edited = _upload("t1").model_copy(update={"raw_source": "TC-1: Logout\n", "revision": 1})
    texts: list[str] = []

    class EditDuringReading(FakeExtractor):
        def extract(
            self, text: str, source_format: TestDataFormat, taken_ids: Collection[str] = ()
        ) -> TestDataSetModel:
            texts.append(text)
            service.uploads["t1"] = edited  # the editor saves a new version meanwhile
            return super().extract(text, source_format, taken_ids)

    processor = TestDataProcessor(
        test_data_service=service,  # type: ignore[arg-type]
        model_connection_service=FakeConnections(),  # type: ignore[arg-type]
        extractor_factory=EditDuringReading(DATA_SET),  # type: ignore[arg-type]
    )

    await processor.submit("t1")
    await _settle(processor)

    assert texts == ["TC-1: Login\n1. Open https://x\n", "TC-1: Logout\n"]
    assert service.ready == {"t1": DATA_SET}


@pytest.mark.anyio
async def test_unreadable_source_is_marked_failed_with_the_reason() -> None:
    service = FakeService(_upload("t1"))
    processor, _ = _processor(service, ValueError("no test cases were found in the source"))

    await processor.submit("t1")
    await _settle(processor)

    assert service.failed == {"t1": "The test data could not be turned into test cases: no test cases were found in the source"}


@pytest.mark.anyio
async def test_a_model_that_does_not_answer_fails_the_source_instead_of_keeping_it_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real run: a text source stayed "processing" for 20+ minutes while the model server answered no request.
    extractor_module = importlib.import_module("app.agents.TestGeneratorAgent.TestCaseExtractor")
    not_answering = MODEL_NOT_ANSWERING.format(seconds=90)
    monkeypatch.setattr(extractor_module, "check_model_answers", lambda config: not_answering)
    service = FakeService(_upload("t1"))
    processor = TestDataProcessor(
        test_data_service=service,  # type: ignore[arg-type]
        model_connection_service=FakeConnections(),  # type: ignore[arg-type]
        extractor_factory=extractor_module.TestCaseExtractor,
    )

    await processor.submit("t1")
    await _settle(processor)

    assert service.failed == {"t1": f"The test data could not be turned into test cases: {not_answering}"}
    assert service.ready == {}


@pytest.mark.anyio
async def test_sources_are_read_one_at_a_time_and_resumed_on_startup() -> None:
    FakeExtractor.most_active = 0
    service = FakeService(_upload("t1"), _upload("t2"), _upload("t3", TestDataStatus.READY))
    processor, _ = _processor(service)

    resumed = await processor.resume_processing()
    await _settle(processor)

    assert resumed == 2
    assert set(service.ready) == {"t1", "t2"}
    assert FakeExtractor.most_active == 1


@pytest.mark.anyio
async def test_finished_or_deleted_sources_are_skipped() -> None:
    service = FakeService(_upload("t1", TestDataStatus.READY))
    processor, connections = _processor(service)

    await processor.submit("t1")
    await processor.submit("missing")
    await _settle(processor)

    assert service.ready == {} and service.failed == {} and connections.calls == []


def _cases(*ids: str) -> TestDataSetModel:
    return TestDataSetModel(source_format=TestDataFormat.TEXT, cases=[
        TestCaseSpecModel(id=case_id, title="Login", steps=[TestStepModel(action="open", target="https://x")])
        for case_id in ids
    ])


@pytest.mark.anyio
async def test_each_reading_gets_the_case_ids_of_the_projects_other_test_data() -> None:
    # Ids the reader gives cases without one (TD-001, ...) must be new in the project: tests are named after them, and
    # a real project got three different tests named TD-001 from three sources.
    read_before = _upload("t0", TestDataStatus.READY).model_copy(update={"data_set": _cases("TD-001", "LOGIN-1")})
    other_project = _upload("x1", TestDataStatus.READY).model_copy(
        update={"project_id": "p-2", "data_set": _cases("X-1")}
    )
    service = FakeService(read_before, other_project, _upload("t1"), _upload("t2"))

    class Numbering(FakeExtractor):
        def extract(
            self, text: str, source_format: TestDataFormat, taken_ids: Collection[str] = ()
        ) -> TestDataSetModel:
            super().extract(text, source_format, taken_ids)
            return _cases(f"TD-00{len(self.taken) + 1}")

    reader = Numbering(DATA_SET)
    processor = TestDataProcessor(
        test_data_service=service,  # type: ignore[arg-type]
        model_connection_service=FakeConnections(),  # type: ignore[arg-type]
        extractor_factory=reader,  # type: ignore[arg-type]
    )

    await processor.submit("t1")
    await processor.submit("t2")
    await _settle(processor)

    # One source at a time: the second reading already sees the first one's cases.
    assert reader.taken == [{"TD-001", "LOGIN-1"}, {"TD-001", "LOGIN-1", "TD-002"}]
    assert [service.ready[i].cases[0].id for i in ("t1", "t2")] == ["TD-002", "TD-003"]


@pytest.mark.anyio
async def test_the_service_finds_the_case_ids_of_the_projects_other_test_data(app_db: MongoClient, org_id: str) -> None:
    project_id = str(ObjectId())

    def seed(project: str, *case_ids: str) -> str:
        db_model = TestDataMapper.to_create_db_model(
            org_id=org_id, project_id=project, user_id=str(ObjectId()), filename="cases.txt",
            source_format=TestDataFormat.TEXT, data_set=_cases(*case_ids) if case_ids else None,
            raw_source=None if case_ids else "TC-1: Login",
        )
        app_db[settings.MONGO_DATABASE].test_data.insert_one(db_model.model_dump(by_alias=True))
        return str(db_model.id)

    seed(project_id, "TD-001", "LOGIN-1")
    seed(project_id)  # still being read: no cases yet
    being_read_again = seed(project_id, "TD-002")  # its own earlier cases are not taken: it is read again
    seed(str(ObjectId()), "TD-003")  # another project

    assert await TestDataService().find_case_ids(project_id, being_read_again) == {"TD-001", "LOGIN-1"}
