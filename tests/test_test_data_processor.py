"""TestDataProcessor: reads free-form sources in the background, one at a time, and records the outcome."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.exceptions import NotFoundError
from app.models.testDataModel import (
    TestCaseSpecModel,
    TestDataFormat,
    TestDataSetModel,
    TestDataStatus,
    TestDataUploadModel,
    TestStepModel,
)
from app.services.testDataService import TestDataProcessor

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

    async def get_by_id(self, test_data_id: str) -> TestDataUploadModel:
        if test_data_id not in self.uploads:
            raise NotFoundError("gone")
        return self.uploads[test_data_id]

    async def find_processing_ids(self) -> list[str]:
        return [i for i, u in self.uploads.items() if u.status == TestDataStatus.PROCESSING]

    async def mark_ready(self, test_data_id: str, data_set: TestDataSetModel) -> None:
        self.ready[test_data_id] = data_set

    async def mark_failed(self, test_data_id: str, reason: str) -> None:
        self.failed[test_data_id] = reason


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

    def __call__(self, llm_config: Any) -> "FakeExtractor":
        return self

    def extract(self, text: str, source_format: TestDataFormat) -> TestDataSetModel:
        import time

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
async def test_unreadable_source_is_marked_failed_with_the_reason() -> None:
    service = FakeService(_upload("t1"))
    processor, _ = _processor(service, ValueError("no test cases were found in the source"))

    await processor.submit("t1")
    await _settle(processor)

    assert service.failed == {"t1": "The test data could not be turned into test cases: no test cases were found in the source"}


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
