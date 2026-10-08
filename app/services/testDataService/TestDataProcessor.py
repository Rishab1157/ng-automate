"""Reads free-form test-data sources (plain text, JSON in any shape) with an LLM in the background.

One source at a time by default: the model is shared and slow, and an upload must not wait for it. A source being
read when the server stops stays "processing" and is read again on the next start.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.core.exceptions import ErrorMessages, NgAutomateException, NotFoundError
from app.models.testDataModel import TestDataFormat, TestDataSetModel, TestDataStatus

if TYPE_CHECKING:
    from app.agents.TestGeneratorAgent.TestCaseExtractor import TestCaseExtractor
    from app.models.llmModel import LlmConfigModel
    from app.services.modelConnectionService import ModelConnectionService

    from .TestDataService import TestDataService

logger = logging.getLogger(__name__)


class TestDataProcessor:
    __test__ = False  # not a pytest test class

    def __init__(
        self,
        max_concurrent: int = 1,
        test_data_service: "TestDataService | None" = None,
        model_connection_service: "ModelConnectionService | None" = None,
        extractor_factory: "Callable[[LlmConfigModel], TestCaseExtractor] | None" = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.max_concurrent = max_concurrent
        # Production defaults are created on first use: they pull in the LLM libraries.
        self._test_data_service = test_data_service
        self._model_connection_service = model_connection_service
        self._extractor_factory = extractor_factory
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._slots: asyncio.Semaphore | None = None
        self._slots_loop: asyncio.AbstractEventLoop | None = None

    @property
    def test_data_service(self) -> "TestDataService":
        if self._test_data_service is None:
            from .TestDataService import TestDataService

            self._test_data_service = TestDataService()
        return self._test_data_service

    @property
    def model_connection_service(self) -> "ModelConnectionService":
        if self._model_connection_service is None:
            from app.services.modelConnectionService import ModelConnectionService

            self._model_connection_service = ModelConnectionService()
        return self._model_connection_service

    @property
    def extractor_factory(self) -> "Callable[[LlmConfigModel], TestCaseExtractor]":
        if self._extractor_factory is None:
            from app.agents.TestGeneratorAgent.TestCaseExtractor import TestCaseExtractor

            self._extractor_factory = TestCaseExtractor
        return self._extractor_factory

    async def submit(self, test_data_id: str) -> None:
        """Read the source in the background. A source already being read here is left alone."""
        task = self._tasks.get(test_data_id)
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._process(test_data_id), name=f"test-data-{test_data_id}")
        self._tasks[test_data_id] = task
        task.add_done_callback(lambda done: self._forget(test_data_id, done))

    async def resume_processing(self) -> int:
        """On startup: read again every source that was still being read."""
        ids = await self.test_data_service.find_processing_ids()
        for test_data_id in ids:
            await self.submit(test_data_id)
        return len(ids)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process(self, test_data_id: str) -> None:
        async with self._concurrency_slots():
            try:
                upload = await self.test_data_service.get_by_id(test_data_id)
            except NotFoundError:
                return
            if upload.status != TestDataStatus.PROCESSING or upload.raw_source is None:
                return
            try:
                llm_config = await self.model_connection_service.get_llm_config(upload.model_connection_id, upload.org_id)
                data_set = await asyncio.to_thread(self._extract, llm_config, upload.raw_source, upload.source_format)
            except asyncio.CancelledError:
                raise  # stays processing: read again on the next start
            except NgAutomateException as error:
                await self.test_data_service.mark_failed(test_data_id, error.message)
            except ValueError as error:
                await self.test_data_service.mark_failed(
                    test_data_id, ErrorMessages.TEST_DATA_READ_FAILED.format(reason=str(error))
                )
            except Exception:
                logger.exception("Reading test data %s failed unexpectedly", test_data_id)
                await self.test_data_service.mark_failed(test_data_id, ErrorMessages.INTERNAL_ERROR)
            else:
                await self.test_data_service.mark_ready(test_data_id, data_set)

    def _extract(self, llm_config: "LlmConfigModel", text: str, source_format: TestDataFormat) -> TestDataSetModel:
        # In the worker thread: the first use imports the LLM libraries, which would stall the event loop.
        return self.extractor_factory(llm_config).extract(text, source_format)

    def _concurrency_slots(self) -> asyncio.Semaphore:
        # asyncio primitives belong to one event loop, and tests start several loops in one process.
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(self.max_concurrent)
            self._slots_loop = loop
        return self._slots

    def _forget(self, test_data_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(test_data_id) is task:
            del self._tasks[test_data_id]
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Test data %s stopped with an unexpected error", test_data_id, exc_info=error)


test_data_processor = TestDataProcessor()
