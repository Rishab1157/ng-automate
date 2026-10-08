"""Test data sources: what the user wants tested, uploaded per project.

Our own JSON/CSV test-case format is read at once (status ready). JSON in any other shape and plain text are stored
as they are (status processing) and turned into test cases by an LLM in the background (TestDataProcessor), with the
upload's model connection or, without one, the default model from .env. The user can then review and edit the cases
(PUT) before a generate run uses them. Bad files are rejected with a clear message (400) before anything is stored.
"""

import logging
from datetime import UTC, datetime

from bson import ObjectId
from pymongo import DESCENDING

from app.core.exceptions import ErrorMessages, NotFoundError, ValidationError
from app.models.testDataModel import (
    TestCaseSpecModel,
    TestDataMapper,
    TestDataSetModel,
    TestDataStatus,
    TestDataUploadModel,
)
from app.projections.testDataProjection import TEST_DATA_DETAIL_PROJECTION, TEST_DATA_LIST_PROJECTION
from app.repositories.testDataRepository import TestDataRepository
from app.services.modelConnectionService import ModelConnectionService, connection_id_or_none
from app.services.projectService import ProjectService
from app.utils.TestDataParser import FreeFormSource, build_test_data_set, read_test_data

logger = logging.getLogger(__name__)

DEFAULT_LIST_LIMIT = 100
MAX_FILENAME_CHARS = 200
MAX_ERROR_CHARS = 500


class TestDataService:
    __test__ = False  # not a pytest test class

    def __init__(self) -> None:
        self.test_data_repo = TestDataRepository()
        self.project_service = ProjectService()
        self.model_connection_service = ModelConnectionService()

    async def create(
        self,
        *,
        project_id: str,
        org_id: str,
        user_id: str,
        filename: str,
        content: bytes,
        model_connection_id: str | None = None,
    ) -> TestDataUploadModel:
        """Ready at once for our own format; otherwise processing (the caller starts the processor)."""
        project = await self.project_service.get(project_id, org_id)
        source = read_test_data(content, filename)
        common = {"org_id": org_id, "project_id": project.id, "user_id": user_id, "filename": filename[-MAX_FILENAME_CHARS:]}
        if isinstance(source, FreeFormSource):
            connection_id = connection_id_or_none(model_connection_id)
            if connection_id is not None:
                # Fail now (400/404) rather than minutes later in the background.
                await self.model_connection_service.get_llm_config(connection_id, org_id)
            db_model = TestDataMapper.to_create_db_model(
                **common, source_format=source.source_format, raw_source=source.text, model_connection_id=connection_id
            )
        else:
            db_model = TestDataMapper.to_create_db_model(**common, source_format=source.source_format, data_set=source)
        await self.test_data_repo.insert_one(db_model.model_dump(by_alias=True))
        logger.info(
            "Test data %s saved for project %s (%s, %s, %d case(s))",
            db_model.id, project.id, db_model.source_format, db_model.status, db_model.case_count,
        )
        return TestDataMapper.to_model(db_model.model_dump(by_alias=True))

    async def get(self, test_data_id: str, project_id: str, org_id: str) -> TestDataUploadModel:
        if not ObjectId.is_valid(test_data_id) or not ObjectId.is_valid(project_id):
            raise NotFoundError(ErrorMessages.TEST_DATA_NOT_FOUND)
        doc = await self.test_data_repo.find_one(
            {"_id": ObjectId(test_data_id), "project_id": ObjectId(project_id), "org_id": ObjectId(org_id)},
            TEST_DATA_DETAIL_PROJECTION,
        )
        if doc is None:
            raise NotFoundError(ErrorMessages.TEST_DATA_NOT_FOUND)
        return TestDataMapper.to_model(doc)

    async def get_by_id(self, test_data_id: str) -> TestDataUploadModel:
        """Internal (processor): no organization check."""
        doc = await self.test_data_repo.find_one({"_id": ObjectId(test_data_id)}, TEST_DATA_DETAIL_PROJECTION)
        if doc is None:
            raise NotFoundError(ErrorMessages.TEST_DATA_NOT_FOUND)
        return TestDataMapper.to_model(doc)

    async def get_all(self, project_id: str, org_id: str, limit: int = DEFAULT_LIST_LIMIT) -> list[TestDataUploadModel]:
        """The project's test data, newest first (without the raw sources)."""
        await self.project_service.get(project_id, org_id)
        docs = await self.test_data_repo.find(
            {"project_id": ObjectId(project_id), "org_id": ObjectId(org_id)},
            TEST_DATA_LIST_PROJECTION,
            sort=[("created_at", DESCENDING), ("_id", DESCENDING)],
            limit=limit,
        )
        return [TestDataMapper.to_model(doc) for doc in docs]

    async def find_processing_ids(self) -> list[str]:
        """Sources still being read: what a restarted server must pick up again."""
        docs = await self.test_data_repo.find(
            {"status": TestDataStatus.PROCESSING.value}, {"_id": 1}, sort=[("created_at", 1)]
        )
        return [str(doc["_id"]) for doc in docs]

    async def update_cases(
        self, test_data_id: str, project_id: str, org_id: str, cases: list[TestCaseSpecModel]
    ) -> TestDataUploadModel:
        """Replace the cases with the user's reviewed version (only once the source has been read)."""
        upload = await self.get(test_data_id, project_id, org_id)
        if upload.status != TestDataStatus.READY:
            raise ValidationError(ErrorMessages.TEST_DATA_NOT_READY.format(status=upload.status.value))
        data_set = build_test_data_set(cases, upload.source_format)
        await self._set(test_data_id, {
            "data_set": data_set.model_dump(mode="json"),
            "case_count": len(data_set.cases),
        })
        return await self.get(test_data_id, project_id, org_id)

    async def retry(self, test_data_id: str, project_id: str, org_id: str) -> TestDataUploadModel:
        """Read a source that failed once more (the caller starts the processor)."""
        upload = await self.get(test_data_id, project_id, org_id)
        if upload.status != TestDataStatus.FAILED or upload.raw_source is None:
            raise ValidationError(ErrorMessages.TEST_DATA_RETRY_NOT_ALLOWED)
        await self._set(test_data_id, {"status": TestDataStatus.PROCESSING.value, "error": None})
        return await self.get(test_data_id, project_id, org_id)

    async def mark_ready(self, test_data_id: str, data_set: TestDataSetModel) -> None:
        await self._set(test_data_id, {
            "status": TestDataStatus.READY.value,
            "data_set": data_set.model_dump(mode="json"),
            "case_count": len(data_set.cases),
            "error": None,
        })
        logger.info("Test data %s read: %d case(s)", test_data_id, len(data_set.cases))

    async def mark_failed(self, test_data_id: str, reason: str) -> None:
        await self._set(test_data_id, {"status": TestDataStatus.FAILED.value, "error": reason[:MAX_ERROR_CHARS]})
        logger.warning("Test data %s could not be read: %s", test_data_id, reason[:MAX_ERROR_CHARS])

    async def _set(self, test_data_id: str, fields: dict) -> None:
        await self.test_data_repo.update_one(
            {"_id": ObjectId(test_data_id)}, {"$set": {**fields, "updated_at": datetime.now(UTC)}}
        )
