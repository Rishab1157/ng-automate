"""Test data sources: what the user wants tested, uploaded per project.

Our own JSON/CSV test-case format is read at once (status ready). JSON in any other shape and plain text are stored
as they are (status processing) and turned into test cases by an LLM in the background (TestDataProcessor), with the
local model (USE_LOCAL_LLM) or the upload's model connection. The user can then review and edit the cases (PUT), or
edit the file itself in the editor (its text is read again), before a generate run uses them. Bad files are rejected
with a clear message (400) before anything is stored.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bson import ObjectId
from pymongo import DESCENDING

from app.core.exceptions import ErrorMessages, NotFoundError, ProjectBusyError, ValidationError
from app.models.testDataModel import (
    TestCaseSpecModel,
    TestDataContentModel,
    TestDataFormat,
    TestDataMapper,
    TestDataSetModel,
    TestDataStatus,
    TestDataUploadModel,
)
from app.models.tokenUsageModel import TokenUsageModel
from app.projections.testDataProjection import TEST_DATA_DETAIL_PROJECTION, TEST_DATA_LIST_PROJECTION
from app.repositories.testDataRepository import TestDataRepository
from app.services.modelConnectionService import ModelConnectionService
from app.services.projectService import ProjectService
from app.services.projectService.ProjectFileService import MAX_EDITABLE_BYTES
from app.utils.TestDataParser import FreeFormSource, build_test_data_set, read_test_data

if TYPE_CHECKING:
    from app.services.runService import RunService

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
        self._run_service: "RunService | None" = None

    @property
    def run_service(self) -> "RunService":
        # Loaded on first use: RunService uses this service (generate runs read their test data).
        if self._run_service is None:
            from app.services.runService import RunService

            self._run_service = RunService()
        return self._run_service

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
        source = read_test_data(content, filename, await self.find_case_ids(project.id))
        # The original file is kept as it is, next to the project (the editor's "Test data" folder, and
        # /workspace/test-data in the sandbox). Its stored name is unique in the project.
        storage = self.project_service.storage
        stored_name = await asyncio.to_thread(storage.save_test_data_file, project.id, filename, content)
        try:
            return await self._insert(project.id, org_id, user_id, stored_name, source, model_connection_id)
        except BaseException:
            await asyncio.to_thread(storage.delete_test_data_file, project.id, stored_name)
            raise

    async def _insert(
        self,
        project_id: str,
        org_id: str,
        user_id: str,
        filename: str,
        source: TestDataSetModel | FreeFormSource,
        model_connection_id: str | None,
    ) -> TestDataUploadModel:
        common = {"org_id": org_id, "project_id": project_id, "user_id": user_id, "filename": filename[-MAX_FILENAME_CHARS:]}
        if isinstance(source, FreeFormSource):
            connection_id = await self.model_connection_service.connection_for_request(model_connection_id, org_id)
            db_model = TestDataMapper.to_create_db_model(
                **common, source_format=source.source_format, raw_source=source.text, model_connection_id=connection_id
            )
        else:
            db_model = TestDataMapper.to_create_db_model(**common, source_format=source.source_format, data_set=source)
        await self.test_data_repo.insert_one(db_model.model_dump(by_alias=True))
        logger.info(
            "Test data %s saved for project %s (%s, %s, %d case(s))",
            db_model.id, project_id, db_model.source_format, db_model.status, db_model.case_count,
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

    async def find_case_ids(self, project_id: str, exclude_id: str | None = None) -> set[str]:
        """The case ids of the project's test data, but those of `exclude_id` (the source being read again). The ids
        given to cases without one must not repeat them: tests are named after case ids."""
        query: dict[str, Any] = {"project_id": ObjectId(project_id), "data_set": {"$ne": None}}
        if exclude_id is not None:
            query["_id"] = {"$ne": ObjectId(exclude_id)}
        docs = await self.test_data_repo.find(query, {"data_set.cases.id": 1})
        return {case["id"] for doc in docs for case in doc["data_set"].get("cases") or [] if case.get("id")}

    async def update_cases(
        self, test_data_id: str, project_id: str, org_id: str, cases: list[TestCaseSpecModel]
    ) -> TestDataUploadModel:
        """Replace the cases with the user's reviewed version (only once the source has been read)."""
        upload = await self.get(test_data_id, project_id, org_id)
        if upload.status != TestDataStatus.READY:
            raise ValidationError(ErrorMessages.TEST_DATA_NOT_READY.format(status=upload.status.value))
        taken_ids = await self.find_case_ids(upload.project_id, test_data_id)
        data_set = build_test_data_set(cases, upload.source_format, taken_ids)
        await self._set(test_data_id, {
            "data_set": data_set.model_dump(mode="json"),
            "case_count": len(data_set.cases),
        })
        return await self.get(test_data_id, project_id, org_id)

    async def retry(
        self, test_data_id: str, project_id: str, org_id: str, model_connection_id: str | None = None
    ) -> TestDataUploadModel:
        """Read a source that failed once more (the caller starts the processor), with the model connection sent or
        the one chosen before."""
        upload = await self.get(test_data_id, project_id, org_id)
        if upload.status != TestDataStatus.FAILED or upload.raw_source is None:
            raise ValidationError(ErrorMessages.TEST_DATA_RETRY_NOT_ALLOWED)
        connection_id = await self.model_connection_service.connection_for_request(
            model_connection_id, org_id, upload.model_connection_id
        )
        await self._set(test_data_id, {
            "status": TestDataStatus.PROCESSING.value,
            "error": None,
            "model_connection_id": ObjectId(connection_id) if connection_id else None,
        })
        return await self.get(test_data_id, project_id, org_id)

    async def read_content(self, test_data_id: str, project_id: str, org_id: str) -> TestDataContentModel:
        """The file's text as it is stored, for the editor (not the cases read from it)."""
        upload = await self.get(test_data_id, project_id, org_id)
        path = self.project_service.storage.test_data_file(upload.project_id, upload.filename)
        return await asyncio.to_thread(_read_content, upload, path)

    async def replace_content(
        self, test_data_id: str, project_id: str, org_id: str, content: str, model_connection_id: str | None = None
    ) -> TestDataUploadModel:
        """The editor's save: the file gets the new text, which is read again like an upload (ready at once for our
        own format; otherwise processing, and the caller starts the processor). Nothing changes when the text is
        invalid (400) or while a run uses the project (409): its sandbox has the test-data folder."""
        upload = await self.get(test_data_id, project_id, org_id)
        if await self.run_service.has_active_run(upload.project_id, org_id):
            raise ProjectBusyError(ErrorMessages.PROJECT_BUSY)
        data = content.encode("utf-8")
        source = read_test_data(data, upload.filename, await self.find_case_ids(upload.project_id, test_data_id))
        fields: dict[str, Any] = {"source_format": TestDataFormat(source.source_format).value, "error": None}
        if isinstance(source, FreeFormSource):
            connection_id = await self.model_connection_service.connection_for_request(
                model_connection_id, org_id, upload.model_connection_id
            )
            fields |= {
                "status": TestDataStatus.PROCESSING.value,
                "raw_source": source.text,
                "model_connection_id": ObjectId(connection_id) if connection_id else None,
                "data_set": None,
                "case_count": 0,
            }
        else:
            fields |= {
                "status": TestDataStatus.READY.value,
                "raw_source": None,
                "data_set": source.model_dump(mode="json"),
                "case_count": len(source.cases),
                "token_usage": None,  # read without an LLM
            }
        storage = self.project_service.storage
        await asyncio.to_thread(storage.replace_test_data_file, upload.project_id, upload.filename, data)
        # A reading of the old text that is still going on must not overwrite this one (see mark_ready).
        await self.test_data_repo.update_one(
            {"_id": ObjectId(test_data_id)},
            {"$set": {**fields, "updated_at": datetime.now(UTC)}, "$inc": {"revision": 1}},
        )
        logger.info("Test data %s edited (%s, %s)", test_data_id, fields["source_format"], fields["status"])
        return await self.get(test_data_id, project_id, org_id)

    async def mark_ready(
        self,
        test_data_id: str,
        data_set: TestDataSetModel,
        revision: int | None = None,
        token_usage: TokenUsageModel | None = None,
    ) -> bool:
        """The cases read from the source, and the tokens reading it used. With `revision` (the source's version that
        was read): written only if the source is still that version, else False."""
        written = await self._set(test_data_id, {
            "status": TestDataStatus.READY.value,
            "data_set": data_set.model_dump(mode="json"),
            "case_count": len(data_set.cases),
            "error": None,
            "token_usage": _dump_usage(token_usage),
        }, revision)
        if written:
            logger.info("Test data %s read: %d case(s)", test_data_id, len(data_set.cases))
        return written

    async def mark_failed(
        self, test_data_id: str, reason: str, revision: int | None = None, token_usage: TokenUsageModel | None = None
    ) -> bool:
        """Like mark_ready, for a source that could not be read."""
        written = await self._set(test_data_id, {
            "status": TestDataStatus.FAILED.value,
            "error": reason[:MAX_ERROR_CHARS],
            "token_usage": _dump_usage(token_usage),
        }, revision)
        if written:
            logger.warning("Test data %s could not be read: %s", test_data_id, reason[:MAX_ERROR_CHARS])
        return written

    async def _set(self, test_data_id: str, fields: dict, revision: int | None = None) -> bool:
        query: dict[str, Any] = {"_id": ObjectId(test_data_id)}
        if revision is not None:
            # Documents from before the editor have no revision: they are revision 0.
            query["revision"] = revision if revision else {"$in": [0, None]}
        changed = await self.test_data_repo.update_one(query, {"$set": {**fields, "updated_at": datetime.now(UTC)}})
        return changed > 0


def _dump_usage(token_usage: TokenUsageModel | None) -> dict[str, Any] | None:
    return token_usage.model_dump(mode="json") if token_usage is not None else None


def _read_content(upload: TestDataUploadModel, path: Path) -> TestDataContentModel:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise NotFoundError(ErrorMessages.FILE_NOT_FOUND) from None
    known = {"test_data_id": upload.id, "filename": upload.filename, "size_bytes": size}
    if size > MAX_EDITABLE_BYTES:
        reason = ErrorMessages.FILE_TOO_LARGE_TO_EDIT.format(limit_kb=MAX_EDITABLE_BYTES // 1024)
        return TestDataContentModel(**known, editable=False, reason=reason)
    data = path.read_bytes()
    try:
        # utf-8-sig: a byte order mark is not part of the text (the parser accepts the file either way).
        text = None if b"\0" in data else data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = None
    if text is None:
        return TestDataContentModel(**known, editable=False, reason=ErrorMessages.TEST_DATA_NOT_TEXT)
    return TestDataContentModel(**known, content=text, editable=True)
