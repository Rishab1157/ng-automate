from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from .TestDataDbModel import TestDataCreateDbModel
from .TestDataModel import TestDataFormat, TestDataSetModel, TestDataStatus
from .TestDataUploadModel import TestDataUploadModel


class TestDataMapper:
    __test__ = False  # not a pytest test class

    @staticmethod
    def to_create_db_model(
        *,
        org_id: str,
        project_id: str,
        user_id: str,
        filename: str,
        source_format: TestDataFormat,
        data_set: TestDataSetModel | None = None,
        raw_source: str | None = None,
        model_connection_id: str | None = None,
    ) -> TestDataCreateDbModel:
        """Ready at once with `data_set` (our own format); otherwise processing until an LLM has read `raw_source`."""
        now = datetime.now(UTC)
        return TestDataCreateDbModel(
            org_id=ObjectId(org_id),
            project_id=ObjectId(project_id),
            created_by=ObjectId(user_id),
            filename=filename,
            source_format=TestDataFormat(source_format).value,
            status=(TestDataStatus.READY if data_set is not None else TestDataStatus.PROCESSING).value,
            raw_source=raw_source,
            model_connection_id=ObjectId(model_connection_id) if model_connection_id else None,
            data_set=data_set.model_dump(mode="json") if data_set is not None else None,
            case_count=len(data_set.cases) if data_set is not None else 0,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> TestDataUploadModel:
        data_set = doc.get("data_set")
        model_connection_id = doc.get("model_connection_id")
        # Documents from before free-form sources were always our own format, read at once.
        source_format = doc.get("source_format") or (data_set or {}).get("source_format") or TestDataFormat.JSON.value
        return TestDataUploadModel(
            id=str(doc["_id"]),
            org_id=str(doc["org_id"]),
            project_id=str(doc["project_id"]),
            created_by=str(doc["created_by"]),
            filename=doc["filename"],
            source_format=TestDataFormat(source_format),
            status=TestDataStatus(doc.get("status") or TestDataStatus.READY.value),
            raw_source=doc.get("raw_source"),
            model_connection_id=str(model_connection_id) if model_connection_id else None,
            data_set=TestDataSetModel.model_validate(data_set) if data_set is not None else None,
            case_count=doc.get("case_count") or 0,
            error=doc.get("error"),
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at") or doc["created_at"],
        )
