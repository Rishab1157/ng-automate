from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel

from .ProjectProfileDbModel import ProjectProfileCreateDbModel
from .ProjectProfileModel import ProjectProfileModel


class ProjectProfileMapper:
    @staticmethod
    def to_create_db_model(
        *,
        org_id: str,
        project_id: str,
        run_id: str,
        fact_sheet: FactSheetModel,
        findings: AnalyzerFindingsModel,
        llm_model: str,
    ) -> ProjectProfileCreateDbModel:
        return ProjectProfileCreateDbModel(
            org_id=ObjectId(org_id),
            project_id=ObjectId(project_id),
            run_id=ObjectId(run_id),
            fact_sheet=fact_sheet.model_dump(mode="json"),
            findings=findings.model_dump(mode="json"),
            llm_model=llm_model,
            created_at=datetime.now(UTC),
        )

    @staticmethod
    def to_model(doc: dict[str, Any]) -> ProjectProfileModel:
        return ProjectProfileModel(
            id=str(doc["_id"]),
            org_id=str(doc["org_id"]),
            project_id=str(doc["project_id"]),
            run_id=str(doc["run_id"]),
            fact_sheet=FactSheetModel.model_validate(doc["fact_sheet"]),
            findings=AnalyzerFindingsModel.model_validate(doc["findings"]),
            llm_model=doc["llm_model"],
            created_at=doc["created_at"],
        )
