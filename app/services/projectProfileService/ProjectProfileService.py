"""STUB (interface only) — replaced by the runs-api build step."""

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.projectProfileModel import ProjectProfileModel


class ProjectProfileService:
    async def create(
        self,
        *,
        org_id: str,
        project_id: str,
        run_id: str,
        fact_sheet: FactSheetModel,
        findings: AnalyzerFindingsModel,
        llm_model: str,
    ) -> ProjectProfileModel:
        raise NotImplementedError

    async def get_latest(self, project_id: str, org_id: str) -> ProjectProfileModel:
        raise NotImplementedError
