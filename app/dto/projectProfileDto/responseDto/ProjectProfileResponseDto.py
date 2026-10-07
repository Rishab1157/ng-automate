from datetime import datetime

from pydantic import BaseModel

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.projectProfileModel import ProjectProfileModel


class ProjectProfileResponseDTO(BaseModel):
    id: str
    project_id: str
    run_id: str
    fact_sheet: FactSheetModel
    findings: AnalyzerFindingsModel
    llm_model: str
    created_at: datetime

    @classmethod
    def from_model(cls, profile: ProjectProfileModel) -> "ProjectProfileResponseDTO":
        return cls(
            id=profile.id,
            project_id=profile.project_id,
            run_id=profile.run_id,
            fact_sheet=profile.fact_sheet,
            findings=profile.findings,
            llm_model=profile.llm_model,
            created_at=profile.created_at,
        )
