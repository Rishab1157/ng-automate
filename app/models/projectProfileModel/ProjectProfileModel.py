from datetime import datetime

from pydantic import BaseModel

from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel


class ProjectProfileModel(BaseModel):
    """The analyzer's result for one project: the single source of truth every later agent reads."""

    id: str
    org_id: str
    project_id: str
    run_id: str
    fact_sheet: FactSheetModel
    findings: AnalyzerFindingsModel
    # The model that produced `findings` (LiteLLM name), for traceability.
    llm_model: str
    created_at: datetime
