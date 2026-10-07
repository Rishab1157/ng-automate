"""Project profiles: the analyzer's result for a project.

A project keeps every profile it ever got; the newest one is current and is what later agents read.
"""

import logging

from bson import ObjectId
from pymongo import DESCENDING

from app.core.exceptions import ErrorMessages, NotFoundError
from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.projectProfileModel import ProjectProfileMapper, ProjectProfileModel
from app.projections.projectProfileProjection import PROJECT_PROFILE_DETAIL_PROJECTION
from app.repositories.projectProfileRepository import ProjectProfileRepository
from app.services.projectService import ProjectService

logger = logging.getLogger(__name__)


class ProjectProfileService:
    def __init__(self) -> None:
        self.profile_repo = ProjectProfileRepository()
        self.project_service = ProjectService()

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
        db_model = ProjectProfileMapper.to_create_db_model(
            org_id=org_id,
            project_id=project_id,
            run_id=run_id,
            # Also accepts the plain dicts a resumed run holds; either way only valid data is stored.
            fact_sheet=FactSheetModel.model_validate(fact_sheet),
            findings=AnalyzerFindingsModel.model_validate(findings),
            llm_model=llm_model,
        )
        await self.profile_repo.insert_one(db_model.model_dump(by_alias=True))
        logger.info("Profile %s saved for project %s (run %s)", db_model.id, project_id, run_id)
        return ProjectProfileMapper.to_model(db_model.model_dump(by_alias=True))

    async def get_latest(self, project_id: str, org_id: str) -> ProjectProfileModel:
        await self.project_service.get(project_id, org_id)
        docs = await self.profile_repo.find(
            {"project_id": ObjectId(project_id), "org_id": ObjectId(org_id)},
            PROJECT_PROFILE_DETAIL_PROJECTION,
            sort=[("created_at", DESCENDING), ("_id", DESCENDING)],
            limit=1,
        )
        if not docs:
            raise NotFoundError(ErrorMessages.PROFILE_NOT_FOUND)
        return ProjectProfileMapper.to_model(docs[0])
