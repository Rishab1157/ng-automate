from bson import ObjectId

from app.core.exceptions import ErrorCode, ErrorMessages, NotFoundError
from app.models.gitConnectionModel import GitCloneSourceModel, GitConnectionMapper
from app.projections.gitConnectionProjection import GIT_CONNECTION_CLONE_PROJECTION
from app.repositories.gitConnectionRepository import GitConnectionRepository
from app.services.moduleService import NGAUTOMATE_MODULE_CODE, ModuleService


class GitConnectionService:
    """Reads QXcel git connections (read-only)."""

    def __init__(self) -> None:
        self.git_conn_repo = GitConnectionRepository()
        self.module_service = ModuleService()

    async def get_clone_source(self, git_connection_id: str, org_id: str) -> GitCloneSourceModel:
        """An active connection of this org that QXcel has enabled for NG Automate."""
        module_id = await self.module_service.get_module_id(NGAUTOMATE_MODULE_CODE)
        pipeline = [
            {"$match": {
                "_id": ObjectId(git_connection_id),
                "org_id": ObjectId(org_id),
                "is_active": True,
                "available_for_module_ids": module_id,
                "active_for_module_ids": module_id,
            }},
            {"$lookup": {
                "from": "git_providers",
                "localField": "git_provider_id",
                "foreignField": "_id",
                "as": "provider",
            }},
            {"$unwind": {"path": "$provider", "preserveNullAndEmptyArrays": True}},
            {"$project": GIT_CONNECTION_CLONE_PROJECTION},
        ]
        docs = await self.git_conn_repo.aggregate(pipeline)
        if not docs:
            raise NotFoundError(ErrorMessages.GIT_CONNECTION_NOT_FOUND, error_code=ErrorCode.CONNECTION_NOT_FOUND)
        return GitConnectionMapper.to_clone_source_model(docs[0])
