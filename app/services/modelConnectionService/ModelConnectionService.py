from bson import ObjectId

from app.core.exceptions import ErrorCode, ErrorMessages, NotFoundError, ValidationError
from app.models.llmModel import LlmConfigModel
from app.models.modelConnectionModel import ModelConnectionMapper
from app.projections.modelConnectionProjection import MODEL_CONNECTION_LLM_PROJECTION
from app.repositories.modelConnectionRepository import ModelConnectionRepository
from app.services.moduleService import NGAUTOMATE_MODULE_CODE, ModuleService
from app.utils.LlmInstance import get_default_llm_config, llm_config_from_connection

# Values the QXcel UI sends when no connection is selected.
_NO_CONNECTION = {"", "undefined", "null", "none"}


def connection_id_or_none(model_connection_id: str | None) -> str | None:
    """The connection id to store, or None when none was chosen (the default model from .env is used then)."""
    if model_connection_id is None or model_connection_id.strip().lower() in _NO_CONNECTION:
        return None
    return model_connection_id.strip()


class ModelConnectionService:
    """Turns a QXcel model connection (read-only) into an LLM config."""

    def __init__(self) -> None:
        self.model_conn_repo = ModelConnectionRepository()
        self.module_service = ModuleService()

    async def get_llm_config(self, model_connection_id: str | None, org_id: str) -> LlmConfigModel:
        """No connection -> the default model (Devstral). Otherwise the org's connection enabled for NG Automate."""
        if connection_id_or_none(model_connection_id) is None:
            return get_default_llm_config()
        if not ObjectId.is_valid(model_connection_id):
            raise ValidationError(ErrorMessages.INVALID_OBJECT_ID.format(field="model_connection_id"))

        module_id = await self.module_service.get_module_id(NGAUTOMATE_MODULE_CODE)
        pipeline = [
            {"$match": {
                "_id": ObjectId(model_connection_id),
                "org_id": ObjectId(org_id),
                "is_active": True,
                "available_for_module_ids": module_id,
                "active_for_module_ids": module_id,
            }},
            {"$lookup": {
                "from": "model_providers",
                "let": {"provider_id": "$model_provider_id"},
                "pipeline": [{"$match": {"$expr": {"$eq": ["$_id", "$$provider_id"]}, "is_active": True}}],
                "as": "provider",
            }},
            {"$unwind": "$provider"},
            {"$lookup": {"from": "model_types", "localField": "model_type_id", "foreignField": "_id", "as": "model_type"}},
            {"$unwind": "$model_type"},
            {"$project": MODEL_CONNECTION_LLM_PROJECTION},
        ]
        docs = await self.model_conn_repo.aggregate(pipeline)
        source = ModelConnectionMapper.to_source_model(docs[0]) if docs else None
        if source is None or not source.model_name or not source.provider_code:
            raise NotFoundError(ErrorMessages.MODEL_CONNECTION_NOT_FOUND, error_code=ErrorCode.CONNECTION_NOT_FOUND)
        return llm_config_from_connection(source)
