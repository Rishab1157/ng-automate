"""STUB (interface only) — replaced by the model-connections build step."""

from app.models.llmModel import LlmConfigModel


class ModelConnectionService:
    async def get_llm_config(self, model_connection_id: str | None, org_id: str) -> LlmConfigModel:
        raise NotImplementedError
