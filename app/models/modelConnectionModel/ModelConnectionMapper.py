from typing import Any

from pydantic import SecretStr

from .ModelConnectionModel import ModelConnectionSourceModel


class ModelConnectionMapper:
    @staticmethod
    def to_source_model(doc: dict[str, Any]) -> ModelConnectionSourceModel:
        """doc: a model_connections document joined with its model_providers entry as `provider`
        and its model_types entry as `model_type`."""
        provider = doc.get("provider") or {}
        model_type = doc.get("model_type") or {}
        api_key = doc.get("api_key")
        return ModelConnectionSourceModel(
            id=str(doc["_id"]),
            userdefined_name=doc.get("userdefined_name") or "",
            provider_code=provider.get("provider_code") or "",
            api_base_url=provider.get("api_base_url") or None,
            model_name=model_type.get("type_code") or "",
            api_key=SecretStr(api_key) if api_key else None,
        )
