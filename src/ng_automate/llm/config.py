"""Where LLM settings come from.

Default: environment variables (.env), e.g. Devstral on the innovation server.
Per connection: a document in the model_connections collection.
"""

from ng_automate.core.env import get_env, require_env
from ng_automate.models.llm import LLMConfig


def load_llm_config() -> LLMConfig:
    """The default LLM, read from environment variables."""
    num_ctx = get_env("LLM_NUM_CTX")
    return LLMConfig(
        model=require_env("LLM_MODEL"),
        base_url=get_env("LLM_BASE_URL"),
        api_key=get_env("LLM_API_KEY"),
        api_mode=get_env("LLM_API_MODE", "auto"),
        num_ctx=int(num_ctx) if num_ctx else None,
        thinking=(get_env("LLM_THINKING", "true") or "true").lower() != "false",
    )


def load_llm_config_from_mongo(connection_id: str) -> LLMConfig:
    """One model connection, read from ng_automate.model_connections."""
    from bson import ObjectId
    from pymongo import MongoClient

    with MongoClient(require_env("MONGO_URI")) as client:
        doc = client["ng_automate"]["model_connections"].find_one(
            {"_id": ObjectId(connection_id)}
        )
    if doc is None:
        raise ValueError(f"No model connection found with id {connection_id}")
    return LLMConfig(
        model=doc["model"],
        base_url=doc.get("base_url"),
        api_key=doc.get("api_key"),
        api_mode=doc.get("api_mode", "auto"),
        num_ctx=doc.get("num_ctx"),
        thinking=doc.get("thinking", True),
    )
