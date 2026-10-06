"""Where NG Automate gets its LLM settings from.

Today: environment variables (.env). Later: a model connection stored in MongoDB.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str | None
    # repr=False keeps the key out of prints, logs and error messages.
    api_key: str = field(repr=False)
    # "chat" = Chat Completions API, "responses" = OpenAI Responses API, "auto" = SDK decides.
    # Gateways that only allow /chat/completions need "chat".
    api_mode: str = "auto"


def load_llm_config() -> LLMConfig:
    """Read the LLM settings from environment variables."""
    return LLMConfig(
        model=_require("LLM_MODEL"),
        base_url=os.environ.get("LLM_BASE_URL") or None,
        api_key=_require("LLM_API_KEY"),
        api_mode=os.environ.get("LLM_API_MODE", "auto"),
    )


def load_llm_config_from_mongo(connection_id: str) -> LLMConfig:
    """Read the LLM settings from a document in ng_automate.model_connections."""
    from bson import ObjectId
    from pymongo import MongoClient

    with MongoClient(_require("MONGO_URI")) as client:
        doc = client["ng_automate"]["model_connections"].find_one(
            {"_id": ObjectId(connection_id)}
        )
    if doc is None:
        raise ValueError(f"No model connection found with id {connection_id}")
    return LLMConfig(
        model=doc["model"],
        base_url=doc.get("base_url"),
        api_key=doc["api_key"],
        api_mode=doc.get("api_mode", "auto"),
    )


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable {name}. See .env.example.")
    return value
