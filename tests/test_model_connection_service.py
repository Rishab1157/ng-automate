from typing import Any

import pytest
from bson import ObjectId
from pydantic import SecretStr
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.modelConnectionModel import ModelConnectionSourceModel
from app.services.modelConnectionService import ModelConnectionService
from app.utils.LlmInstance import llm_config_from_connection
from tests.conftest import NG_MODULE_ID


def _source(provider: str, model: str = "gpt-4o", base: str | None = "https://gw.example/v1") -> ModelConnectionSourceModel:
    return ModelConnectionSourceModel(
        id=str(ObjectId()), userdefined_name="x", provider_code=provider, api_base_url=base,
        model_name=model, api_key=SecretStr("sk-secret"),
    )


@pytest.mark.parametrize(
    ("provider", "model", "expected_model", "keeps_base", "api_mode"),
    [
        ("AgenticQE", "gpt-4o", "openai/gpt-4o", True, "chat"),
        ("OPENAI", "gpt-4o", "openai/gpt-4o", True, "auto"),
        ("ANTHROPIC", "claude-x", "anthropic/claude-x", False, "auto"),
        ("GEMINI", "GEMINI-2.5-PRO", "gemini/gemini-2.5-pro", False, "auto"),
        ("GROK(xAI)", "grok-4.3", "xai/grok-4.3", False, "auto"),
        ("SomethingNew", "m1", "openai/m1", True, "chat"),
    ],
)
def test_provider_routing(provider: str, model: str, expected_model: str, keeps_base: bool, api_mode: str) -> None:
    config = llm_config_from_connection(_source(provider, model))

    assert config.model == expected_model
    assert config.base_url == ("https://gw.example/v1" if keeps_base else None)
    assert config.api_mode == api_mode
    assert "sk-secret" not in repr(config)


def test_non_http_base_url_is_dropped() -> None:
    assert llm_config_from_connection(_source("Unknown", base="string")).base_url is None


def _seed(mongo: MongoClient, org_id: str, *, enabled: bool = True, active: bool = True, provider_active: bool = True) -> str:
    qx = mongo[settings.QXCEL_DATABASE]
    provider_id = qx.model_providers.insert_one({"provider_code": "AgenticQE", "api_base_url": "https://agenticqe.ai/v1", "is_active": provider_active}).inserted_id
    type_id = qx.model_types.insert_one({"type_code": "gpt-4o", "model_provider_id": provider_id}).inserted_id
    modules = [NG_MODULE_ID] if enabled else []
    return str(qx.model_connections.insert_one({
        "org_id": ObjectId(org_id), "model_provider_id": provider_id, "model_type_id": type_id, "api_key": "sk-secret",
        "is_active": active, "available_for_module_ids": modules, "active_for_module_ids": modules,
    }).inserted_id)


class NoDatabase:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError("the local model never reads the database")


def _without_database() -> ModelConnectionService:
    service = ModelConnectionService()
    service.model_conn_repo = NoDatabase()  # type: ignore[assignment]
    service.module_service = NoDatabase()  # type: ignore[assignment]
    return service


@pytest.mark.anyio
@pytest.mark.parametrize("chosen", [None, "", "  ", "undefined", "null", "None", str(ObjectId()), "abc"])
async def test_the_local_model_is_used_whatever_connection_was_chosen(app_db: MongoClient, chosen: str | None) -> None:
    config = await _without_database().get_llm_config(chosen, str(ObjectId()))

    # USE_LOCAL_LLM (the default): everything comes from .env, the free local model.
    assert config.model == settings.LLM_MODEL
    assert config.base_url == settings.LLM_BASE_URL
    assert config.num_ctx == settings.LLM_NUM_CTX
    assert (config.api_key.get_secret_value() if config.api_key else None) == (
        settings.LLM_API_KEY.get_secret_value() if settings.LLM_API_KEY else None
    )


@pytest.mark.anyio
async def test_with_the_local_model_a_request_uses_no_connection(app_db: MongoClient) -> None:
    service = _without_database()

    assert await service.connection_for_request(str(ObjectId()), str(ObjectId()), stored=str(ObjectId())) is None


@pytest.mark.anyio
@pytest.mark.usefixtures("model_connections_only")
@pytest.mark.parametrize("blank", [None, "", "  ", "undefined", "null", "None"])
async def test_without_the_local_model_a_connection_is_required(app_db: MongoClient, blank: str | None) -> None:
    service = _without_database()

    with pytest.raises(ValidationError) as error:
        await service.get_llm_config(blank, str(ObjectId()))
    assert error.value.error_code.value == "MODEL_CONNECTION_REQUIRED"
    with pytest.raises(ValidationError):
        await service.connection_for_request(blank, str(ObjectId()))


@pytest.mark.anyio
@pytest.mark.usefixtures("model_connections_only")
async def test_a_request_uses_the_connection_sent_or_else_the_one_chosen_before(app_db: MongoClient, org_id: str) -> None:
    sent, stored = _seed(app_db, org_id), _seed(app_db, org_id)
    service = ModelConnectionService()

    assert await service.connection_for_request(sent, org_id, stored=stored) == sent
    assert await service.connection_for_request("undefined", org_id, stored=stored) == stored
    with pytest.raises(NotFoundError):  # checked now, not minutes later in the background
        await service.connection_for_request(str(ObjectId()), org_id)


@pytest.mark.anyio
@pytest.mark.usefixtures("model_connections_only")
async def test_enabled_connection_resolves(app_db: MongoClient, org_id: str) -> None:
    config = await ModelConnectionService().get_llm_config(_seed(app_db, org_id), org_id)

    assert config.model == "openai/gpt-4o"
    assert config.base_url == "https://agenticqe.ai/v1"
    assert config.api_key.get_secret_value() == "sk-secret"


@pytest.mark.anyio
@pytest.mark.usefixtures("model_connections_only")
@pytest.mark.parametrize("variant", ["other_org", "not_enabled", "inactive", "provider_inactive", "unknown_id"])
async def test_unusable_connections_are_not_found(app_db: MongoClient, org_id: str, variant: str) -> None:
    connection_id = {
        "other_org": lambda: _seed(app_db, str(ObjectId())),
        "not_enabled": lambda: _seed(app_db, org_id, enabled=False),
        "inactive": lambda: _seed(app_db, org_id, active=False),
        "provider_inactive": lambda: _seed(app_db, org_id, provider_active=False),
        "unknown_id": lambda: str(ObjectId()),
    }[variant]()

    with pytest.raises(NotFoundError) as error:
        await ModelConnectionService().get_llm_config(connection_id, org_id)
    assert error.value.error_code.value == "CONNECTION_NOT_FOUND"


@pytest.mark.anyio
@pytest.mark.usefixtures("model_connections_only")
async def test_malformed_id_is_rejected(app_db: MongoClient) -> None:
    with pytest.raises(ValidationError):
        await ModelConnectionService().get_llm_config("abc", str(ObjectId()))
