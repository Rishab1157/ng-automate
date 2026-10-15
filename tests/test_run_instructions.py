"""The user's own instructions for a run: checked when the run is created, stored on generate and test runs (an
analysis run has no agent that follows them), shown back by the API and handed to the master agent."""

from collections.abc import Callable
from typing import Any

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
from pymongo import MongoClient

from app.agents.MasterAgent.MasterState import initial_state
from app.config import settings
from app.dto.runDto import CreateRunDTO, RunResponseDTO
from app.models.runModel import MAX_RUN_INSTRUCTIONS_CHARS, RunMapper, RunMode, RunOutputsModel
from app.models.testDataModel import TestCaseSpecModel, TestDataFormat, TestDataMapper, TestDataSetModel, TestStepModel
from app.services.runService import RunService
from app.services.runService.RunService import _run_instructions
from tests.test_run_endpoint import FakeRunExecutor
from tests.test_run_service import seed_project

BASE = "/tx-agents/ng-automate/run"
INSTRUCTIONS = "write the new tests as BDD feature files"


def _doc(instructions: str | None = None) -> dict[str, Any]:
    """A new generate run's document, as stored."""
    db_model = RunMapper.to_create_db_model(
        org_id=str(ObjectId()), project_id=str(ObjectId()), user_id=str(ObjectId()), model_connection_id=None,
        mode=RunMode.GENERATE, test_data_id=str(ObjectId()), instructions=instructions,
    )
    return db_model.model_dump(by_alias=True)


def _seed_ready_test_data(mongo: MongoClient, org_id: str, project_id: str) -> str:
    """Test data in our own format (ready at once) with one case the generator can write."""
    case = TestCaseSpecModel(
        id="TD-001", title="Valid login", steps=[TestStepModel(action="open", target="https://shop.example")]
    )
    db_model = TestDataMapper.to_create_db_model(
        org_id=org_id, project_id=project_id, user_id=str(ObjectId()), filename="cases.json",
        source_format=TestDataFormat.JSON, data_set=TestDataSetModel(source_format=TestDataFormat.JSON, cases=[case]),
    )
    mongo[settings.MONGO_DATABASE].test_data.insert_one(db_model.model_dump(by_alias=True))
    return str(db_model.id)


def _runs(mongo: MongoClient):
    return mongo[settings.MONGO_DATABASE].runs


@pytest.fixture
def executor(monkeypatch: pytest.MonkeyPatch) -> FakeRunExecutor:
    fake = FakeRunExecutor()
    monkeypatch.setattr("app.api.endpoints.RunEndpoint.RunEndpoint.run_executor", fake)
    return fake


# ---------- the request ----------

@pytest.mark.parametrize(("given", "kept"), [
    (f"  {INSTRUCTIONS}  ", INSTRUCTIONS),
    ("reuse the login steps\nkeep the page objects\n", "reuse the login steps\nkeep the page objects"),
    (" \n\t ", None),
    ("", None),
    (None, None),
])
def test_the_request_trims_the_instructions_and_drops_blank_ones(given: str | None, kept: str | None) -> None:
    dto = CreateRunDTO(
        project_id=str(ObjectId()), mode=RunMode.GENERATE, test_data_id=str(ObjectId()), instructions=given
    )

    assert dto.instructions == kept


def test_a_request_without_instructions_has_none() -> None:
    assert CreateRunDTO(project_id=str(ObjectId()), mode=RunMode.TEST).instructions is None


def test_instructions_longer_than_the_limit_are_rejected() -> None:
    longest = "a" * MAX_RUN_INSTRUCTIONS_CHARS

    assert CreateRunDTO(project_id=str(ObjectId()), mode=RunMode.TEST, instructions=longest).instructions == longest
    with pytest.raises(PydanticValidationError):
        CreateRunDTO(project_id=str(ObjectId()), mode=RunMode.TEST, instructions=longest + "a")


# ---------- stored, read back, shown and handed on ----------

def test_the_run_document_keeps_the_instructions_and_reads_them_back() -> None:
    doc = _doc(INSTRUCTIONS)

    assert doc["instructions"] == INSTRUCTIONS
    assert RunMapper.to_model(doc).instructions == INSTRUCTIONS


def test_a_run_stored_before_instructions_existed_has_none() -> None:
    doc = _doc()
    del doc["instructions"]

    assert RunMapper.to_model(doc).instructions is None


@pytest.mark.parametrize("instructions", [INSTRUCTIONS, None])
def test_the_api_shows_the_instructions_the_run_was_given(instructions: str | None) -> None:
    run = RunMapper.to_model(_doc(instructions))

    assert RunResponseDTO.from_model(run).instructions == instructions


def test_the_master_agent_gets_the_instructions_even_when_the_work_starts_over() -> None:
    run = RunMapper.to_model(_doc(INSTRUCTIONS)).model_copy(
        update={"outputs": RunOutputsModel(project_dir="/data/runs/x/project", profile_id=str(ObjectId()))}
    )

    state = initial_state(run)

    assert state["instructions"] == INSTRUCTIONS
    assert "project_dir" not in state  # a generate run whose tests are not saved yet starts over on a fresh copy
    assert initial_state(RunMapper.to_model(_doc()))["instructions"] is None


# ---------- the run service ----------

@pytest.mark.parametrize(("mode", "given", "stored"), [
    (RunMode.GENERATE, f"  {INSTRUCTIONS} \n", INSTRUCTIONS),
    (RunMode.TEST, "reuse the existing login steps", "reuse the existing login steps"),
    (RunMode.TEST, " \n ", None),
    (RunMode.GENERATE, None, None),
    (RunMode.ANALYZE, INSTRUCTIONS, None),
])
def test_only_runs_with_an_agent_that_follows_them_keep_the_instructions(
    mode: RunMode, given: str | None, stored: str | None
) -> None:
    assert _run_instructions(mode, given) == stored


@pytest.mark.anyio
@pytest.mark.parametrize("mode", [RunMode.TEST, RunMode.GENERATE])
async def test_generate_and_test_runs_store_the_users_instructions(
    app_db: MongoClient, org_id: str, mode: RunMode
) -> None:
    project_id = seed_project(app_db, org_id)
    test_data_id = _seed_ready_test_data(app_db, org_id, project_id) if mode == RunMode.GENERATE else None
    service = RunService()

    run = await service.create_run(
        project_id=project_id, org_id=org_id, user_id=str(ObjectId()), model_connection_id=None, mode=mode,
        test_data_id=test_data_id, instructions=f"  {INSTRUCTIONS}  ",
    )

    assert run.instructions == INSTRUCTIONS
    assert _runs(app_db).find_one({"_id": ObjectId(run.id)})["instructions"] == INSTRUCTIONS
    assert (await service.get(run.id, org_id)).instructions == INSTRUCTIONS
    assert [item.instructions for item in await service.list_for_project(project_id, org_id)] == [INSTRUCTIONS]


@pytest.mark.anyio
@pytest.mark.parametrize(("mode", "given"), [(RunMode.ANALYZE, INSTRUCTIONS), (RunMode.TEST, " \n\t ")])
async def test_an_analysis_run_or_blank_instructions_store_none(
    app_db: MongoClient, org_id: str, mode: RunMode, given: str
) -> None:
    run = await RunService().create_run(
        project_id=seed_project(app_db, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id=None,
        mode=mode, instructions=given,
    )

    assert run.instructions is None
    assert _runs(app_db).find_one({"_id": ObjectId(run.id)})["instructions"] is None


# ---------- the API ----------

def test_the_api_takes_shows_and_limits_the_instructions(
    client: TestClient,
    mongo: MongoClient,
    executor: FakeRunExecutor,
    auth_headers: Callable[..., dict[str, str]],
    org_id: str,
) -> None:
    headers = auth_headers(org_id)
    project_id = seed_project(mongo, org_id)
    body = {"project_id": project_id, "mode": "test"}
    longest = "a" * MAX_RUN_INSTRUCTIONS_CHARS

    too_long = client.post(f"{BASE}/", headers=headers, json=body | {"instructions": longest + "a"})
    created = client.post(f"{BASE}/", headers=headers, json=body | {"instructions": f"  {INSTRUCTIONS}  "})

    assert too_long.status_code == 422
    assert created.status_code == 202, created.text
    run_id = created.json()["id"]
    assert created.json()["instructions"] == INSTRUCTIONS
    assert client.get(f"{BASE}/{run_id}", headers=headers).json()["instructions"] == INSTRUCTIONS
    assert executor.submitted == [run_id]
