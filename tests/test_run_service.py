import asyncio
import importlib
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from bson import ObjectId
from pymongo import MongoClient

from app.config import settings
from app.core.exceptions import ErrorCode, ErrorMessages, NotFoundError, ValidationError
from app.db.indexes import ensure_indexes
from app.dto.runDto import RunEventResponseDTO
from app.models.analyzerModel import (
    AnalyzerFindingsModel,
    Confidence,
    FactModel,
    FactSheetModel,
    FactSource,
    ImportantPathModel,
)
from app.models.projectModel import ArchiveModel, ProjectMapper, ProjectSource
from app.models.projectProfileModel import ProjectProfileMapper
from app.models.runModel import RunEventLevel, RunEventType, RunOutputsModel, RunStage, RunStatus
from app.services.projectProfileService import ProjectProfileService
from app.services.runService import RunService
from app.services.runService.RunService import _OUTPUT_DUMPERS
from tests.conftest import NG_MODULE_ID

# ---------- helpers (also used by test_run_endpoint.py) ----------


def seed_project(mongo: MongoClient, org_id: str) -> str:
    db_model = ProjectMapper.to_create_db_model(
        project_id=ObjectId(),
        org_id=org_id,
        user_id=str(ObjectId()),
        name="demo",
        source=ProjectSource.UPLOAD,
        archive=ArchiveModel(size_bytes=100, sha256="0" * 64, file_count=2),
    )
    mongo[settings.MONGO_DATABASE].projects.insert_one(db_model.model_dump(by_alias=True))
    return str(db_model.id)


def seed_model_connection(mongo: MongoClient, org_id: str, api_key: str = "sk-secret") -> str:
    """An AgenticQE connection of `org_id`, enabled for NG Automate."""
    qxcel = mongo[settings.QXCEL_DATABASE]
    provider_id = qxcel.model_providers.insert_one(
        {"provider_code": "AgenticQE", "api_base_url": "https://agenticqe.ai/v1", "is_active": True}
    ).inserted_id
    type_id = qxcel.model_types.insert_one({"type_code": "gpt-4o", "model_provider_id": provider_id}).inserted_id
    return str(qxcel.model_connections.insert_one({
        "org_id": ObjectId(org_id),
        "userdefined_name": "Gateway",
        "model_provider_id": provider_id,
        "model_type_id": type_id,
        "api_key": api_key,
        "is_active": True,
        "available_for_module_ids": [NG_MODULE_ID],
        "active_for_module_ids": [NG_MODULE_ID],
    }).inserted_id)


def _fact(value: str | list[str] | None, evidence: list[str], source: FactSource = FactSource.CODE) -> FactModel:
    return FactModel(value=value, source=source, evidence=evidence, confidence=Confidence.HIGH if value else Confidence.LOW)


def make_fact_sheet(build_tool: str = "Maven") -> FactSheetModel:
    return FactSheetModel(
        total_files=3,
        language_files={"Java": 2, "Gherkin": 1},
        primary_language=_fact("Java", ["src/test/java/LoginTest.java"]),
        build_tool=_fact(build_tool, ["pom.xml"]),
        test_frameworks=_fact(["TestNG"], ["testng.xml"]),
        automation_tools=_fact(["Selenium"], ["pom.xml"]),
        bdd_tool=_fact(None, []),
        marker_files=["pom.xml", "testng.xml"],
        test_dirs=["src/test"],
        feature_file_count=1,
        top_level_tree=["pom.xml", "src/", "src/test/"],
    )


def make_findings(summary: str = "Login tests for a web shop") -> AnalyzerFindingsModel:
    return AnalyzerFindingsModel(
        project_summary=_fact(summary, ["README.md"], FactSource.LLM),
        architecture_pattern=_fact("Page Object Model", ["src/test/java/pages/LoginPage.java"], FactSource.LLM),
        test_command=_fact("mvn test", ["pom.xml"], FactSource.LLM),
        reporting_tools=_fact(["Allure"], ["pom.xml"], FactSource.LLM),
        important_paths=[ImportantPathModel(path="src/test/java/pages", role="page objects")],
        open_questions=["Which environment do the tests target?"],
    )


def seed_profile(
    mongo: MongoClient, org_id: str, project_id: str, created_at: datetime, llm_model: str = "openai/gpt-4o"
) -> str:
    db_model = ProjectProfileMapper.to_create_db_model(
        org_id=org_id,
        project_id=project_id,
        run_id=str(ObjectId()),
        fact_sheet=make_fact_sheet(),
        findings=make_findings(),
        llm_model=llm_model,
    )
    doc = db_model.model_dump(by_alias=True) | {"created_at": created_at}
    mongo[settings.MONGO_DATABASE].project_profiles.insert_one(doc)
    return str(db_model.id)


def _runs(mongo: MongoClient):
    return mongo[settings.MONGO_DATABASE].runs


async def _new_run(mongo: MongoClient, org_id: str, service: RunService | None = None) -> str:
    service = service or RunService()
    run = await service.create_run(
        project_id=seed_project(mongo, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id=None
    )
    return run.id


# ---------- create_run ----------

@pytest.mark.anyio
async def test_create_run_inserts_a_queued_run(app_db: MongoClient, org_id: str) -> None:
    project_id = seed_project(app_db, org_id)
    user_id = str(ObjectId())

    run = await RunService().create_run(project_id=project_id, org_id=org_id, user_id=user_id, model_connection_id=None)

    assert (run.org_id, run.project_id, run.created_by) == (org_id, project_id, user_id)
    assert run.status == RunStatus.QUEUED
    assert run.stage == RunStage.QUEUED
    assert run.model_connection_id is None
    assert run.outputs == RunOutputsModel()
    assert run.started_at is None and run.finished_at is None and run.error is None
    doc = _runs(app_db).find_one({"_id": ObjectId(run.id)})
    assert doc["org_id"] == ObjectId(org_id)
    assert doc["project_id"] == ObjectId(project_id)
    assert doc["event_seq"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("blank", [None, "", "   ", "undefined", "null", "None", "NULL"])
async def test_blank_model_connection_is_stored_as_none(app_db: MongoClient, org_id: str, blank: str | None) -> None:
    run = await RunService().create_run(
        project_id=seed_project(app_db, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id=blank
    )

    assert run.model_connection_id is None
    assert _runs(app_db).find_one({"_id": ObjectId(run.id)})["model_connection_id"] is None


@pytest.mark.anyio
async def test_enabled_model_connection_is_stored_by_id_only(app_db: MongoClient, org_id: str) -> None:
    connection_id = seed_model_connection(app_db, org_id, api_key="sk-very-secret")

    run = await RunService().create_run(
        project_id=seed_project(app_db, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id=connection_id
    )

    assert run.model_connection_id == connection_id
    doc = _runs(app_db).find_one({"_id": ObjectId(run.id)})
    assert doc["model_connection_id"] == ObjectId(connection_id)
    assert "sk-very-secret" not in str(doc)


@pytest.mark.anyio
async def test_model_connection_of_another_org_is_not_found(app_db: MongoClient, org_id: str) -> None:
    connection_id = seed_model_connection(app_db, str(ObjectId()))

    with pytest.raises(NotFoundError) as error:
        await RunService().create_run(
            project_id=seed_project(app_db, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id=connection_id
        )
    assert error.value.error_code == ErrorCode.CONNECTION_NOT_FOUND
    assert _runs(app_db).count_documents({}) == 0


@pytest.mark.anyio
async def test_malformed_model_connection_is_rejected(app_db: MongoClient, org_id: str) -> None:
    with pytest.raises(ValidationError):
        await RunService().create_run(
            project_id=seed_project(app_db, org_id), org_id=org_id, user_id=str(ObjectId()), model_connection_id="abc"
        )
    assert _runs(app_db).count_documents({}) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("variant", ["other_org", "unknown", "malformed"])
async def test_project_must_exist_in_the_org(app_db: MongoClient, org_id: str, variant: str) -> None:
    project_id = {
        "other_org": lambda: seed_project(app_db, str(ObjectId())),
        "unknown": lambda: str(ObjectId()),
        "malformed": lambda: "not-an-id",
    }[variant]()

    with pytest.raises(NotFoundError) as error:
        await RunService().create_run(project_id=project_id, org_id=org_id, user_id=str(ObjectId()), model_connection_id=None)
    assert error.value.message == ErrorMessages.PROJECT_NOT_FOUND
    assert _runs(app_db).count_documents({}) == 0


# ---------- reading runs ----------

@pytest.mark.anyio
async def test_get_is_scoped_to_the_org(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    assert (await service.get(run_id, org_id)).id == run_id
    for bad_id, org in [(run_id, str(ObjectId())), (str(ObjectId()), org_id), ("not-an-id", org_id)]:
        with pytest.raises(NotFoundError) as error:
            await service.get(bad_id, org)
        assert error.value.message == ErrorMessages.RUN_NOT_FOUND


@pytest.mark.anyio
async def test_get_by_id_has_no_org_check(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    assert (await service.get_by_id(run_id)).org_id == org_id
    with pytest.raises(NotFoundError):
        await service.get_by_id(str(ObjectId()))


@pytest.mark.anyio
async def test_find_unfinished_returns_queued_and_running_oldest_first(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    queued, running, completed, failed = [await _new_run(app_db, org_id, service) for _ in range(4)]
    await service.mark_running(running)
    await service.mark_completed(completed)
    await service.mark_failed(failed, ErrorCode.ANALYSIS_FAILED.value, "nope")

    assert [run.id for run in await service.find_unfinished()] == [queued, running]


# ---------- state changes ----------

@pytest.mark.anyio
async def test_mark_running_keeps_the_first_start_time(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    await service.mark_running(run_id)
    first = await service.get_by_id(run_id)
    await asyncio.sleep(0.05)  # clearly later, even with a coarse Windows clock
    await service.mark_running(run_id)  # resumed after a restart
    second = await service.get_by_id(run_id)

    assert first.status == second.status == RunStatus.RUNNING
    assert first.started_at is not None
    assert second.started_at == first.started_at
    assert second.updated_at > first.updated_at


@pytest.mark.anyio
async def test_stage_and_sandbox_container_are_saved(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    created = await service.get_by_id(run_id)

    await service.set_stage(run_id, RunStage.ANALYZING)
    await service.set_sandbox_container(run_id, "c0ffee")
    during = await service.get_by_id(run_id)
    await service.set_sandbox_container(run_id, None)
    after = await service.get_by_id(run_id)

    assert during.stage == RunStage.ANALYZING
    assert during.sandbox_container_id == "c0ffee"
    assert during.updated_at >= created.updated_at
    assert after.sandbox_container_id is None


@pytest.mark.anyio
async def test_mark_completed(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    await service.mark_completed(run_id)
    run = await service.get_by_id(run_id)

    assert run.status == RunStatus.COMPLETED
    assert run.stage == RunStage.COMPLETED
    assert run.finished_at is not None
    assert run.error is None


@pytest.mark.anyio
@pytest.mark.parametrize("code", [ErrorCode.SANDBOX_FAILED, ErrorCode.SANDBOX_FAILED.value])
async def test_mark_failed_records_the_error(app_db: MongoClient, org_id: str, code: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    await service.mark_failed(run_id, code, "The sandbox could not be started: timeout")
    run = await service.get_by_id(run_id)

    assert run.status == RunStatus.FAILED
    assert run.stage == RunStage.FAILED
    assert run.finished_at is not None
    assert run.error is not None
    assert (run.error.code, run.error.message) == ("SANDBOX_FAILED", "The sandbox could not be started: timeout")


@pytest.mark.anyio
async def test_mark_failed_caps_a_long_message(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    await service.mark_failed(run_id, "ANALYSIS_FAILED", "x" * 5000)

    assert len((await service.get_by_id(run_id)).error.message) == 1000


# ---------- outputs ----------

@pytest.mark.anyio
async def test_save_outputs_round_trip(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    profile_id = str(ObjectId())
    project_dir = Path(settings.DATA_DIR) / "runs" / run_id / "project"

    # One stage at a time, like the master agent: earlier outputs stay.
    await service.save_outputs(run_id, project_dir=project_dir)
    await service.save_outputs(run_id, fact_sheet=make_fact_sheet())
    await service.save_outputs(run_id, findings=make_findings(), llm_model="openai/gpt-4o")
    await service.save_outputs(run_id, profile_id=profile_id)
    outputs = (await service.get_by_id(run_id)).outputs

    assert outputs.project_dir == str(project_dir)
    assert outputs.fact_sheet == make_fact_sheet()
    assert outputs.findings == make_findings()
    assert outputs.llm_model == "openai/gpt-4o"
    assert outputs.profile_id == profile_id
    stored = _runs(app_db).find_one({"_id": ObjectId(run_id)})["outputs"]
    assert stored["profile_id"] == ObjectId(profile_id)
    assert stored["fact_sheet"]["primary_language"]["source"] == "code"
    assert stored["findings"]["project_summary"]["confidence"] == "high"


@pytest.mark.anyio
async def test_save_outputs_accepts_plain_dicts_and_none_clears(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    await service.save_outputs(run_id, fact_sheet=make_fact_sheet().model_dump(mode="json"), project_dir="/tmp/x")
    await service.save_outputs(run_id, project_dir=None)
    outputs = (await service.get_by_id(run_id)).outputs

    assert outputs.fact_sheet == make_fact_sheet()
    assert outputs.project_dir is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outputs",
    [
        {"container_id": "abc"},
        {"profile_id": "not-an-id"},
        {"fact_sheet": {"total_files": "many"}},
        {"findings": {"project_summary": "no evidence"}},
    ],
)
async def test_save_outputs_rejects_bad_input(app_db: MongoClient, org_id: str, outputs: dict) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    with pytest.raises(ValueError):
        await service.save_outputs(run_id, **outputs)
    assert (await service.get_by_id(run_id)).outputs == RunOutputsModel()


def test_output_keys_match_the_run_outputs_model() -> None:
    assert set(_OUTPUT_DUMPERS) == set(RunOutputsModel.model_fields)


# ---------- events ----------

@pytest.mark.anyio
async def test_concurrent_events_get_seq_1_to_n(app_db: MongoClient, org_id: str) -> None:
    await ensure_indexes()  # the unique (run_id, seq) index would reject a duplicate seq
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    events = await asyncio.gather(*(
        service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, f"step {i}") for i in range(50)
    ))

    assert sorted(event.seq for event in events) == list(range(1, 51))
    stored = app_db[settings.MONGO_DATABASE].run_events.find({"run_id": ObjectId(run_id)}).sort("seq", 1)
    assert [doc["seq"] for doc in stored] == list(range(1, 51))
    assert _runs(app_db).find_one({"_id": ObjectId(run_id)})["event_seq"] == 50


@pytest.mark.anyio
async def test_append_event_stores_the_event(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    event = await service.append_event(
        run_id, org_id, RunEventType.AGENT_OBSERVATION, "grep found 3 files", RunEventLevel.DETAIL, {"tool": "grep"}
    )

    assert (event.run_id, event.seq, event.type, event.level) == (run_id, 1, RunEventType.AGENT_OBSERVATION, RunEventLevel.DETAIL)
    assert event.message == "grep found 3 files"
    assert event.data == {"tool": "grep"}
    doc = app_db[settings.MONGO_DATABASE].run_events.find_one({"run_id": ObjectId(run_id)})
    assert doc["org_id"] == ObjectId(org_id)
    assert (doc["type"], doc["level"]) == ("agent_observation", "detail")


@pytest.mark.anyio
async def test_append_event_accepts_plain_strings_for_type_and_level(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    event = await service.append_event(run_id, org_id, "stage_started", "Preparing the workspace", level="info")

    assert (event.type, event.level) == (RunEventType.STAGE_STARTED, RunEventLevel.INFO)
    assert event.data == {}


@pytest.mark.anyio
async def test_long_event_text_is_truncated_and_data_made_safe(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    cyclic: dict = {}
    cyclic["self"] = cyclic

    event = await service.append_event(
        run_id,
        org_id,
        RunEventType.AGENT_OBSERVATION,
        "m" * 5000,
        RunEventLevel.DETAIL,
        {
            "output": "o" * 10_000,
            "nested": {"items": ["i" * 5000, 1, 2.5, True, None]},
            "path": Path("src") / "Main.java",
            "tags": ("a", "b"),
            "nan": math.nan,
            "huge": 2**70,
            "cyclic": cyclic,
        },
    )
    [stored] = await service.get_events(run_id, org_id)

    assert len(event.message) == 1000 and event.message.endswith("[truncated]")
    assert len(event.data["output"]) == 4000 and event.data["output"].endswith("[truncated]")
    assert len(event.data["nested"]["items"][0]) == 4000
    assert event.data["nested"]["items"][1:] == [1, 2.5, True, None]
    assert event.data["path"] == str(Path("src") / "Main.java")
    assert event.data["tags"] == ["a", "b"]
    assert event.data["nan"] == "nan"
    assert event.data["huge"] == str(2**70)
    assert stored.message == event.message
    assert stored.data == event.data


@pytest.mark.anyio
async def test_event_data_keys_are_made_mongo_and_json_safe(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    # A JSON Schema in tool args: PyMongo would read {"$ref", "$id"} back as a DBRef, which is not JSON.
    data = {"args": {"schema": {"$ref": "#/defs/page", "$id": "login"}}, "bad\x00key": 1}

    await service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, "view schema.json", data=data)
    [stored] = await service.get_events(run_id, org_id)

    assert stored.data == {"args": {"schema": {"＄ref": "#/defs/page", "＄id": "login"}}, "badkey": 1}
    RunEventResponseDTO.from_model(stored).model_dump_json()  # the events endpoint can return it


@pytest.mark.anyio
async def test_every_update_sets_updated_at(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    updates = [
        lambda: service.mark_running(run_id),
        lambda: service.set_stage(run_id, RunStage.ANALYZING),
        lambda: service.save_outputs(run_id, llm_model="openai/gpt-4o"),
        lambda: service.set_sandbox_container(run_id, "c0ffee"),
        lambda: service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, "x"),
        lambda: service.mark_failed(run_id, "ANALYSIS_FAILED", "nope"),
        lambda: service.mark_completed(run_id),
    ]

    for update in updates:
        before = (await service.get_by_id(run_id)).updated_at
        await asyncio.sleep(0.02)  # MongoDB keeps milliseconds; a coarse Windows clock needs a clear gap
        await update()
        assert (await service.get_by_id(run_id)).updated_at > before


@pytest.mark.anyio
async def test_short_event_text_is_kept_exactly(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    event = await service.append_event(run_id, org_id, RunEventType.AGENT_MESSAGE, "m" * 1000, data={"o": "o" * 4000})

    assert event.message == "m" * 1000
    assert event.data["o"] == "o" * 4000


@pytest.mark.anyio
async def test_append_event_needs_the_run_in_that_org(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)

    for bad_run, bad_org in [(run_id, str(ObjectId())), (str(ObjectId()), org_id)]:
        with pytest.raises(NotFoundError):
            await service.append_event(bad_run, bad_org, RunEventType.AGENT_ACTION, "x")

    assert app_db[settings.MONGO_DATABASE].run_events.count_documents({}) == 0
    assert _runs(app_db).find_one({"_id": ObjectId(run_id)})["event_seq"] == 0


@pytest.mark.anyio
async def test_get_events_pages_in_seq_order(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    other_run_id = await _new_run(app_db, org_id, service)
    for i in range(5):
        await service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, f"e{i + 1}")
    await service.append_event(other_run_id, org_id, RunEventType.AGENT_ACTION, "other run")

    first_page = await service.get_events(run_id, org_id, after_seq=0, limit=2)
    next_page = await service.get_events(run_id, org_id, after_seq=first_page[-1].seq, limit=2)
    rest = await service.get_events(run_id, org_id, after_seq=4)
    nothing_new = await service.get_events(run_id, org_id, after_seq=5)

    assert [(e.seq, e.message) for e in first_page] == [(1, "e1"), (2, "e2")]
    assert [e.seq for e in next_page] == [3, 4]
    assert [e.seq for e in rest] == [5]
    assert nothing_new == []


@pytest.mark.anyio
async def test_get_events_limit_is_kept_in_bounds(
    app_db: MongoClient, org_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    for i in range(3):
        await service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, f"e{i}")
    # The module, not the class of the same name that the package re-exports.
    monkeypatch.setattr(importlib.import_module("app.services.runService.RunService"), "MAX_EVENTS_LIMIT", 2)

    # 0 must not mean "no limit" to MongoDB, and a huge limit is capped.
    assert len(await service.get_events(run_id, org_id, limit=0)) == 1
    assert len(await service.get_events(run_id, org_id, limit=10_000)) == 2


@pytest.mark.anyio
async def test_get_events_is_scoped_to_the_org(app_db: MongoClient, org_id: str) -> None:
    service = RunService()
    run_id = await _new_run(app_db, org_id, service)
    await service.append_event(run_id, org_id, RunEventType.AGENT_ACTION, "x")

    for bad_run, bad_org in [(run_id, str(ObjectId())), ("not-an-id", org_id)]:
        with pytest.raises(NotFoundError) as error:
            await service.get_events(bad_run, bad_org)
        assert error.value.message == ErrorMessages.RUN_NOT_FOUND


# ---------- project profiles ----------

@pytest.mark.anyio
async def test_profile_create_then_get_latest(app_db: MongoClient, org_id: str) -> None:
    service = ProjectProfileService()
    project_id = seed_project(app_db, org_id)
    run_id = str(ObjectId())

    created = await service.create(
        org_id=org_id, project_id=project_id, run_id=run_id,
        fact_sheet=make_fact_sheet(), findings=make_findings(), llm_model="openai/gpt-4o",
    )
    latest = await service.get_latest(project_id, org_id)

    assert latest.id == created.id
    assert (latest.org_id, latest.project_id, latest.run_id) == (org_id, project_id, run_id)
    assert latest.fact_sheet == make_fact_sheet()
    assert latest.findings == make_findings()
    assert latest.llm_model == "openai/gpt-4o"
    doc = app_db[settings.MONGO_DATABASE].project_profiles.find_one({"_id": ObjectId(created.id)})
    assert (doc["org_id"], doc["project_id"], doc["run_id"]) == (ObjectId(org_id), ObjectId(project_id), ObjectId(run_id))


@pytest.mark.anyio
async def test_profile_create_accepts_plain_dicts(app_db: MongoClient, org_id: str) -> None:
    project_id = seed_project(app_db, org_id)

    created = await ProjectProfileService().create(
        org_id=org_id, project_id=project_id, run_id=str(ObjectId()),
        fact_sheet=make_fact_sheet().model_dump(mode="json"), findings=make_findings().model_dump(mode="json"),
        llm_model="ollama/devstral",
    )

    assert created.fact_sheet == make_fact_sheet()
    assert created.findings == make_findings()


@pytest.mark.anyio
async def test_latest_profile_is_the_newest(app_db: MongoClient, org_id: str) -> None:
    project_id = seed_project(app_db, org_id)
    now = datetime.now(UTC)
    seed_profile(app_db, org_id, project_id, now - timedelta(hours=1), llm_model="old")
    newest = seed_profile(app_db, org_id, project_id, now, llm_model="new")
    seed_profile(app_db, org_id, seed_project(app_db, org_id), now + timedelta(hours=1), llm_model="other project")

    latest = await ProjectProfileService().get_latest(project_id, org_id)

    assert (latest.id, latest.llm_model) == (newest, "new")


@pytest.mark.anyio
async def test_project_without_profile_has_none(app_db: MongoClient, org_id: str) -> None:
    with pytest.raises(NotFoundError) as error:
        await ProjectProfileService().get_latest(seed_project(app_db, org_id), org_id)
    assert error.value.message == ErrorMessages.PROFILE_NOT_FOUND


@pytest.mark.anyio
@pytest.mark.parametrize("variant", ["other_org", "malformed"])
async def test_profile_needs_the_project_in_the_org(app_db: MongoClient, org_id: str, variant: str) -> None:
    other_org = str(ObjectId())
    project_id = seed_project(app_db, other_org) if variant == "other_org" else "not-an-id"
    if variant == "other_org":
        seed_profile(app_db, other_org, project_id, datetime.now(UTC))

    with pytest.raises(NotFoundError) as error:
        await ProjectProfileService().get_latest(project_id, org_id)
    assert error.value.message == ErrorMessages.PROJECT_NOT_FOUND
