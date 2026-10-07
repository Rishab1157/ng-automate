"""STUB (interface only) — replaced by the runs-api build step."""

from typing import Any

from app.models.runModel import RunEventLevel, RunEventModel, RunEventType, RunModel, RunStage


class RunService:
    async def create_run(
        self, *, project_id: str, org_id: str, user_id: str, model_connection_id: str | None
    ) -> RunModel:
        raise NotImplementedError

    async def get(self, run_id: str, org_id: str) -> RunModel:
        raise NotImplementedError

    async def get_by_id(self, run_id: str) -> RunModel:
        raise NotImplementedError

    async def get_events(self, run_id: str, org_id: str, after_seq: int = 0, limit: int = 200) -> list[RunEventModel]:
        raise NotImplementedError

    async def find_unfinished(self) -> list[RunModel]:
        raise NotImplementedError

    async def mark_running(self, run_id: str) -> None:
        raise NotImplementedError

    async def set_stage(self, run_id: str, stage: RunStage) -> None:
        raise NotImplementedError

    async def save_outputs(self, run_id: str, **outputs: Any) -> None:
        raise NotImplementedError

    async def set_sandbox_container(self, run_id: str, container_id: str | None) -> None:
        raise NotImplementedError

    async def mark_completed(self, run_id: str) -> None:
        raise NotImplementedError

    async def mark_failed(self, run_id: str, code: str, message: str) -> None:
        raise NotImplementedError

    async def append_event(
        self,
        run_id: str,
        org_id: str,
        type: RunEventType,
        message: str,
        level: RunEventLevel = RunEventLevel.INFO,
        data: dict[str, Any] | None = None,
    ) -> RunEventModel:
        raise NotImplementedError
