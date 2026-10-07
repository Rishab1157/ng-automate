"""STUB (interface only) — replaced by the analyzer-agent build step."""

from dataclasses import dataclass, field
from typing import Any

from app.models.runModel import RunEventLevel, RunEventType


@dataclass(frozen=True)
class AgentEvent:
    type: RunEventType
    level: RunEventLevel
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def map_openhands_event(event: Any) -> AgentEvent | None:
    raise NotImplementedError
