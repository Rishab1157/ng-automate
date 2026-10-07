"""STUB (interface only) — replaced by the analyzer-agent build step."""

from collections.abc import Callable

from openhands.sdk.workspace import BaseWorkspace

from app.agents.AnalyzerAgent.AgentEventMapper import AgentEvent
from app.config import settings
from app.models.analyzerModel import AnalyzerFindingsModel, FactSheetModel
from app.models.llmModel import LlmConfigModel


def parse_findings(text: str) -> AnalyzerFindingsModel:
    raise NotImplementedError


class AnalyzerAgent:
    def __init__(self, llm_config: LlmConfigModel, max_iterations: int = settings.ANALYZER_MAX_ITERATIONS) -> None:
        self.llm_config = llm_config
        self.max_iterations = max_iterations

    def analyze(
        self,
        workspace: BaseWorkspace,
        fact_sheet: FactSheetModel,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> AnalyzerFindingsModel:
        raise NotImplementedError
