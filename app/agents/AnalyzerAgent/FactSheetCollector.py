"""STUB (interface only) — replaced by the fact-sheet build step."""

from pathlib import Path

from app.models.analyzerModel import FactSheetModel


def collect_fact_sheet(project_dir: Path) -> FactSheetModel:
    raise NotImplementedError
