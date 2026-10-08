"""Check the analyzer's evidence against the real project files.

A small model sometimes cites files that do not exist. After the analysis, every evidence path and important path
is checked on the host copy of the project: what does not exist (or points outside the project) is removed, and
an answer that lost its evidence loses confidence. Plain code, no LLM.
"""

import re
from pathlib import Path, PurePosixPath

from app.agents.AnalyzerAgent.AnalyzerPrompts import project_relative_path
from app.models.analyzerModel import AnalyzerFindingsModel, Confidence, FactModel

# "src/a.py:12" or "src/a.py:12-30" -> "src/a.py"
_LINE_SUFFIX = re.compile(r":\d+(-\d+)?$")


def verify_evidence(findings: AnalyzerFindingsModel, project_dir: Path) -> tuple[AnalyzerFindingsModel, list[str]]:
    """The findings with only real evidence, and the paths that were removed (in order, without duplicates)."""
    root = project_dir.resolve()
    removed: list[str] = []
    updates: dict[str, object] = {}

    for name, value in findings:
        if isinstance(value, FactModel):
            kept = [entry for entry in value.evidence if _exists(root, entry) or _note(removed, entry)]
            updates[name] = value.model_copy(update={
                "evidence": kept,
                "confidence": _confidence(value.confidence, had=len(value.evidence), kept=len(kept)),
            })

    updates["important_paths"] = [
        entry for entry in findings.important_paths if _exists(root, entry.path) or _note(removed, entry.path)
    ]
    return findings.model_copy(update=updates), removed


def _exists(root: Path, entry: str) -> bool:
    relative = _LINE_SUFFIX.sub("", project_relative_path(entry.strip().replace("\\", "/"))).removeprefix("./")
    if not relative or relative == ".":
        return True
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or ":" in relative:
        return False
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    # A link may not lead outside the project.
    return resolved.is_relative_to(root) and resolved.exists()


def _note(removed: list[str], entry: str) -> bool:
    if entry not in removed:
        removed.append(entry)
    return False


def _confidence(confidence: Confidence, had: int, kept: int) -> Confidence:
    if had and not kept:
        return Confidence.LOW
    if kept < had and confidence == Confidence.HIGH:
        return Confidence.MEDIUM
    return confidence
