import os
from pathlib import Path

import pytest

from app.agents.AnalyzerAgent.EvidenceVerifier import verify_evidence
from app.models.analyzerModel import AnalyzerFindingsModel, Confidence, FactModel, FactSource, ImportantPathModel


def _fact(evidence: list[str], confidence: Confidence = Confidence.HIGH) -> FactModel:
    return FactModel(value="x", source=FactSource.LLM, evidence=evidence, confidence=confidence)


def _findings(**facts: FactModel) -> AnalyzerFindingsModel:
    base = {name: _fact([]) for name in ("project_summary", "architecture_pattern", "test_command", "reporting_tools")}
    return AnalyzerFindingsModel(**{**base, **facts})


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for relative in ("pom.xml", "README.md", "src/test/java/LoginTest.java"):
        file = tmp_path / "project" / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x")
    (tmp_path / "secret.txt").write_text("outside")
    return tmp_path / "project"


def test_real_paths_in_every_form_are_kept(project: Path) -> None:
    evidence = ["pom.xml", "./README.md", "/workspace/project/src/test/java/LoginTest.java", "src/test/java/LoginTest.java:12",
                "src/test/java/LoginTest.java:3-9", "src/test", "src\\test\\java\\LoginTest.java"]

    findings, removed = verify_evidence(_findings(project_summary=_fact(evidence)), project)

    assert removed == []
    assert findings.project_summary.evidence == evidence
    assert findings.project_summary.confidence == Confidence.HIGH


def test_missing_files_are_removed_and_confidence_drops(project: Path) -> None:
    findings, removed = verify_evidence(
        _findings(
            project_summary=_fact(["pom.xml", "src/main/java/Imaginary.java"]),
            test_command=_fact(["build.gradle"], Confidence.MEDIUM),
        ),
        project,
    )

    assert findings.project_summary.evidence == ["pom.xml"]
    assert findings.project_summary.confidence == Confidence.MEDIUM
    assert findings.test_command.evidence == []
    assert findings.test_command.confidence == Confidence.LOW
    assert removed == ["src/main/java/Imaginary.java", "build.gradle"]


@pytest.mark.parametrize("escape", ["../secret.txt", "/etc/passwd", "C:/Windows/win.ini", "src/../../secret.txt"])
def test_paths_outside_the_project_are_removed(project: Path, escape: str) -> None:
    findings, removed = verify_evidence(_findings(project_summary=_fact(["pom.xml", escape])), project)

    assert findings.project_summary.evidence == ["pom.xml"]
    assert removed == [escape]


def test_link_leading_outside_is_removed(project: Path) -> None:
    link = project / "escape.txt"
    try:
        os.symlink(project.parent / "secret.txt", link)
    except OSError:
        pytest.skip("creating symlinks needs extra rights on this machine")

    _, removed = verify_evidence(_findings(project_summary=_fact(["escape.txt"])), project)

    assert removed == ["escape.txt"]


def test_important_paths_are_filtered(project: Path) -> None:
    findings = _findings().model_copy(update={"important_paths": [
        ImportantPathModel(path="src/test/java", role="tests"),
        ImportantPathModel(path="src/test/java/pages", role="page objects"),
    ]})

    cleaned, removed = verify_evidence(findings, project)

    assert [p.path for p in cleaned.important_paths] == ["src/test/java"]
    assert removed == ["src/test/java/pages"]


def test_answers_without_evidence_keep_their_confidence(project: Path) -> None:
    findings, removed = verify_evidence(_findings(reporting_tools=_fact([], Confidence.LOW)), project)

    assert findings.reporting_tools.confidence == Confidence.LOW
    assert removed == []
