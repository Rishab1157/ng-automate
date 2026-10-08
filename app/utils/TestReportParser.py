"""Read test report files into test case results.

Formats: JUnit XML (Maven Surefire/Failsafe, Gradle, pytest --junitxml, Playwright/Cypress/Robot junit or xunit
output) and TestNG's testng-results.xml. Report files come from the project under test: they are untrusted.
Files that are too big, declare a DOCTYPE/ENTITY (XML bombs, external entities) or are malformed are skipped,
never fatal.
"""

import logging
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from app.models.testRunModel import TestCaseResultModel, TestCaseStatus

logger = logging.getLogger(__name__)

MAX_REPORT_BYTES = 20 * 1024 * 1024
MAX_MESSAGE_CHARS = 1000
MAX_DETAILS_CHARS = 4000
# Only the start of a file is checked for declarations: they must come before the root element.
_PROLOG_SCAN_BYTES = 64 * 1024
_FORBIDDEN_DECLARATIONS = (b"<!DOCTYPE", b"<!ENTITY")
_TESTNG_STATUS = {"PASS": TestCaseStatus.PASSED, "FAIL": TestCaseStatus.FAILED, "SKIP": TestCaseStatus.SKIPPED}


def parse_report_file(path: Path) -> list[TestCaseResultModel]:
    """Test cases in one report file; [] when the file is skipped or holds no test cases."""
    root = _load_root(path)
    if root is None:
        return []
    tag = _local_name(root.tag)
    if tag == "testng-results":
        return _parse_testng(root)
    if tag in ("testsuites", "testsuite"):
        return _parse_junit(root)
    return []


def parse_reports(project_dir: Path, report_globs: list[str]) -> tuple[list[TestCaseResultModel], list[str]]:
    """All test cases from the files matching the repo-relative globs, and those files (repo-relative, sorted)."""
    root = project_dir.resolve()
    files: set[Path] = set()
    for pattern in report_globs:
        for match in root.glob(pattern):
            resolved = match.resolve()
            # A link inside the project must not make us read files outside it.
            if match.is_file() and resolved.is_relative_to(root):
                files.add(resolved)

    cases: list[TestCaseResultModel] = []
    report_files: list[str] = []
    for file in sorted(files):
        relative = file.relative_to(root).as_posix()
        report_files.append(relative)
        cases.extend(parse_report_file(file))
    return cases, report_files


def _load_root(path: Path) -> ElementTree.Element | None:
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            logger.warning("Skipping report %s: larger than %d bytes", path.name, MAX_REPORT_BYTES)
            return None
        data = path.read_bytes()
    except OSError:
        logger.warning("Skipping unreadable report %s", path.name)
        return None
    if any(declaration in data[:_PROLOG_SCAN_BYTES] for declaration in _FORBIDDEN_DECLARATIONS):
        logger.warning("Skipping report %s: it declares a DOCTYPE or ENTITY", path.name)
        return None
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError:
        logger.warning("Skipping malformed report %s", path.name)
        return None


def _parse_junit(root: ElementTree.Element) -> list[TestCaseResultModel]:
    cases: list[TestCaseResultModel] = []
    for testcase in root.iter():
        if _local_name(testcase.tag) != "testcase":
            continue
        status, message, details = TestCaseStatus.PASSED, None, None
        for child in testcase:
            kind = _local_name(child.tag)
            if kind in ("failure", "error", "skipped"):
                status = {"failure": TestCaseStatus.FAILED, "error": TestCaseStatus.ERROR}.get(kind, TestCaseStatus.SKIPPED)
                message = child.get("message") or child.get("type")
                details = (child.text or "").strip() or None
                if status != TestCaseStatus.SKIPPED:
                    break
        cases.append(TestCaseResultModel(
            name=testcase.get("name") or "(unnamed)",
            suite=testcase.get("classname") or None,
            status=status,
            duration_seconds=_seconds(testcase.get("time")),
            message=_clip(message, MAX_MESSAGE_CHARS),
            details=_clip(details, MAX_DETAILS_CHARS),
        ))
    return cases


def _parse_testng(root: ElementTree.Element) -> list[TestCaseResultModel]:
    cases: list[TestCaseResultModel] = []
    for test_class in root.iter("class"):
        for method in test_class.iter("test-method"):
            if method.get("is-config") == "true":
                continue
            status = _TESTNG_STATUS.get((method.get("status") or "").upper())
            if status is None:
                continue
            message = details = None
            exception = method.find("exception")
            if exception is not None:
                message = (exception.findtext("message") or "").strip() or exception.get("class")
                details = (exception.findtext("full-stacktrace") or "").strip() or None
            duration_ms = _seconds(method.get("duration-ms"))
            cases.append(TestCaseResultModel(
                name=method.get("name") or "(unnamed)",
                suite=test_class.get("name") or None,
                status=status,
                duration_seconds=duration_ms / 1000 if duration_ms is not None else None,
                message=_clip(message, MAX_MESSAGE_CHARS),
                details=_clip(details, MAX_DETAILS_CHARS),
            ))
    return cases


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _seconds(value: str | None) -> float | None:
    try:
        return float(value.replace(",", "")) if value else None
    except ValueError:
        return None


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"
