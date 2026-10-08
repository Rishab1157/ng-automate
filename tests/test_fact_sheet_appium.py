"""Fact sheet: Appium (mobile) projects are recognised from their dependencies, capability files and app builds."""

import json
from pathlib import Path

from app.agents.AnalyzerAgent.FactSheetCollector import collect_fact_sheet


def _write(root: Path, files: dict[str, str | bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


def _tools(fact_sheet) -> list[str]:
    value = fact_sheet.automation_tools.value
    return [value] if isinstance(value, str) else list(value or [])


POM = """<project><modelVersion>4.0.0</modelVersion><groupId>a</groupId><artifactId>b</artifactId><version>1</version>
<dependencies><dependency><groupId>io.appium</groupId><artifactId>java-client</artifactId><version>9.3.0</version></dependency>
<dependency><groupId>org.testng</groupId><artifactId>testng</artifactId><version>7.10.2</version></dependency></dependencies></project>"""


def test_java_appium_project_with_capabilities_and_an_apk(tmp_path: Path) -> None:
    root = _write(tmp_path, {
        "pom.xml": POM,
        "src/test/java/com/acme/LoginTest.java": "class LoginTest {}",
        "src/test/resources/capabilities.json": json.dumps({"platformName": "Android", "appium:automationName": "UiAutomator2"}),
        "apps/app-debug.apk": b"PK\x03\x04 not a real apk",
    })

    fact_sheet = collect_fact_sheet(root)

    assert "Appium" in _tools(fact_sheet)
    assert "apps/app-debug.apk" in fact_sheet.marker_files
    assert "src/test/resources/capabilities.json" in fact_sheet.marker_files


def test_python_appium_project(tmp_path: Path) -> None:
    root = _write(tmp_path, {
        "requirements.txt": "Appium-Python-Client==4.2.0\npytest==8.3.3\n",
        "tests/test_login.py": "def test_login():\n    assert True\n",
        "app/Shop.ipa": b"not a real ipa",
    })

    fact_sheet = collect_fact_sheet(root)

    assert "Appium" in _tools(fact_sheet)
    assert "app/Shop.ipa" in fact_sheet.marker_files


def test_capability_file_alone_is_evidence_of_appium(tmp_path: Path) -> None:
    root = _write(tmp_path, {
        "package.json": json.dumps({"name": "m", "devDependencies": {"webdriverio": "^9.0.0"}}),
        "config/ios.caps.json": json.dumps({"platformName": "iOS"}),
    })

    fact_sheet = collect_fact_sheet(root)

    assert "Appium" in _tools(fact_sheet)


def test_web_project_is_not_mistaken_for_appium(tmp_path: Path) -> None:
    root = _write(tmp_path, {
        "package.json": json.dumps({"name": "w", "devDependencies": {"@playwright/test": "^1.48.0"}}),
        "playwright.config.ts": "export default {}",
        "tests/recaps.json": "{}",
    })

    fact_sheet = collect_fact_sheet(root)

    assert "Appium" not in _tools(fact_sheet)
