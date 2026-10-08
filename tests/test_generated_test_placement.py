"""Where generated tests go and how only they are run: selectors per stack, the @ngauto tag, skipped cases."""

from pathlib import Path

import pytest

from app.agents.RunnerAgent.TestCommandResolver import resolve_test_command
from app.agents.TestGeneratorAgent.GeneratedTestPlacement import (
    GENERATED_TAG,
    generated_selector,
    placement_rule,
    runnable_cases,
    tag_untagged_features,
)
from app.models.analyzerModel import Confidence, FactModel, FactSheetModel, FactSource
from app.models.healerModel import FileChangeModel
from app.models.testDataModel import TestCaseSpecModel, TestStepModel


def _fact(value: str | list[str] | None) -> FactModel:
    return FactModel(value=value, source=FactSource.CODE, evidence=[], confidence=Confidence.HIGH)


def _sheet(build: str | None, frameworks: list[str], bdd: str | None = None) -> FactSheetModel:
    return FactSheetModel(
        total_files=1, language_files={}, primary_language=_fact(None), build_tool=_fact(build),
        test_frameworks=_fact(frameworks), automation_tools=_fact(["Selenium"]), bdd_tool=_fact(bdd),
        marker_files=[], test_dirs=[], feature_file_count=0, top_level_tree=[],
    )


CUCUMBER = _sheet("Maven", ["TestNG"], "Cucumber")
MAVEN = _sheet("Maven", ["TestNG"])
GRADLE = _sheet("Gradle", ["JUnit 5"])
PYTEST = _sheet("pip", ["pytest"])
PLAYWRIGHT = _sheet("npm", ["Playwright Test"])
CYPRESS = _sheet("npm", ["Cypress"])


def _added(*paths: str) -> list[FileChangeModel]:
    return [FileChangeModel(path=path, change="added") for path in paths]


@pytest.mark.parametrize(
    ("sheet", "files", "expected"),
    [
        (CUCUMBER, _added("src/test/resources/features/login_ngauto.feature"), GENERATED_TAG),
        (CUCUMBER, _added("src/test/java/steps/NewSteps.java"), None),
        (MAVEN, _added("src/test/java/com/acme/ngauto/LoginTest.java", "src/test/java/com/acme/ngauto/CartTest.java"), "**/ngauto/*"),
        (MAVEN, _added("src/test/java/com/acme/LoginTest.java"), None),
        (GRADLE, _added("src/test/java/com/acme/ngauto/LoginTest.java"), "*.ngauto.*"),
        (PYTEST, _added("tests/ngauto/test_login.py", "tests/ngauto/test_cart.py"), "tests/ngauto"),
        (PYTEST, _added("tests/ngauto/test_login.py", "other/ngauto/test_cart.py"), None),
        (PLAYWRIGHT, _added("tests/ngauto/login.spec.ts"), "tests/ngauto"),
        (CYPRESS, _added("cypress/e2e/ngauto/login.cy.js"), "cypress/e2e/ngauto/**"),
        (PYTEST, [FileChangeModel(path="tests/ngauto/test_login.py", change="modified")], None),
    ],
)
def test_selector_runs_only_the_generated_tests(sheet: FactSheetModel, files: list[FileChangeModel], expected: str | None) -> None:
    assert generated_selector(sheet, files) == expected


@pytest.mark.parametrize(
    ("sheet", "files", "fragment"),
    [
        (CUCUMBER, _added("f/ngauto.feature"), "-Dcucumber.filter.tags=@ngauto"),
        (MAVEN, _added("src/test/java/a/ngauto/LoginTest.java"), "-Dtest=**/ngauto/*"),
        (GRADLE, _added("src/test/java/a/ngauto/LoginTest.java"), "--tests '*.ngauto.*'"),
        (PYTEST, _added("tests/ngauto/test_a.py"), "tests/ngauto"),
    ],
)
def test_selectors_are_accepted_by_the_command_resolver(sheet: FactSheetModel, files: list[FileChangeModel], fragment: str) -> None:
    command = resolve_test_command(sheet, generated_selector(sheet, files))

    assert fragment in command.command


def test_placement_rule_names_the_tag_or_folder() -> None:
    assert GENERATED_TAG in placement_rule(CUCUMBER)
    assert "package named 'ngauto'" in placement_rule(MAVEN)
    assert "folder named 'ngauto'" in placement_rule(PYTEST)


def test_cases_with_a_step_without_locator_are_left_out() -> None:
    good = TestCaseSpecModel(id="A", title="ok", steps=[TestStepModel(action="click", target="id=go")])
    bad = TestCaseSpecModel(
        id="B", title="missing", steps=[TestStepModel(action="open", target="https://x"), TestStepModel(action="click", needs_locator=True)]
    )

    runnable, skipped = runnable_cases([good, bad])

    assert [case.id for case in runnable] == ["A"]
    assert skipped == [("B", "step 2 (click) has no locator in the test data")]


def test_untagged_new_feature_gets_the_tag(tmp_path: Path) -> None:
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "new.feature").write_text("@smoke\nFeature: Login\n  Scenario: ok\n", encoding="utf-8")
    (tmp_path / "f" / "tagged.feature").write_text("@ngauto\nFeature: Cart\n", encoding="utf-8")
    (tmp_path / "f" / "old.feature").write_text("Feature: Old\n", encoding="utf-8")
    files = [*_added("f/new.feature", "f/tagged.feature"), FileChangeModel(path="f/old.feature", change="modified")]

    tagged = tag_untagged_features(tmp_path, files)

    assert tagged == ["f/new.feature"]
    assert (tmp_path / "f" / "new.feature").read_text(encoding="utf-8") == "@smoke\n@ngauto\nFeature: Login\n  Scenario: ok\n"
    assert (tmp_path / "f" / "old.feature").read_text(encoding="utf-8") == "Feature: Old\n"
