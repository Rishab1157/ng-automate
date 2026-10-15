"""Where a generation stands, worked out from the files the generator wrote: which case is in which file, the cases
written twice, the cases no file names and how the last check ended. Plain code on a tmp project, no LLM."""

import json
import re
from pathlib import Path

import pytest

from app.agents.TestGeneratorAgent.GenerationState import (
    MAX_ERROR_LINES_IN_STATE,
    MAX_FILE_BYTES,
    MAX_LISTED_FILES,
    collect_state,
    describe_state,
)
from app.models.healerModel import FileChangeModel

CHECK = "npx playwright test --list"
LOGIN_TEST = "test('TD-001 Valid login', async ({ page }) => {});\n"


def _written(root: Path, files: dict[str, str | bytes]) -> list[FileChangeModel]:
    """Writes `files` into the project, as the generator would, and returns them as its changes."""
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return [FileChangeModel(path=path, change="added") for path in files]


def _line(text: str, part: str) -> str:
    """The one line of `text` that holds `part`."""
    [line] = [line for line in text.splitlines() if part in line]
    return line


# ---------- which file holds which case ----------

@pytest.mark.parametrize("path", [
    "tests/ngauto/TD-001.spec.ts",
    "tests/ngauto/td_001.spec.ts",
    "tests/ngauto/TD001.spec.ts",
    "tests/ngauto/td-001-valid-login.spec.ts",
])
def test_a_case_id_in_a_file_name_counts_whatever_its_case_and_separators(tmp_path: Path, path: str) -> None:
    changes = _written(tmp_path, {path: "test('valid login', async ({ page }) => {});\n"})

    state = collect_state(tmp_path, changes, ["TD-001"], ["TD-001"])

    assert state.case_files == {"TD-001": [path]}


@pytest.mark.parametrize(("path", "content"), [
    ("tests/ngauto/login.spec.ts", LOGIN_TEST),
    ("tests/ngauto/test_login.py", "def test_td_001_valid_login(page):\n    pass\n"),
    ("src/test/java/com/shop/ngauto/LoginTest.java", "@Test\npublic void TD001ValidLogin() {}\n"),
    ("src/test/resources/features/ngauto/login.feature", "Scenario: td-001-valid-login\n"),
])
def test_a_case_id_in_a_files_content_counts(tmp_path: Path, path: str, content: str) -> None:
    changes = _written(tmp_path, {path: content})

    state = collect_state(tmp_path, changes, ["TD-001"], ["TD-001"])

    assert state.case_files == {"TD-001": [path]}


def test_a_longer_id_never_counts_for_a_shorter_one(tmp_path: Path) -> None:
    path = "tests/ngauto/td-0010-search.spec.ts"
    changes = _written(tmp_path, {path: "test('TD-0010 Search', async ({ page }) => {});\n"})

    state = collect_state(tmp_path, changes, ["TD-001", "TD-0010"], ["TD-001", "TD-0010"])

    assert state.case_files == {"TD-0010": [path]}


def test_a_numeric_id_counts_only_in_file_names(tmp_path: Path) -> None:
    changes = _written(tmp_path, {
        "tests/ngauto/test_7_login.py": "def test_login(page):\n    pass\n",
        "tests/ngauto/test_cart.py": "def test_cart(cart):\n    assert cart.count() == 7\n",
        "tests/ngauto/test_17_search.py": "def test_search(page):\n    pass\n",
    })

    state = collect_state(tmp_path, changes, ["7"], ["7"])

    assert state.case_files == {"7": ["tests/ngauto/test_7_login.py"]}


def test_a_dotted_numeric_id_never_counts_from_a_number_in_the_code(tmp_path: Path) -> None:
    # An id without letters counts by file name only, with its separator: 12 (the date below) is not case 1.2.
    login, logout = "tests/ngauto/case-1.1-valid-login.spec.ts", "tests/ngauto/case-1.2-logout.spec.ts"
    changes = _written(tmp_path, {
        login: "test('1.1 Valid login', async ({ page }) => {\n  await page.fill('#birthday', '2026-12-01');\n});\n",
        logout: "test('1.2 Logout', async ({ page }) => {});\n",
    })

    state = collect_state(tmp_path, changes, ["1.1", "1.2"], ["1.1", "1.2"])

    assert state.case_files == {"1.1": [login], "1.2": [logout]}
    assert state.duplicates == {}


def test_reports_and_data_files_never_count_as_a_cases_test(tmp_path: Path) -> None:
    every_case = "TD-001 Valid login, TD-002 Logout"
    changes = _written(tmp_path, {
        "playwright-report/index.html": f"<html><body>{every_case}</body></html>\n",
        "test-results/results.json": json.dumps({"tests": every_case}),
        "test-results/ngauto-td-002-logout-chromium/error-context.md": every_case,
        "tests/ngauto/login.spec.ts": LOGIN_TEST,
    })

    state = collect_state(tmp_path, changes, ["TD-001", "TD-002"], ["TD-001", "TD-002"])

    assert state.case_files == {"TD-001": ["tests/ngauto/login.spec.ts"]}


def test_a_jbehave_story_holds_its_cases_like_a_feature_file(tmp_path: Path) -> None:
    path = "src/test/resources/stories/ngauto/td_001_valid_login.story"
    changes = _written(tmp_path, {path: "Scenario: TD-001 Valid login\nGiven the login page is open\n"})

    state = collect_state(tmp_path, changes, ["TD-001"], ["TD-001"])

    assert state.case_files == {"TD-001": [path]}


def test_files_the_generator_deleted_are_neither_listed_nor_counted(tmp_path: Path) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": LOGIN_TEST})
    changes.append(FileChangeModel(path="tests/ngauto/td-001-copy.spec.ts", change="deleted"))

    state = collect_state(tmp_path, changes, ["TD-001"], ["TD-001"])

    assert state.files == ["tests/ngauto/login.spec.ts"]
    assert state.case_files == {"TD-001": ["tests/ngauto/login.spec.ts"]}


def test_a_file_too_big_to_read_counts_by_its_name_only(tmp_path: Path) -> None:
    path = "tests/ngauto/td-001-bundle.spec.ts"
    changes = _written(tmp_path, {path: "// TD-002\n" + "x" * MAX_FILE_BYTES})

    state = collect_state(tmp_path, changes, ["TD-001", "TD-002"], [])

    assert state.case_files == {"TD-001": [path]}


def test_a_file_that_is_not_utf8_is_still_read(tmp_path: Path) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": b"\xff\xfe// \x80\ntest('TD-001 Valid login');\n"})

    state = collect_state(tmp_path, changes, ["TD-001"], [])

    assert state.case_files == {"TD-001": ["tests/ngauto/login.spec.ts"]}


# ---------- what the model is told ----------

def test_before_anything_was_written_there_is_no_state(tmp_path: Path) -> None:
    state = collect_state(tmp_path, [], ["TD-001", "TD-002"], [], CHECK)

    assert describe_state(state) == ""


def test_a_case_written_twice_is_a_duplicate_with_a_note_to_keep_one_file(tmp_path: Path) -> None:
    first, second = "tests/ngauto/td-001-valid-login.spec.ts", "tests/ngauto/validLogin.spec.ts"
    changes = _written(tmp_path, {first: LOGIN_TEST, second: "test('TD001 valid login', async ({ page }) => {});\n"})

    state = collect_state(tmp_path, changes, ["TD-001"], ["TD-001"])

    assert state.duplicates == {"TD-001": [first, second]}
    note = _line(describe_state(state), "keep")
    assert "TD-001" in note and first in note and second in note
    assert re.search(r"\brm\b", note)


def test_a_done_case_that_no_file_names_gets_a_note(tmp_path: Path) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": LOGIN_TEST})

    text = describe_state(collect_state(tmp_path, changes, ["TD-001", "TD-002", "TD-003"], ["TD-001", "TD-002"]))

    assert "tests/ngauto/login.spec.ts" in _line(text, "TD-001")  # written, and where
    missing = _line(text, "none of your files names")
    assert "TD-002" in missing
    assert "TD-001" not in missing and "TD-003" not in missing  # TD-003's batch has not been asked for yet


def test_the_files_written_are_listed_up_to_a_limit(tmp_path: Path) -> None:
    extra = 5
    changes = _written(tmp_path, {f"tests/ngauto/case-{n:02d}.spec.ts": "" for n in range(MAX_LISTED_FILES + extra)})

    listed = _line(describe_state(collect_state(tmp_path, changes, [], [])), "case-00.spec.ts")

    assert f"case-{MAX_LISTED_FILES - 1:02d}.spec.ts" in listed
    assert f"case-{MAX_LISTED_FILES:02d}.spec.ts" not in listed
    assert f"{extra} more" in listed


@pytest.mark.parametrize(("passed", "expected"), [(None, "not run"), (True, "passed")])
def test_the_state_says_how_the_last_check_ended(tmp_path: Path, passed: bool | None, expected: str) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": LOGIN_TEST})

    text = describe_state(collect_state(tmp_path, changes, ["TD-001"], ["TD-001"], CHECK, passed))

    line = _line(text, CHECK)
    assert expected in line
    assert "failed" not in line


def test_a_failed_check_shows_its_first_error_lines(tmp_path: Path) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": LOGIN_TEST})
    errors = [f"tests/ngauto/login.spec.ts:{n}:5 - error TS2304: Cannot find name 'step{n}'." for n in range(1, 6)]

    text = describe_state(collect_state(tmp_path, changes, ["TD-001"], ["TD-001"], CHECK, False, "\n".join(errors)))

    line = _line(text, CHECK)
    assert "failed" in line
    assert all(error in line for error in errors[:MAX_ERROR_LINES_IN_STATE])
    assert not any(error in line for error in errors[MAX_ERROR_LINES_IN_STATE:])


def test_a_project_without_a_quick_check_gets_no_check_line(tmp_path: Path) -> None:
    changes = _written(tmp_path, {"tests/ngauto/login.spec.ts": LOGIN_TEST})

    checked = describe_state(collect_state(tmp_path, changes, ["TD-001"], ["TD-001"], CHECK, True))
    unchecked = describe_state(collect_state(tmp_path, changes, ["TD-001"], ["TD-001"], None, None, None))

    assert unchecked.splitlines() == [line for line in checked.splitlines() if CHECK not in line]
    assert "None" not in unchecked
