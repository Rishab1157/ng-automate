"""The vendored file editor's str_replace fallback keeps the file's own indentation (third_party/README.md, item 2).

Before the fix, an old_str with a guessed indentation (8 spaces where the file has 4) matched only after strip(),
but new_str was written with its 8 spaces on top of the file's 4: 12 spaces, and every retry added more.
"""

from pathlib import Path

import pytest
from openhands.tools.file_editor.editor import FileEditor, _drop_unmatched_whitespace
from openhands.tools.file_editor.exceptions import ToolError

POM = (
    "<project>\n"
    "  <dependencies>\n"
    "    <dependency><artifactId>testng</artifactId><version>99.0.0</version></dependency>\n"
    "  </dependencies>\n"
    "</project>\n"
)
LINE = "<dependency><artifactId>testng</artifactId><version>{}</version></dependency>"


def _replace(tmp_path: Path, old: str, new: str, content: str = POM) -> list[str]:
    path = tmp_path / "pom.xml"
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    FileEditor(workspace_root=str(tmp_path))(command="str_replace", path=str(path), old_str=old, new_str=new)
    return path.read_text(encoding="utf-8").splitlines()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_guessed_indentation_is_not_added_to_the_file_indentation(tmp_path: Path) -> None:
    # The model guesses 8 spaces (the file has 4) in both old_str and new_str: before the fix this gave 4 + 8 = 12.
    lines = _replace(tmp_path, " " * 8 + LINE.format("99.0.0"), " " * 8 + LINE.format("7.10.2"))

    assert lines[2] == " " * 4 + LINE.format("7.10.2")


def test_a_retry_with_the_same_guess_does_not_drift(tmp_path: Path) -> None:
    _replace(tmp_path, " " * 8 + LINE.format("99.0.0"), " " * 8 + LINE.format("7.10.2"))

    lines = _replace(tmp_path, " " * 8 + LINE.format("7.10.2"), " " * 4 + LINE.format("7.10.2"))

    assert _indent(lines[2]) == 4


def test_extra_indentation_asked_for_on_purpose_is_kept(tmp_path: Path) -> None:
    # old_str guessed 2 spaces too many; new_str asks for 2 more than old_str: the relative change survives.
    lines = _replace(tmp_path, " " * 6 + LINE.format("99.0.0"), " " * 8 + LINE.format("7.10.2"))

    assert _indent(lines[2]) == 6


def test_trailing_whitespace_from_old_str_is_not_written(tmp_path: Path) -> None:
    lines = _replace(tmp_path, LINE.format("99.0.0") + "  \n\n", LINE.format("7.10.2") + "  \n\n")

    assert lines[2] == " " * 4 + LINE.format("7.10.2")
    assert lines[3] == "  </dependencies>"


def test_a_verbatim_match_is_unchanged(tmp_path: Path) -> None:
    lines = _replace(tmp_path, " " * 4 + LINE.format("99.0.0"), " " * 6 + LINE.format("7.10.2"))

    assert _indent(lines[2]) == 6


def test_no_match_still_fails(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        FileEditor(workspace_root=str(tmp_path))(
            command="str_replace", path=str(_write(tmp_path)), old_str="  <missing/>", new_str="  <x/>"
        )


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("        a", "        b", "b"),
        ("        a", "    b", "b"),
        ("  a", "      b", "    b"),
        ("\t\ta", "    b", "    b"),  # different whitespace: nothing to drop
        (" a", "B  ", "B  "),  # upstream regression test: a Markdown hard line break in new_str stays
        ("a  ", "b  ", "b"),
        ("  a  ", "   ", ""),
        ("a", "b", "b"),
    ],
)
def test_drop_unmatched_whitespace(old: str, new: str, expected: str) -> None:
    assert _drop_unmatched_whitespace(old, new) == expected


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "pom.xml"
    path.write_text(POM, encoding="utf-8")
    return path
