"""Tests for error handling in file editor."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.impl import file_editor

from .conftest import assert_error_result


def test_validation_error_formatting(tmp_path):
    """Test that validation errors are properly formatted in the output."""
    missing_file = tmp_path / "nonexistent" / "file.txt"
    result = file_editor(
        command="view",
        path=str(missing_file),
    )
    assert_error_result(result)
    assert result.is_error and "does not exist" in result.text
    # NG Automate: it names the nearest folder that exists and what to try.
    assert f"neither does its folder {missing_file.parent}" in result.text
    assert f"Check the path with `view {tmp_path}`" in result.text
    assert "find the file with glob; use `create` for a new file." in result.text

    # Test directory validation for non-view commands
    result = file_editor(
        command="str_replace",
        path=str(tmp_path),
        old_str="something",
        new_str="new",
    )
    assert_error_result(result)
    assert result.is_error and "only `view` works on folders" in result.text
    assert f"`view {tmp_path}` lists its files" in result.text


def test_missing_file_in_an_existing_folder(tmp_path):
    """NG Automate: the message says how to check the name and how to create it."""
    missing_file = tmp_path / "Login.java"
    result = file_editor(command="str_replace", path=str(missing_file), old_str="a")

    assert_error_result(result)
    assert result.text.endswith(
        f"Invalid `path` parameter: {missing_file}. This path does not exist. Check "
        f"the name with `view {tmp_path}` or find the file with glob; use `create` "
        "for a new file."
    )

    result = file_editor(command="undo_edit", path=str(missing_file))
    assert result.is_error and "use `create`" not in result.text


def test_create_on_an_existing_path(tmp_path):
    """NG Automate: create never overwrites; the message says what to do instead."""
    existing = tmp_path / "a.txt"
    existing.write_text("old")

    on_file = file_editor(command="create", path=str(existing), file_text="new")
    on_folder = file_editor(command="create", path=str(tmp_path), file_text="new")

    assert on_file.is_error and existing.read_text() == "old"
    assert "A file already exists there, and `create` never overwrites" in on_file.text
    assert "edit it with `str_replace`; for a new file, pick another name" in (
        on_file.text
    )
    assert on_folder.is_error and "`create` makes files" in on_folder.text
    assert f"`view {tmp_path}` shows what the folder has" in on_folder.text


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only regression test")
def test_create_rejects_foreign_platform_absolute_paths(tmp_path, monkeypatch):
    """Create should reject absolute-path syntax that is not absolute on this host."""
    monkeypatch.chdir(tmp_path)
    result = file_editor(command="create", path=r"C:\foo", file_text="hello")

    assert_error_result(result)
    assert "absolute path" in result.text
    assert not (tmp_path / r"C:\foo").exists()


def test_str_replace_error_handling(temp_file):
    """Test error handling in str_replace command."""
    # Create a test file
    content = "line 1\nline 2\nline 3\n"
    with open(temp_file, "w") as f:
        f.write(content)

    # Test non-existent string
    result = file_editor(
        command="str_replace",
        path=temp_file,
        old_str="nonexistent",
        new_str="something",
    )
    assert_error_result(result)
    assert result.is_error and "did not appear verbatim" in result.text

    # Test multiple occurrences
    with open(temp_file, "w") as f:
        f.write("line\nline\nother")

    result = file_editor(
        command="str_replace",
        path=temp_file,
        old_str="line",
        new_str="new_line",
    )
    assert_error_result(result)
    assert result.is_error and "Multiple occurrences" in result.text
    assert result.is_error and "lines [1, 2]" in result.text


def test_view_range_validation(temp_file):
    """Test validation of view_range parameter."""
    # Create a test file
    content = "line 1\nline 2\nline 3\n"
    with open(temp_file, "w") as f:
        f.write(content)

    # Test invalid range format
    result = file_editor(
        command="view",
        path=temp_file,
        view_range=[1],  # Should be [start, end]
    )
    assert_error_result(result)
    assert result.is_error and "should be a list of two integers" in result.text
    assert "[start_line, -1] shows the rest of the file" in result.text

    # Test out of bounds range: should clamp to file end and show a warning
    result = file_editor(
        command="view",
        path=temp_file,
        view_range=[1, 10],  # File only has 3 lines
    )
    # This should succeed but show a warning
    assert not result.is_error
    assert (
        "NOTE: We only show up to 3 since there're only 3 lines in this file."
        in result.text
    )

    # Test invalid range order
    result = file_editor(
        command="view",
        path=temp_file,
        view_range=[3, 1],  # End before start
    )
    assert_error_result(result)
    assert result.is_error and "The end line 1 is before the start line 3" in (
        result.text
    )
    assert "Use [3, -1] to see from line 3 to the end of the file" in result.text

    # Start line beyond the end of the file: the message says how many lines
    result = file_editor(command="view", path=temp_file, view_range=[5, 9])
    assert_error_result(result)
    assert "The file has 3 lines, so the start line must be 1 to 3" in result.text
    assert "Use for example [1, 3], or view the file without view_range" in (
        result.text
    )

    # view_range on a folder
    result = file_editor(
        command="view", path=str(Path(temp_file).parent), view_range=[1, 2]
    )
    assert_error_result(result)
    assert "view_range works on files only" in result.text


def test_insert_validation(temp_file):
    """Test validation in insert command."""
    # Create a test file
    content = "line 1\nline 2\nline 3\n"
    with open(temp_file, "w") as f:
        f.write(content)

    # Test insert at negative line
    result = file_editor(
        command="insert",
        path=temp_file,
        insert_line=-1,
        new_str="new line",
    )
    assert_error_result(result)
    assert result.is_error and "insert_line must be 0 to 3" in result.text

    # Test insert beyond file length
    result = file_editor(
        command="insert",
        path=temp_file,
        insert_line=10,
        new_str="new line",
    )
    assert_error_result(result)
    assert (
        "The file has 3 lines, so insert_line must be 0 to 3: new_str goes after "
        "that line (0 puts it at the top, 3 at the end)."
    ) in result.text


def test_undo_validation(temp_file):
    """Test undo_edit validation."""
    # Create a test file
    content = "line 1\nline 2\nline 3\n"
    with open(temp_file, "w") as f:
        f.write(content)

    # Try to undo without any previous edits
    result = file_editor(
        command="undo_edit",
        path=temp_file,
    )
    assert_error_result(result)
    assert result.is_error and "No edit history found" in result.text


def test_view_directory_permission_error_returns_error_observation():
    """Directory view should return an error observation on PermissionError."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        editor = FileEditor()
        with patch.object(
            editor,
            "_count_hidden_children",
            side_effect=PermissionError("denied"),
        ):
            result = editor.view(path)
        assert result.is_error
        assert "denied" in result.text


def test_view_subdirectory_permission_error_skips_inaccessible_dir():
    """Subdirectory permission errors should be silently skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        sub = path / "sub"
        sub.mkdir()
        (path / "visible.txt").write_text("hello")

        # Simulate iterdir on the subdirectory raising PermissionError.
        original_iterdir = Path.iterdir

        def patched_iterdir(self: Path):
            if self == sub:
                raise PermissionError("denied")
            return original_iterdir(self)

        editor = FileEditor()
        with patch.object(Path, "iterdir", patched_iterdir):
            result = editor.view(path)
        assert not result.is_error
        assert "visible.txt" in result.text
