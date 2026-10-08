"""ProjectSnapshot: what an agent changed in the project copy, and undoing one file."""

from pathlib import Path

import pytest

from app.utils.ProjectSnapshot import MAX_FILE_BYTES, ProjectSnapshot, project_fingerprint


def _project(root: Path) -> Path:
    (root / "src").mkdir()
    (root / "pom.xml").write_text("<project>\n  <version>1.0</version>\n</project>\n", encoding="utf-8")
    (root / "src" / "LoginTest.java").write_text("class LoginTest {}\n", encoding="utf-8")
    return root


def test_no_changes_after_capture(tmp_path: Path) -> None:
    snapshot = ProjectSnapshot.capture(_project(tmp_path))

    assert snapshot.changes() == []
    assert snapshot.content("pom.xml") is not None


def test_modified_added_and_deleted_files_are_reported_in_path_order(tmp_path: Path) -> None:
    root = _project(tmp_path)
    snapshot = ProjectSnapshot.capture(root)

    (root / "pom.xml").write_text("<project>\n  <version>2.0</version>\n</project>\n", encoding="utf-8")
    (root / "src" / "LoginTest.java").unlink()
    (root / "README.md").write_text("hello\n", encoding="utf-8")

    changes = snapshot.changes()

    assert [(c.path, c.change) for c in changes] == [
        ("README.md", "added"),
        ("pom.xml", "modified"),
        ("src/LoginTest.java", "deleted"),
    ]
    pom = changes[1]
    assert "-  <version>1.0</version>" in pom.diff
    assert "+  <version>2.0</version>" in pom.diff
    assert pom.diff.startswith("--- a/pom.xml")


def test_build_outputs_binaries_and_large_files_are_ignored(tmp_path: Path) -> None:
    root = _project(tmp_path)
    snapshot = ProjectSnapshot.capture(root)

    for folder in ("target", "node_modules", "build", "ngauto-results"):
        (root / folder).mkdir()
        (root / folder / "out.txt").write_text("generated\n", encoding="utf-8")
    (root / "driver.bin").write_bytes(b"\x00\x01\x02")
    (root / "huge.log").write_text("x" * (MAX_FILE_BYTES + 1), encoding="utf-8")

    assert snapshot.changes() == []


def test_restore_puts_a_modified_file_back(tmp_path: Path) -> None:
    root = _project(tmp_path)
    original = (root / "src" / "LoginTest.java").read_text(encoding="utf-8")
    snapshot = ProjectSnapshot.capture(root)
    (root / "src" / "LoginTest.java").write_text("class LoginTest { /* gone */ }\n", encoding="utf-8")

    snapshot.restore("src/LoginTest.java")

    assert (root / "src" / "LoginTest.java").read_text(encoding="utf-8") == original
    assert snapshot.changes() == []


def test_restore_brings_back_a_deleted_file_and_removes_an_added_one(tmp_path: Path) -> None:
    root = _project(tmp_path)
    snapshot = ProjectSnapshot.capture(root)
    (root / "src" / "LoginTest.java").unlink()
    (root / "src" / "Extra.java").write_text("class Extra {}\n", encoding="utf-8")

    snapshot.restore("src/LoginTest.java")
    snapshot.restore("src/Extra.java")

    assert snapshot.changes() == []


def test_restore_keeps_windows_line_endings(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"one\r\ntwo\r\n")
    snapshot = ProjectSnapshot.capture(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"changed\n")

    snapshot.restore("a.txt")

    assert (tmp_path / "a.txt").read_bytes() == b"one\r\ntwo\r\n"


def test_restore_refuses_a_path_outside_the_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    snapshot = ProjectSnapshot.capture(_project(project))

    with pytest.raises(ValueError):
        snapshot.restore("../outside.txt")
    assert not (tmp_path / "outside.txt").exists()


def test_fingerprint_changes_with_the_files_and_ignores_build_outputs(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project>1</project>\n", encoding="utf-8")
    first = project_fingerprint(tmp_path)
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "report.txt").write_text("run at 10:42\n", encoding="utf-8")

    assert project_fingerprint(tmp_path) == first
    (tmp_path / "pom.xml").write_text("<project>2</project>\n", encoding="utf-8")
    assert project_fingerprint(tmp_path) != first
    (tmp_path / "pom.xml").write_text("<project>1</project>\n", encoding="utf-8")
    assert project_fingerprint(tmp_path) == first  # back to the same files, the same fingerprint
