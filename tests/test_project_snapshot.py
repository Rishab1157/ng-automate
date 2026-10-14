"""ProjectSnapshot: what an agent changed in the project copy."""

from pathlib import Path

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
