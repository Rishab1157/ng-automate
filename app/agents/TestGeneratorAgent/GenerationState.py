"""Where a generation stands, worked out from the project's files after each batch. Plain code, no LLM.

A small model loses track of its own work: after the condenser summarized its earlier steps, it wrote the same test
case again under another name (valid-login, validLogin, valid_login) instead of reusing the file it had written. So
every later batch and every build-fix message tells it, from the files themselves (not from a summary): which cases
are written and in which files, which files it created or changed, how the last build check ended, and which cases it
wrote more than once.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.models.healerModel import FileChangeModel

# Files that can hold a test or the code around it. Reports, logs and data files (an HTML report repeats every test's
# name) never count as a case's test.
CODE_SUFFIXES = {
    ".java", ".kt", ".kts", ".groovy", ".scala", ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".rb", ".cs",
    ".go", ".rs", ".php", ".robot", ".resource", ".feature", ".story",
}
MAX_FILE_BYTES = 200_000
MAX_LISTED_FILES = 30
MAX_ERROR_LINES_IN_STATE = 3


@dataclass
class GenerationState:
    """The generator's work so far: case ids it was given, the files it created or changed (project-relative), the
    files each case id appears in, and how the last build check ended (None: not run yet)."""

    cases_done: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    case_files: dict[str, list[str]] = field(default_factory=dict)
    check: str | None = None
    check_passed: bool | None = None
    check_errors: str | None = None

    @property
    def duplicates(self) -> dict[str, list[str]]:
        """Case ids that appear in more than one file the generator wrote."""
        return {case_id: paths for case_id, paths in self.case_files.items() if len(paths) > 1}


def collect_state(
    project_root: Path,
    changes: Sequence[FileChangeModel],
    case_ids: Sequence[str],
    cases_done: Sequence[str],
    check: str | None = None,
    check_passed: bool | None = None,
    check_errors: str | None = None,
) -> GenerationState:
    """The state from the files the generator changed (`changes`, from the project snapshot) and the run's case ids."""
    files = sorted(change.path for change in changes if change.change != "deleted")
    patterns = {case_id: _id_pattern(case_id) for case_id in case_ids}
    case_files: dict[str, list[str]] = {}
    for path in files:
        if Path(path).suffix.lower() not in CODE_SUFFIXES:
            continue
        content = _read(project_root / path)
        for case_id, pattern in patterns.items():
            # An id without letters ("7", "1.2") would match any such number in the code: only the file name counts.
            text = f"{path}\n{content}" if _has_letters(case_id) else path
            if pattern is not None and pattern.search(text):
                case_files.setdefault(case_id, []).append(path)
    return GenerationState(
        cases_done=list(cases_done),
        files=files,
        case_files=case_files,
        check=check,
        check_passed=check_passed,
        check_errors=check_errors,
    )


def describe_state(state: GenerationState) -> str:
    """The state as a few lines for the model, or "" before anything was written."""
    if not state.files and not state.cases_done:
        return ""
    lines = ["Where the work stands (worked out from the project's files: trust it over your memory):"]
    if state.cases_done:
        written = [f"{case_id} in {_join(state.case_files.get(case_id, []))}" for case_id in state.cases_done
                   if state.case_files.get(case_id)]
        missing = [case_id for case_id in state.cases_done if not state.case_files.get(case_id)]
        if written:
            lines.append(f"- Cases already written: {'; '.join(written)}.")
        if missing:
            lines.append(
                f"- Cases asked for earlier that none of your files names: {', '.join(missing)}. Write one only if it "
                "is really missing; if you wrote it under another name, put its case id in that test's name instead."
            )
    if state.files:
        shown = state.files[:MAX_LISTED_FILES]
        more = f" (and {len(state.files) - MAX_LISTED_FILES} more)" if len(state.files) > MAX_LISTED_FILES else ""
        lines.append(f"- Files you created or changed: {', '.join(shown)}{more}.")
    if state.check:
        if state.check_passed is None:
            lines.append(f"- The check `{state.check}` has not run yet.")
        elif state.check_passed:
            lines.append(f"- The last check `{state.check}` passed.")
        else:
            errors = (state.check_errors or "").strip().splitlines()[:MAX_ERROR_LINES_IN_STATE]
            first = f": {' | '.join(line.strip() for line in errors)}" if errors else ""
            lines.append(f"- The last check `{state.check}` failed{first}")
    for case_id, paths in state.duplicates.items():
        lines.append(
            f"- {case_id} is in {len(paths)} of your files ({_join(paths)}). A case gets one test: keep the best "
            "file and delete your other copies of it with `rm` (only files you created in this run)."
        )
    return "\n".join(lines)


def _id_pattern(case_id: str) -> re.Pattern[str] | None:
    """The case id in a file name or text, whatever the separators: TD-001 also matches td_001, TD001 and td-001-login,
    never TD-0010. An id without letters keeps a separator between its numbers (1.2 is not 12) and may follow a
    letter (case7_login), because class names cannot start with a digit."""
    parts = re.findall(r"[A-Za-z]+|\d+", case_id)
    if not parts:
        return None
    if _has_letters(case_id):
        body = r"[\s_.\-]*".join(re.escape(part) for part in parts)
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![0-9])", re.IGNORECASE)
    body = r"[\s_.\-]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![0-9]){body}(?![0-9])")


def _has_letters(case_id: str) -> bool:
    return any(character.isalpha() for character in case_id)


def _read(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _join(paths: Sequence[str]) -> str:
    if len(paths) <= 1:
        return "".join(paths)
    return ", ".join(paths[:-1]) + " and " + paths[-1]
