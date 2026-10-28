"""Grep tool executor implementation."""

import fnmatch
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import sanitized_env


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
from openhands.tools.grep.definition import GrepAction, GrepObservation
from openhands.tools.utils import (
    _check_grep_available,
    _check_ripgrep_available,
    _expand_braces,
    _log_ripgrep_fallback_warning,
    _resolve_search_path,
)


logger = get_logger(__name__)


@lru_cache(maxsize=64)
def _include_globs(include: str, search_root: str) -> tuple[tuple[str, ...], ...]:
    """Turn an include pattern into globs split at "/", one per {a,b} choice.

    The rules are ripgrep's -g rules: a pattern without "/" matches the file
    name in any folder ("*.java"); a pattern with "/" matches the path relative
    to the search folder ("src/**/*.java"), where "**" stands for any number of
    folders and "*" stays inside one folder. An absolute pattern inside the
    search folder is read relative to it.

    Raises:
        ValueError: for a pattern with too many {a,b} groups.
    """
    globs: list[tuple[str, ...]] = []
    for pattern in _expand_braces(include):
        anchored = False
        if pattern.startswith(f"{search_root.rstrip('/')}/"):
            pattern, anchored = pattern[len(search_root.rstrip("/")) + 1 :], True
        while pattern.startswith("./"):
            pattern, anchored = pattern[2:], True
        if pattern.startswith("/"):
            pattern, anchored = pattern.lstrip("/"), True
        if anchored or "/" in pattern:
            globs.append(tuple(pattern.split("/")))
        else:
            globs.append(("**", pattern))
    return tuple(globs)


def _glob_matches(glob: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """fnmatch for paths split at "/", where a "**" part matches any folders.

    A trailing "**" needs at least one name after the folder ("src/**" matches
    what is inside src). Dynamic programming keeps many "**" parts cheap.
    """
    # reachable[j]: the glob parts so far can match the first j parts of path.
    reachable = [True] + [False] * len(path)
    for index, part in enumerate(glob):
        if part == "**":
            trailing = index == len(glob) - 1
            seen = False
            for j, was_reachable in enumerate(reachable):
                if trailing:
                    reachable[j], seen = seen, seen or was_reachable
                else:
                    seen = seen or was_reachable
                    reachable[j] = seen
        else:
            reachable = [False] + [
                reachable[j] and fnmatch.fnmatch(name, part)
                for j, name in enumerate(path)
            ]
    return reachable[-1]


def _has_wildcard(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _split_top_level(pattern: str, separators: str) -> list[str]:
    """Split at the separators that are not inside {...}; drop empty parts."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(pattern):
        if char == "{":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
        elif char in separators and not depth:
            parts.append(pattern[start:index])
            start = index + 1
    parts.append(pattern[start:])
    return [part.strip() for part in parts if part.strip()]


def _pattern_list(include: str) -> list[str]:
    """The patterns of an include written as a list ("*.js,*.ts"), else []."""
    by_comma = _split_top_level(include, ",")
    if len(by_comma) > 1:
        return by_comma
    by_space = _split_top_level(include, " \t\r\n")
    if len(by_space) > 1 and all(_has_wildcard(part) for part in by_space):
        return by_space
    return []


def _needs_python_search(pattern: str, regex: re.Pattern[str]) -> bool:
    """Whether a line-by-line grep or ripgrep would miss what the regex finds.

    Both read one line at a time, so a pattern that spans lines (a newline in
    it, or (?s) so that "." crosses lines) needs the Python search, and so does
    a NUL character, which cannot be passed to a program.
    """
    if regex.flags & re.DOTALL:
        return True
    return any(text in pattern for text in ("\n", "\x00", "\\n"))


class GrepExecutor(ToolExecutor[GrepAction, GrepObservation]):
    """Executor for grep content search operations.

    This implementation prefers ripgrep for performance, falls back to the
    system grep binary when available, and finally uses a Python recursive
    search when no grep binary is installed.

    Every backend reads the pattern as a Python-style regex (grep with -P, and
    the Python search when a program cannot take the pattern), and filters the
    found files with the same include rules (see _include_globs), so the
    backends agree.
    """

    _MAX_MATCHES = 100

    def __init__(self, working_dir: str):
        """Initialize the grep executor.

        Args:
            working_dir: The working directory to use as the base for searches
        """
        self.working_dir: Path = Path(working_dir).resolve()
        self._search_backend = self._select_search_backend()

        if self._search_backend == "grep":
            _log_ripgrep_fallback_warning("grep", "system grep")
        elif self._search_backend == "python":
            _log_ripgrep_fallback_warning("grep", "system grep, then Python search")

    def _select_search_backend(self) -> str:
        if _check_ripgrep_available():
            return "ripgrep"
        if _check_grep_available():
            return "grep"
        return "python"

    def __call__(
        self,
        action: GrepAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> GrepObservation:
        """Execute grep content search using the best available backend."""
        try:
            if action.path:
                search_path = _resolve_search_path(action.path, self.working_dir)
                if not search_path.is_dir():
                    return GrepObservation.from_text(
                        text=self._invalid_path_message(action.path, search_path),
                        matches=[],
                        pattern=action.pattern,
                        search_path=str(search_path),
                        include_pattern=action.include,
                        is_error=True,
                    )
            else:
                search_path = self.working_dir

            try:
                regex = re.compile(action.pattern, re.IGNORECASE | re.MULTILINE)
            except re.error as e:
                return GrepObservation.from_text(
                    text=f"Invalid regex pattern: {e}",
                    matches=[],
                    pattern=action.pattern,
                    search_path=str(search_path),
                    include_pattern=action.include,
                    is_error=True,
                )

            if action.include:
                try:
                    _include_globs(action.include, search_path.as_posix())
                except ValueError as e:
                    return GrepObservation.from_text(
                        text=f"Invalid include pattern: {e}",
                        matches=[],
                        pattern=action.pattern,
                        search_path=str(search_path),
                        include_pattern=action.include,
                        is_error=True,
                    )

            if self._search_backend == "python" or _needs_python_search(
                action.pattern, regex
            ):
                return self._execute_with_python_search(action, search_path, regex)
            if self._search_backend == "ripgrep":
                return self._execute_with_ripgrep(action, search_path)
            return self._execute_with_system_grep(action, search_path)

        except Exception as e:
            try:
                if action.path:
                    error_search_path = str(
                        _resolve_search_path(action.path, self.working_dir)
                    )
                else:
                    error_search_path = str(self.working_dir)
            except Exception:
                error_search_path = "unknown"

            return GrepObservation.from_text(
                text=str(e),
                matches=[],
                pattern=action.pattern,
                search_path=error_search_path,
                include_pattern=action.include,
                is_error=True,
            )

    def _format_output(
        self,
        matches: list[str],
        pattern: str,
        search_path: str,
        include_pattern: str | None,
        truncated: bool,
    ) -> str:
        """Format the grep observation output message."""
        if not matches:
            include_info = (
                f" (filtered by '{include_pattern}')" if include_pattern else ""
            )
            return (
                f"No files found containing pattern '{pattern}' "
                f"in directory '{search_path}'{include_info}"
            )

        include_info = f" (filtered by '{include_pattern}')" if include_pattern else ""
        file_list = "\n".join(matches)
        output = (
            f"Found {len(matches)} file(s) containing pattern "
            f"'{pattern}' in '{search_path}'{include_info}:\n{file_list}"
        )
        if truncated:
            output += (
                "\n\n[Results truncated to first 100 files. "
                "Consider using a more specific pattern.]"
            )
        return output

    def _invalid_path_message(self, path: str, search_path: Path) -> str:
        """Say why ``path`` cannot be searched, and what to do instead."""
        if search_path.is_file():
            return (
                f"Search path '{path}' is a file, not a valid directory. To search "
                f"only this file, set path='{search_path.parent}' and "
                f"include='{search_path.name}'."
            )
        message = f"Search path '{path}' is not a valid directory ('{search_path}'"
        message += " is not a folder" if search_path.exists() else " does not exist"
        if not Path(path).is_absolute():
            message += (
                "; a relative path starts at the working directory "
                f"'{self.working_dir}'"
            )
        return message + ")."

    def _include_hint(self, include: str, search_path: Path) -> str:
        """Explain an include that matched nothing: models often pass a folder
        or a list, or a path that is relative to another folder."""
        patterns = _pattern_list(include)
        if patterns:
            return (
                "Note: include is one glob pattern, not a list. To match any of "
                f"several patterns, use braces: include='{{{','.join(patterns)}}}'."
            )
        for pattern in _expand_braces(include):
            folder = pattern.rstrip("/")
            while folder.startswith("./"):
                folder = folder[2:]
            if not folder or _has_wildcard(folder):
                continue
            candidate = (search_path / folder).resolve()
            if candidate.is_dir():
                try:
                    shown = candidate.relative_to(self.working_dir).as_posix()
                except ValueError:
                    shown = str(candidate)
                return (
                    f"Note: '{folder}' is a folder, but include only filters file "
                    "names or paths. To search inside the folder, set "
                    f"path='{shown}' and leave include empty (or set it to a file "
                    "pattern such as '*.js')."
                )
        if "/" not in include:
            return ""
        hint = (
            "Note: an include with '/' is matched against the file path relative "
            f"to the search folder '{search_path}', where '**' stands for any "
            "number of folders (e.g. 'src/**/*.js'); one without '/' (e.g. '*.js') "
            "matches file names in every folder."
        )
        folders = include.split("/")[:-1]
        if any(part.startswith(".") and part not in (".", "..") for part in folders):
            hint += (
                " Folders whose names start with '.' are skipped unless path is "
                "inside them."
            )
        return hint

    def _path_matches_filters(
        self,
        path: Path,
        search_path: Path,
        include_pattern: str | None,
    ) -> bool:
        """Return whether a matched path should be surfaced to the user."""
        relative_parts = self._relative_parts(path, search_path)
        if any(part.startswith(".") for part in relative_parts[:-1]):
            return False

        if include_pattern:
            globs = _include_globs(include_pattern, search_path.as_posix())
            return any(_glob_matches(glob, relative_parts) for glob in globs)
        return not relative_parts[-1].startswith(".")

    @staticmethod
    def _relative_parts(path: Path, search_path: Path) -> tuple[str, ...]:
        """The parts of ``path`` below ``search_path`` (its name if not below)."""
        for candidate, base in (
            (path, search_path),
            (path.resolve(), search_path.resolve()),
        ):
            try:
                relative_parts = candidate.relative_to(base).parts
            except ValueError:
                continue
            if relative_parts:
                return relative_parts
        return (path.name,)

    def _match_mtime(self, path: Path) -> float:
        """Return a sortable modification time for matched paths."""
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    def _finalize_matches(
        self,
        matches: list[Path],
        search_path: Path,
        include_pattern: str | None,
    ) -> tuple[list[str], bool]:
        """Filter, sort, and truncate raw match paths."""
        unique_matches: dict[str, Path] = {}
        for match in matches:
            try:
                resolved = match.resolve()
            except OSError:
                continue
            if not self._path_matches_filters(match, search_path, include_pattern):
                continue
            unique_matches[str(resolved)] = resolved

        sorted_matches = sorted(
            unique_matches.values(),
            key=self._match_mtime,
            reverse=True,
        )
        truncated = len(sorted_matches) > self._MAX_MATCHES
        return [str(path) for path in sorted_matches[: self._MAX_MATCHES]], truncated

    def _build_observation(
        self,
        action: GrepAction,
        search_path: Path,
        matches: list[Path],
    ) -> GrepObservation:
        formatted_matches, truncated = self._finalize_matches(
            matches,
            search_path,
            action.include,
        )
        output = self._format_output(
            matches=formatted_matches,
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            truncated=truncated,
        )
        if not formatted_matches and action.include:
            hint = self._include_hint(action.include, search_path)
            if hint:
                output += f"\n{hint}"
        return GrepObservation.from_text(
            text=output,
            matches=formatted_matches,
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            truncated=truncated,
        )

    def _run_search_program(
        self, cmd: list[str], action: GrepAction, search_path: Path
    ) -> GrepObservation:
        """Run rg or grep, which print the matching files separated by NUL.

        Exit code 0 means files were found and 1 that none were. Anything else
        (the program cannot take this regex, or some file could not be read) is
        not trusted to be complete, so the Python search answers instead.
        """
        env = sanitized_env()
        if os.name == "nt":
            # The grep of Git for Windows is an MSYS program: it expands
            # wildcards in its arguments, which also drops the backslashes of
            # a regex such as "\d+" or "\b?".
            for name in ("MSYS", "CYGWIN"):
                env[name] = f"{env.get(name, '')} noglob".strip()
        result = subprocess.run(
            cmd,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=30,
            check=False,
            env=env,
        )
        if result.returncode not in (0, 1):
            logger.warning(
                "%s backend failed with exit code %s; falling back to Python search",
                cmd[0],
                result.returncode,
            )
            return self._execute_with_python_search(action, search_path)

        matches = [Path(line) for line in result.stdout.split("\0") if line]
        return self._build_observation(action, search_path, matches)

    def _execute_with_ripgrep(
        self, action: GrepAction, search_path: Path
    ) -> GrepObservation:
        """Execute grep content search using ripgrep.

        The include filter is applied to ripgrep's results, as for the other
        backends: ripgrep reads -g globs relative to its own process folder, so
        "src/**/*.java" found nothing when that was not the search folder.
        --hidden with the "!.*/" glob skips hidden folders but not hidden files,
        which an include may ask for (e.g. ".env").
        """
        cmd = [
            "rg",
            "-l",
            "-i",
            "--null",
            "--hidden",
            "--glob=!.*/",
            "--sortr=modified",
            "-e",
            action.pattern,
            "--",
            str(search_path),
        ]
        return self._run_search_program(cmd, action, search_path)

    def _execute_with_system_grep(
        self, action: GrepAction, search_path: Path
    ) -> GrepObservation:
        """Execute grep content search using the system grep binary.

        -P (Perl-compatible) reads the regex the way Python does: "a|b", "\\d",
        "\\s", "(?:...)", look-arounds. A grep without -P, or a regex PCRE does
        not take, exits with 2 and the Python search answers instead.
        """
        cmd = [
            "grep",
            "-R",
            "-I",
            "-l",
            "-i",
            "-Z",
            "-P",
            "-e",
            action.pattern,
            "--",
            str(search_path),
        ]
        return self._run_search_program(cmd, action, search_path)

    def _execute_with_python_search(
        self,
        action: GrepAction,
        search_path: Path,
        regex: re.Pattern[str] | None = None,
    ) -> GrepObservation:
        """Execute grep content search using Python file walking."""
        compiled_regex = regex or re.compile(
            action.pattern, re.IGNORECASE | re.MULTILINE
        )
        matches: list[Path] = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for filename in files:
                file_path = Path(root) / filename
                if not self._path_matches_filters(
                    file_path, search_path, action.include
                ):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                # Like grep -I and ripgrep, skip binary files (they hold NUL bytes).
                if "\x00" not in content and compiled_regex.search(content):
                    matches.append(file_path)

        return self._build_observation(action, search_path, matches)
