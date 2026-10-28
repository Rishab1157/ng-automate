"""Shared utilities."""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# More patterns than this from one {a,b} pattern is a mistake, not a search.
_MAX_BRACE_PATTERNS = 1000


def _check_command_available(
    command: str,
    probe_args: Sequence[str] | None = ("--version",),
) -> bool:
    """Check if a command is available and optionally responds to a probe."""

    try:
        if shutil.which(command) is None:
            return False
        if probe_args is None:
            return True
        result = subprocess.run(
            [command, *probe_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available on the system."""

    return _check_command_available("rg")


def _check_grep_available() -> bool:
    """Check if grep is available on the system."""

    return _check_command_available("grep", probe_args=None)


def _log_ripgrep_fallback_warning(tool_name: str, fallback_method: str) -> None:
    """Log a warning about falling back from ripgrep to alternative method.

    Args:
        tool_name: Name of the tool (e.g., "glob", "grep")
        fallback_method: Description of the fallback method being used
    """
    logger.warning(
        f"{tool_name}: ripgrep (rg) not available. "
        f"Falling back to {fallback_method}. "
        f"For better performance, consider installing ripgrep: "
        f"https://github.com/BurntSushi/ripgrep#installation"
    )


def _resolve_search_path(path: str, working_dir: Path) -> Path:
    """Resolve a tool's ``path`` argument; a relative path starts at working_dir.

    ``Path(path).resolve()`` alone would start at the server process' own
    folder, which is not the agent's working directory.
    """
    search_path = Path(path)
    if not search_path.is_absolute():
        search_path = working_dir / search_path
    return search_path.resolve()


def _expand_braces(pattern: str, limit: int = _MAX_BRACE_PATTERNS) -> list[str]:
    """Expand the ``{a,b}`` groups of a glob pattern into separate patterns.

    ripgrep globs support braces, but fnmatch and Python's glob module do not,
    so ``*.{ts,tsx}`` would only match a file named exactly that. Groups may be
    nested; a brace that is never closed, or one inside ``[...]``, is a plain
    character.

    Raises:
        ValueError: when the pattern expands to more than ``limit`` patterns
            (e.g. ``{a,b}`` written 30 times) or nests groups too deeply.
    """
    try:
        expanded = _expand_brace_groups(pattern, limit)
    except RecursionError:
        raise ValueError(f"'{pattern}' has {{a,b}} groups nested too deeply") from None
    if expanded is None:
        raise ValueError(
            f"'{pattern}' has too many {{a,b}} groups: it expands to more than "
            f"{limit} patterns"
        )
    return expanded


def _expand_brace_groups(pattern: str, limit: int) -> list[str] | None:
    """Expand the brace groups of ``pattern``; None if that gives over ``limit``."""
    group = _find_brace_group(pattern)
    if group is None:
        return [pattern]
    start, end, alternatives = group
    expanded: list[str] = []
    for alternative in alternatives:
        tails = _expand_brace_groups(alternative + pattern[end + 1 :], limit)
        if tails is None or len(expanded) + len(tails) > limit:
            return None
        expanded.extend(pattern[:start] + tail for tail in tails)
    return expanded


def _find_brace_group(pattern: str) -> tuple[int, int, list[str]] | None:
    """Return (start, end, alternatives) of the first closed ``{...}`` group."""
    index = 0
    while index < len(pattern):
        if pattern[index] == "[":
            index = _skip_bracket(pattern, index)
            continue
        if pattern[index] == "{":
            group = _read_brace_group(pattern, index)
            if group is not None:
                return group
        index += 1
    return None


def _read_brace_group(pattern: str, start: int) -> tuple[int, int, list[str]] | None:
    """Read the group opened at ``start``; None if it is never closed."""
    depth = 1
    alternatives: list[str] = []
    alternative_start = start + 1
    index = start + 1
    while index < len(pattern):
        char = pattern[index]
        if char == "[":
            index = _skip_bracket(pattern, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                alternatives.append(pattern[alternative_start:index])
                return start, index, alternatives
        elif char == "," and depth == 1:
            alternatives.append(pattern[alternative_start:index])
            alternative_start = index + 1
        index += 1
    return None


def _skip_bracket(pattern: str, start: int) -> int:
    """Return the index after the ``[...]`` set opened at ``start``.

    Follows fnmatch: a ``]`` right after ``[`` or ``[!`` is part of the set, and
    a ``[`` that is never closed is a plain character.
    """
    index = start + 1
    if index < len(pattern) and pattern[index] == "!":
        index += 1
    if index < len(pattern) and pattern[index] == "]":
        index += 1
    end = pattern.find("]", index)
    return start + 1 if end < 0 else end + 1
