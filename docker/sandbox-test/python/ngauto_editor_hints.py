"""Clearer `str_replace` errors for the agent server's file editor.

The agent server imports this module when it starts (`--import-modules ngauto_editor_hints`, added by
ngauto-entrypoint.sh; this folder is on OH_EXTRA_PYTHON_PATH). It wraps FileEditor so that a failed str_replace
tells the model how to fix its call, worked out from the file itself:

- old_str matches several places: for each place (the first few), a ready-to-use old_str that matches only there,
  made of the place and its nearest neighbouring lines;
- old_str is not in the file: line numbers copied from `view`, other indentation or spacing (with the exact text of
  the file), or else the most similar text in the file;
- old_str and new_str are the same, old_str is empty, or a value is missing: what each value must hold.

Why: when the error only said "Please ensure it is unique" or "did not appear verbatim", a small model sent the same
failing edit again and again (12 times in a row in one real heal). The edit still fails exactly as before; only the
error text grows. Any problem while working out a hint leaves the editor's own message unchanged.

The agent server is a frozen (PyInstaller) program: besides its own modules, only the standard-library modules it
already bundles can be imported here (re, difflib, functools, logging, pathlib).
"""

from __future__ import annotations

import difflib
import functools
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Places shown with a ready-to-use old_str; further places are only listed by line number (at most MAX_LISTED_LINES).
MAX_PLACES = 3
MAX_LISTED_LINES = 20
# Neighbouring lines tried, at most, to make old_str match only one place.
MAX_CONTEXT_LINES = 12
# Longer text is pointed to by its line numbers instead of being printed.
MAX_BLOCK_LINES = 20
MAX_BLOCK_CHARS = 2000
# The most similar text is shown from this difflib ratio on, with at most this many differing lines.
SIMILAR_ENOUGH = 0.6
MAX_DIFF_LINES = 5
MAX_LINE_CHARS = 160
# Bigger files or longer old_str values are not searched for similar text: it would take too long.
MAX_SIMILARITY_FILE_LINES = 5000
MAX_SIMILARITY_OLD_LINES = 80
# When no line of the file equals a line of old_str, every place is compared, in files up to this size.
MAX_FULL_SCAN_LINES = 2000

# What `view` prints before each line: the line number right-aligned in 6 columns, then a tab ("    29\t"). Models
# also copy it with the tab dropped or turned into spaces ("    30        <dependency>").
_VIEW_NUMBER = re.compile(r"^ *(\d+)(\t| +(?=\S))")
_VIEW_NUMBER_PREFIX = re.compile(r"^ *\d+\t?")
# The editor's own message for an old_str that matches no place or several places.
_NO_REPLACEMENT = "No replacement was performed"

_EMPTY_OLD = (
    "No replacement was performed: old_str is empty (or only spaces), so it matches everywhere in the file. old_str "
    "must be the exact text you want to replace, copied from the file. To add new lines without replacing anything, "
    "use the `insert` command with `insert_line` (the line number after which the new text goes) and `new_str`."
)
_MISSING = {
    "old_str": (
        " old_str is the exact text to replace, copied from the file, and new_str is the text to put in its place. To "
        "add new lines without replacing anything, use the `insert` command with `insert_line` and `new_str`."
    ),
    "new_str": " new_str is the text to put in place of old_str. To delete old_str, give new_str an empty value.",
}
_SAME = (
    "No replacement was performed: old_str and new_str are exactly the same, so this edit would change nothing. "
    "new_str must be the text as it should look after your change: to change lines, write them changed; to delete "
    "lines, leave them out; to move lines, use an old_str that covers both their old and their new place, and write "
    "them at the new place in new_str. If you only meant to change spaces or line breaks at the very start or end of "
    "the text, add the neighbouring line to both old_str and new_str, so that the change is inside them."
)
_VIEW_NUMBERS_NOTE = (
    "Your old_str starts its lines with line numbers ({numbers}): they come from the output of `view` and are not "
    "part of the file. Leave them out."
)
_READY_TO_USE = (
    "Copy it exactly (same spaces, no line numbers) and write new_str as the same text with your change made in it: "
    "keep the lines you do not change."
)
_NOTHING_SIMILAR = (
    "Nothing in the file looks like your old_str. View the file (or the lines you mean) and copy the text exactly; if "
    "you changed this file earlier, it may no longer look the way you remember."
)


def patch_file_editor(editor_class: type | None = None) -> bool:
    """Wrap FileEditor.__call__ (once) so that a failed str_replace explains how to fix the call."""
    from openhands.tools.file_editor.editor import FileEditor
    from openhands.tools.file_editor.exceptions import ToolError

    cls = editor_class or FileEditor
    call = cls.__call__
    if getattr(call, "_ngauto_hints", False):
        return True

    @functools.wraps(call)
    def call_with_hints(self, *args, **kwargs):
        try:
            return call(self, *args, **kwargs)
        except ToolError as error:
            try:
                message = explain(self, kwargs, error)
            except Exception:  # a hint must never hide the editor's own error
                logger.debug("No str_replace hint", exc_info=True)
                message = None
            if message:
                error.message = message
                error.args = (message,)
            raise

    call_with_hints._ngauto_hints = True  # type: ignore[attr-defined]
    cls.__call__ = call_with_hints
    return True


def explain(editor, call: dict, error: Exception) -> str | None:
    """The message of a failed str_replace with a hint added, or None to keep the editor's message."""
    from openhands.tools.file_editor.exceptions import (
        EditorToolParameterInvalidError,
        EditorToolParameterMissingError,
        ToolError,
    )

    if call.get("command") != "str_replace":
        return None
    if isinstance(error, EditorToolParameterMissingError):
        hint = _MISSING.get(error.parameter)
        return error.message + hint if hint else None
    old, new = call.get("old_str"), call.get("new_str")
    if old is None:
        return None
    same = isinstance(error, EditorToolParameterInvalidError) and error.parameter == "new_str" and new == old
    no_match = type(error) is ToolError and error.message.startswith(_NO_REPLACEMENT)
    if not (same or no_match):
        return None
    if not old.strip():
        return _EMPTY_OLD
    path = Path(call["path"])
    content = editor.read_file(path)
    if same:
        return _SAME + _where_old_str_is(content, old, path)
    # The editor matches old_str as given and, when that matches nothing, without its outer spaces.
    needle = old if old in content else old.strip()
    count = content.count(needle)
    if count > 1:
        return f"{error.message}\n\n{_several_places(content, needle, path, count)}"
    if count == 0:
        return f"{error.message}\n\n{_not_found(content, old)}"
    return None


def _where_old_str_is(content: str, old: str, path: Path) -> str:
    needle = old if old in content else old.strip()
    count = content.count(needle)
    if count == 1:
        return ""
    if count > 1:
        return (
            f"\n\nAlso: this old_str matches {count} places in {path}; it must match exactly one. Add neighbouring "
            "lines until it matches only the place you mean."
        )
    return f"\n\nAlso: this old_str is not in {path}: view the file and copy the text exactly."


def _several_places(content: str, needle: str, path: Path, count: int) -> str:
    spans = [_span(content, start, needle) for start in _starts(content, needle)]
    return (
        f"Why: old_str must match exactly one place in {path}, and it matches {count}. Fix: add neighbouring lines of "
        "the place you mean to old_str, until it matches only there. A ready-to-use old_str for each place follows. "
        f"{_READY_TO_USE}\n\n{_places(content, spans)}"
    )


def _not_found(content: str, old: str) -> str:
    notes: list[str] = []
    text = old
    lines = content.split("\n")
    numbers = _view_numbers(old)
    if numbers:
        cleaned = "\n".join(_drop_view_number(line) for line in old.split("\n"))
        needle = cleaned if cleaned in content else cleaned.strip()
        count = content.count(needle) if needle.strip() else 0
        # One numbered line could be real text (a table row): it counts only when the text without it is in the file.
        if not (count or len(numbers) > 1 or _same_ignoring_spaces(content, lines, cleaned)):
            numbers = []
    if numbers:
        text = cleaned
        shown = ", ".join(str(number) for number in numbers[:3]) + (", ..." if len(numbers) > 3 else "")
        notes.append(_VIEW_NUMBERS_NOTE.format(numbers=shown))
        if count == 1:
            first, last = _span(content, content.index(needle), needle)
            notes.append(f"Without them, old_str matches {_lines(first, last)}, and only there:\n{_block(needle)}")
            return "\n\n".join(notes)
        if count > 1:
            spans = [_span(content, start, needle) for start in _starts(content, needle)]
            notes.append(
                f"Without them, old_str matches {count} places; it must match exactly one. A ready-to-use old_str "
                f"for each place follows. {_READY_TO_USE}\n\n{_places(content, spans)}"
            )
            return "\n\n".join(notes)
    notes.append(
        _same_ignoring_spaces(content, lines, text)
        or _partly_there(content, lines, text)
        or _most_similar(content, lines, text)
        or _NOTHING_SIMILAR
    )
    return "\n\n".join(notes)


def _same_ignoring_spaces(content: str, lines: list[str], text: str) -> str | None:
    """old_str is in the file if spaces are ignored: say what differs, and give the exact text."""
    old = _trimmed(text.split("\n"))
    if not old:
        return None
    starts = _loose_starts(lines, old)
    if not starts:
        return None
    size = len(old)
    first = starts[0]
    span = _lines(first + 1, first + size)
    differences = _differences(lines, first, old)
    head = f"Your old_str matches {span} of the file if spaces are ignored, but not character for character"
    head += f": {'; '.join(differences)}." if differences else "."
    if len(starts) > 1:
        spans = [(start + 1, start + size) for start in starts]
        return (
            f"{head}\nIgnoring spaces, it matches {len(starts)} places. A ready-to-use old_str for each place "
            f"follows. {_READY_TO_USE}\n\n{_places(content, spans)}"
        )
    window = "\n".join(lines[first:first + size])
    if _unique(content, window) and size <= MAX_BLOCK_LINES and len(window) <= MAX_BLOCK_CHARS:
        return f"{head}\nThe exact text of {span}, ready to use as old_str (it matches only there):\n{_block(window)}\n{_READY_TO_USE}"
    return f"{head}\n\n{_places(content, [(first + 1, first + size)])}"


def _most_similar(content: str, lines: list[str], text: str) -> str | None:
    """The place in the file most like old_str (same number of lines), when it is similar enough."""
    old = _trimmed(text.split("\n"))
    size = len(old)
    if not size or size > MAX_SIMILARITY_OLD_LINES or len(lines) > MAX_SIMILARITY_FILE_LINES or size > len(lines):
        return None
    norm_old = [_norm(line) for line in old]
    norm_file = [_norm(line) for line in lines]
    last_start = len(lines) - size
    positions: dict[str, list[int]] = {}
    for position, line in enumerate(norm_file):
        if line:
            positions.setdefault(line, []).append(position)
    starts = {
        position - index
        for index, line in enumerate(norm_old)
        for position in positions.get(line, ())
        if 0 <= position - index <= last_start
    }
    if not starts and len(lines) <= MAX_FULL_SCAN_LINES:
        starts = set(range(last_start + 1))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2("\n".join(norm_old))
    best_ratio, best = 0.0, None
    for start in sorted(starts):
        matcher.set_seq1("\n".join(norm_file[start:start + size]))
        if matcher.real_quick_ratio() < SIMILAR_ENOUGH or matcher.quick_ratio() < SIMILAR_ENOUGH:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, start
    if best is None or best_ratio < SIMILAR_ENOUGH:
        return None
    span = _lines(best + 1, best + size)
    differing = [index for index in range(size) if norm_file[best + index] != norm_old[index]]
    parts = [
        f"Nothing in the file matches your old_str, even with other spaces. The most similar text is {span} "
        f"({best_ratio:.0%} alike); it differs here:"
    ]
    for index in differing[:MAX_DIFF_LINES]:
        parts.append(
            f"- line {best + index + 1} of the file: `{_short(lines[best + index])}`\n"
            f"  in your old_str: `{_short(old[index])}`"
        )
    if len(differing) > MAX_DIFF_LINES:
        parts.append(f"- and {len(differing) - MAX_DIFF_LINES} more lines")
    window = "\n".join(lines[best:best + size])
    if _unique(content, window) and size <= MAX_BLOCK_LINES and len(window) <= MAX_BLOCK_CHARS:
        parts.append(f"If that is the place you mean, its exact current text is below. {_READY_TO_USE}\n{_block(window)}")
    else:
        parts.append(f"If that is the place you mean:\n\n{_places(content, [(best + 1, best + size)])}")
    parts.append("If you changed this file earlier, view it again first: it may no longer look the way you remember.")
    return "\n".join(parts)


def _places(content: str, spans: list[tuple[int, int]]) -> str:
    """For each place (first and last line, 1-based; the first few): the shortest old_str that matches only there."""
    lines = content.split("\n")
    parts = []
    for number, (first, last) in enumerate(spans[:MAX_PLACES], start=1):
        where = f"Place {number} (starts at line {first})"
        window = _unique_window(content, lines, first, last)
        if window is None:
            parts.append(
                f"{where}: no text of up to {MAX_CONTEXT_LINES} lines around it matches only there; view the lines "
                "around it and add more of them to old_str."
            )
            continue
        start, end, text = window
        if end - start + 1 > MAX_BLOCK_LINES or len(text) > MAX_BLOCK_CHARS:
            parts.append(f"{where}: {_lines(start, end)} together match only there; view them and copy them exactly.")
        else:
            parts.append(f"{where}: this old_str ({_lines(start, end)}) matches only there:\n{_block(text)}")
    rest = spans[MAX_PLACES:]
    if rest:
        shown = ", ".join(str(first) for first, _ in rest[:MAX_LISTED_LINES])
        more = f" and {len(rest) - MAX_LISTED_LINES} more" if len(rest) > MAX_LISTED_LINES else ""
        if len(rest) == 1:
            parts.append(f"1 more place starts at line {shown}.")
        else:
            parts.append(f"{len(rest)} more places start at lines {shown}{more}.")
    return "\n\n".join(parts)


def _unique_window(content: str, lines: list[str], first: int, last: int) -> tuple[int, int, str] | None:
    """The fewest whole lines around lines first..last (1-based) whose text matches only one place in the file,
    also without its outer spaces (the text format of tool calls drops them). Lines above are tried first."""
    for extra in range(MAX_CONTEXT_LINES + 1):
        for before in range(extra, -1, -1):
            start, end = first - before, last + extra - before
            if start < 1 or end > len(lines):
                continue
            text = "\n".join(lines[start - 1:end])
            if _unique(content, text):
                return start, end, text
    return None


def _unique(content: str, text: str) -> bool:
    return bool(text.strip()) and content.count(text) == 1 and content.count(text.strip()) == 1


def _loose_starts(lines: list[str], old: list[str]) -> list[int]:
    """Where (0-based) the lines of old_str are in the file when spaces are ignored. Its first line may be the end
    of a file line and its last line the start of one."""
    norm_old = [_norm(line) for line in old]
    norm_file = [_norm(line) for line in lines]
    size = len(old)
    starts = []
    for start in range(len(lines) - size + 1):
        if size == 1:
            found = norm_old[0] in norm_file[start]
        else:
            found = (
                norm_file[start].endswith(norm_old[0])
                and norm_file[start + size - 1].startswith(norm_old[-1])
                and norm_file[start + 1:start + size - 1] == norm_old[1:-1]
            )
        if found:
            starts.append(start)
    return starts


def _differences(lines: list[str], start: int, old: list[str]) -> list[str]:
    """How the file's lines from `start` (0-based) differ from old_str's lines in their spaces, in words."""
    size = len(old)
    indents: dict[int, list[int]] = {}
    examples: dict[int, tuple[int, int, int]] = {}
    tabs_in_file, tabs_in_old, trailing, inner = [], [], [], []
    for index, mine in enumerate(old):
        number = start + index + 1
        theirs = lines[start + index]
        if index > 0:  # the first line of old_str may begin in the middle of a file line
            file_indent, old_indent = _indent(theirs), _indent(mine)
            if "\t" in file_indent and "\t" not in old_indent:
                tabs_in_file.append(number)
            elif "\t" in old_indent and "\t" not in file_indent:
                tabs_in_old.append(number)
            elif len(file_indent) != len(old_indent):
                delta = len(file_indent) - len(old_indent)
                indents.setdefault(delta, []).append(number)
                examples.setdefault(delta, (number, len(file_indent), len(old_indent)))
        if size == 1:
            exact = mine.strip() in theirs
        elif index == 0:
            exact = theirs.rstrip().endswith(mine.strip())
        elif index == size - 1:
            exact = theirs.lstrip().startswith(mine.strip())
        else:
            exact = theirs.strip() == mine.strip()
        if not exact:
            inner.append(number)
        if index < size - 1 and theirs != theirs.rstrip() and mine == mine.rstrip():
            trailing.append(number)
    notes = []
    for delta, numbers in sorted(indents.items()):
        number, in_file, in_old = examples[delta]
        if len(numbers) == 1:
            notes.append(f"line {number} starts with {in_file} spaces in the file but {in_old} in your old_str")
        else:
            more_or_fewer = "more" if delta > 0 else "fewer"
            notes.append(
                f"{_numbers(numbers)} start with {abs(delta)} {more_or_fewer} spaces in the file than in your old_str "
                f"(for example line {number}: {in_file} in the file, {in_old} in yours)"
            )
    if tabs_in_file:
        notes.append(f"{_numbers(tabs_in_file)} {_is(tabs_in_file)} indented with tabs in the file but spaces in yours")
    if tabs_in_old:
        notes.append(f"{_numbers(tabs_in_old)} {_is(tabs_in_old)} indented with spaces in the file but tabs in yours")
    if trailing:
        ends = "ends" if len(trailing) == 1 else "end"
        notes.append(f"{_numbers(trailing)} {ends} with spaces in the file that your old_str does not have")
    if inner:
        notes.append(f"the spaces inside {_numbers(inner)} differ")
    return notes


def _view_numbers(old: str) -> list[int]:
    """The line numbers of `view` at the start of old_str's lines, or [] when old_str does not look like that: at
    least half of its lines numbered, one line only in view's own format (number, tab), several lines counting up by
    one per line."""
    lines = old.split("\n")
    found = []
    for index, line in enumerate(lines):
        match = _VIEW_NUMBER.match(line)
        if match:
            found.append((index, int(match.group(1)), match.group(2) == "\t"))
    if not found or 2 * len(found) < sum(1 for line in lines if line.strip()):
        return []
    if len(found) == 1:
        return [found[0][1]] if found[0][2] else []
    if all(later[1] - earlier[1] == later[0] - earlier[0] for earlier, later in zip(found, found[1:])):
        return [number for _, number, _ in found]
    return []


def _drop_view_number(line: str) -> str:
    return _VIEW_NUMBER_PREFIX.sub("", line, count=1) if _VIEW_NUMBER.match(line) else line


def _partly_there(content: str, lines: list[str], text: str) -> str | None:
    """Only the start (or only the end) of old_str is in the file, spaces ignored: say where the two part."""
    old = _trimmed(text.split("\n"))
    if len(old) < 3:
        return None
    norm_old = [_norm(line) for line in old]
    norm_file = [_norm(line) for line in lines]
    found: dict[tuple[bool, int], int] = {}  # (from the start?, first matching file line, 0-based) -> lines matched
    for position in range(len(lines)):
        # The start of old_str from this line on (its first line may be the end of a file line).
        count = 0
        while count < len(old) - 1 and position + count < len(lines) and (
            norm_file[position].endswith(norm_old[0]) if count == 0 else norm_file[position + count] == norm_old[count]
        ):
            count += 1
        if count >= 2:
            found[(True, position)] = count
        # The end of old_str up to this line (its last line may be the start of a file line).
        count = 0
        while count < len(old) - 1 and position - count >= 0 and (
            norm_file[position].startswith(norm_old[-1]) if count == 0
            else norm_file[position - count] == norm_old[len(old) - 1 - count]
        ):
            count += 1
        if count >= 2:
            found[(False, position - count + 1)] = count
    if not found:
        return None
    longest = max(found.values())
    best = [key for key, count in found.items() if count == longest]
    from_start, start = best[0]
    matched = old[:longest] if from_start else old[len(old) - longest:]
    # One place only, and more than a couple of brackets: otherwise this says nothing reliable.
    if len(best) > 1 or sum(len(_norm(line)) for line in matched) < 12:
        return None
    span = _lines(start + 1, start + longest)
    if from_start:
        after = start + longest
        theirs = f"line {after + 1} of the file is {_quoted(lines[after])}" if after < len(lines) else "the file ends"
        where = (
            f"Only the first {longest} lines of your old_str are in the file ({span}, spaces ignored); then {theirs} "
            f"where your old_str has {_quoted(old[longest])}."
        )
    else:
        before = start - 1
        theirs = f"line {before + 1} of the file is {_quoted(lines[before])}" if before >= 0 else "the file starts"
        where = (
            f"Only the last {longest} lines of your old_str are in the file ({span}, spaces ignored); before them "
            f"{theirs} where your old_str has {_quoted(old[len(old) - longest - 1])}."
        )
    where += (
        "\nThe file may have changed since you last viewed it (view it again), or your old_str has lines the file "
        "does not have."
    )
    window = "\n".join(lines[start:start + longest])
    if _unique(content, window) and longest <= MAX_BLOCK_LINES and len(window) <= MAX_BLOCK_CHARS:
        return f"{where}\nThe exact text of {span} (it matches only there):\n{_block(window)}"
    return where


def _quoted(line: str) -> str:
    return "an empty line" if not line.strip() else f"`{_short(line)}`"


def _starts(content: str, needle: str) -> list[int]:
    return [match.start() for match in re.finditer(re.escape(needle), content)]


def _span(content: str, start: int, needle: str) -> tuple[int, int]:
    """First and last line (1-based) of the match of `needle` at `start`."""
    return _line_of(content, start), _line_of(content, start + max(len(needle) - 1, 0))


def _line_of(content: str, index: int) -> int:
    return content.count("\n", 0, index) + 1


def _lines(first: int, last: int) -> str:
    return f"line {first}" if first == last else f"lines {first}-{last}"


def _numbers(numbers: list[int]) -> str:
    """e.g. "line 7", "lines 30-35" or "lines 3, 9-10 and 14"."""
    ranges: list[list[int]] = []
    for number in sorted(set(numbers)):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    texts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    joined = texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]
    return f"{'line' if len(set(numbers)) == 1 else 'lines'} {joined}"


def _is(numbers: list[int]) -> str:
    return "is" if len(set(numbers)) == 1 else "are"


def _block(text: str) -> str:
    fence = "~~~~" if "```" in text else "```"
    return f"{fence}\n{text}\n{fence}"


def _trimmed(lines: list[str]) -> list[str]:
    """The lines without blank lines at the start and the end."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _norm(line: str) -> str:
    return " ".join(line.split())


def _short(line: str) -> str:
    line = line.strip()
    return line if len(line) <= MAX_LINE_CHARS else line[: MAX_LINE_CHARS - 1] + "…"


# Imported by the agent server at start (a test loads this file under another name and patches by hand).
if __name__ == "ngauto_editor_hints":
    try:
        patch_file_editor()
        logger.info("ngauto_editor_hints: failed str_replace calls now explain how to fix them")
    except Exception:  # never keep the agent server from starting
        logger.warning("ngauto_editor_hints: the file editor was not patched", exc_info=True)
