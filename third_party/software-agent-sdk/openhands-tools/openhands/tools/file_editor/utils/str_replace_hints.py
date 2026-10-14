"""Clearer errors for a failed `str_replace` (an NG Automate change to the vendored file editor).

When str_replace fails, the editor's own message says what went wrong but not how to fix the call, and a small model
then sends the same failing call again and again (12 times in a row in one real heal). FileEditor.__call__ hands every
failed str_replace to `add_fix_hint`, which adds a short hint to the message, worked out from the file itself:

- old_str matches several places: for each place (the first few), a ready-to-use old_str that matches only there,
  made of the place and its nearest neighbouring lines (named, so that new_str keeps them);
- old_str is not in the file: line numbers copied from `view`, other indentation or spacing, other line breaks, one
  changed line or a few lines more or less (each with the exact text of the file), only its start or its end in the
  file, or else the most similar text in the file;
- old_str and new_str are the same (often because the text format of tool calls dropped the spaces at their start and
  end), old_str is empty, or a value is missing: what each value must hold;
- a value holds another parameter's tag (`<parameter=new_str>`): that value was not closed with `</parameter>`.

Only the message changes: the edit fails exactly as before. Any problem while working out a hint leaves the editor's
own message as it was.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger
from openhands.tools.file_editor.exceptions import (
    EditorToolParameterInvalidError,
    EditorToolParameterMissingError,
    ToolError,
)


if TYPE_CHECKING:
    from openhands.tools.file_editor.editor import FileEditor


logger = get_logger(__name__)

# Places shown with a ready-to-use old_str; further places are only listed by line number (at most MAX_LISTED_LINES).
MAX_PLACES = 3
MAX_LISTED_LINES = 20
# Neighbouring lines tried, at most, to make old_str match only one place.
MAX_CONTEXT_LINES = 12
# Longer text is pointed to by its line numbers instead of being printed. A model told to view 26 lines and copy them
# retyped them 8 times with the same wrong indentation (a real heal); printed, the text only has to be copied.
MAX_BLOCK_LINES = 40
MAX_BLOCK_CHARS = 2000
# The most similar text is shown from this difflib ratio on, with at most this many differing lines.
SIMILAR_ENOUGH = 0.6
MAX_DIFF_LINES = 5
MAX_LINE_CHARS = 160
# Bigger files or longer old_str values are not searched for similar text: it would take too long.
MAX_SIMILARITY_FILE_LINES = 5000
MAX_SIMILARITY_OLD_LINES = 80
# Places compared exactly (difflib's slow ratio), the likeliest first.
MAX_SIMILARITY_CANDIDATES = 50
# When no line of the file equals a line of old_str, every place is compared, in files up to this size.
MAX_FULL_SCAN_LINES = 2000

# What `view` prints before each line: the line number right-aligned in 6 columns, then a tab ("    29\t"). Models
# also copy it with the tab dropped or turned into spaces ("    30        <dependency>").
_VIEW_NUMBER = re.compile(r"^ *(\d+)(\t| +(?=\S))")
_VIEW_NUMBER_PREFIX = re.compile(r"^ *\d+\t?")
# An empty line of `view` is its number and the tab; at the end of a value the text format of tool calls drops the tab.
_BARE_NUMBER = re.compile(r" *(\d+)")
# The editor's own message for an old_str that matches no place or several places.
_NO_REPLACEMENT = "No replacement was performed"

# Adding lines: an anchor line in old_str, not `insert`. The text format of tool calls drops the spaces at the start
# of each value, so the first line `insert` writes loses its indentation; inside new_str the new lines keep theirs.
_ADD_LINES = (
    "To add new lines, set old_str to the line just above where they go, copied exactly, and new_str to that same "
    "line followed by the new lines."
)
_EMPTY_OLD = (
    "No replacement was performed: old_str is empty (or only spaces), so it matches everywhere in the file. old_str "
    f"must be the exact text you want to replace, copied from the file. {_ADD_LINES}"
)
_MISSING = {
    "old_str": (
        " old_str is the exact text to replace, copied from the file, and new_str is the text to put in its place. "
        + _ADD_LINES
    ),
    "new_str": " new_str is the text to put in place of old_str. To delete old_str, give new_str an empty value.",
}
_SAME = (
    "No replacement was performed. `new_str` and `old_str` must be different. The editor received the same text for "
    "both, so this edit would change nothing.\n"
    "- If your change was only in spaces or line breaks at the very start or end: those were dropped from both values "
    "before they reached the editor, so the change was lost. To change the indentation of the first line, add the line "
    "above it to both old_str and new_str; to change the end of the last line, add the line below it to both.\n"
    "- Otherwise, new_str must be the text as it should look after your change: to change lines, write them changed; "
    "to delete lines, leave them out; to move lines, use an old_str that covers both their old and their new place, "
    "and write them at the new place in new_str."
)
_VIEW_NUMBERS_NOTE = (
    "Your old_str starts its lines with line numbers ({numbers}): they come from the output of `view` and are not "
    "part of the file. Leave them out."
)
_READY_TO_USE = (
    "Copy it exactly (same spaces, no line numbers). new_str replaces all of it, so new_str must be the same text with "
    "only your change made in it: repeat every line you do not change."
)
_NOTHING_SIMILAR = (
    "Nothing in the file looks like your old_str. View the file (or the lines you mean) and copy the text exactly; if "
    "you changed this file earlier, it may no longer look the way you remember."
)
# When the file (or old_str) is too big to compare every place with old_str.
_NOT_SEARCHED = (
    "No text like your old_str was found, but the file is too big to compare all of it with your old_str. View the "
    "lines you mean (with view_range) and copy them exactly; if you changed this file earlier, it may no longer look "
    "the way you remember."
)
# The text format of tool calls ends a value at the first `</parameter>`: when one is left out, the next parameter's
# tag and text become part of the value before it (and that parameter is missing).
_PARAMETER_TAG = re.compile(r"<parameter=[^>]+>")
_UNCLOSED = (
    "Your {name} contains `{tag}`, which is not in the file: {name} was not closed with `</parameter>`, so `{tag}` and "
    "the text after it were read as part of {name}. End every parameter with `</parameter>` before the next one starts."
)


def add_fix_hint(
    editor: FileEditor, error: ToolError, path: Path, old_str: str | None, new_str: str | None
) -> None:
    """Add to the message of a failed str_replace how to fix the call. Never raises."""
    try:
        message = explain(editor, error, path, old_str, new_str)
    except Exception:  # a hint must never hide the editor's own error
        logger.debug("No str_replace hint", exc_info=True)
        return
    if message:
        error.message = message
        error.args = (message,)


def explain(
    editor: FileEditor, error: ToolError, path: Path, old_str: str | None, new_str: str | None
) -> str | None:
    """The message of a failed str_replace with a hint added, or None to keep the editor's message."""
    unclosed = _unclosed_parameter(editor, path, old_str, new_str)
    if unclosed:
        return f"{error.message}\n\n{unclosed}"
    if isinstance(error, EditorToolParameterMissingError):
        hint = _MISSING.get(error.parameter)
        return error.message + hint if hint else None
    old, new = old_str, new_str
    if old is None:
        return None
    same = isinstance(error, EditorToolParameterInvalidError) and error.parameter == "new_str" and new == old
    no_match = type(error) is ToolError and error.message.startswith(_NO_REPLACEMENT)
    if not (same or no_match):
        return None
    if not old.strip():
        return _EMPTY_OLD
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


def _unclosed_parameter(editor: FileEditor, path: Path, old_str: str | None, new_str: str | None) -> str | None:
    """A value holding a parameter tag that the file does not have: the parameter before it was not closed."""
    for name, value in (("old_str", old_str), ("new_str", new_str)):
        match = _PARAMETER_TAG.search(value or "")
        if match and match.group(0) not in editor.read_file(path):
            return _UNCLOSED.format(name=name, tag=match.group(0))
    return None


def _where_old_str_is(content: str, old: str, path: Path) -> str:
    needle = old if old in content else old.strip()
    count = content.count(needle)
    if count == 1:
        if needle != old:
            return ""
        # A model "fixing" text that already looks the way it wants sends this edit again and again (a real heal).
        first, last = _span(content, content.index(needle), needle)
        one = first == last
        return (
            f"\n\n{_lines(first, last).capitalize()} of the file already {'reads' if one else 'read'} exactly like your "
            f"new_str: if that is how {'it' if one else 'they'} should look, there is nothing left to change there."
        )
    if count > 1:
        spans = [_span(content, start, needle) for start in _starts(content, needle)]
        return (
            f"\n\nAlso: this old_str matches {count} places in {path}; it must match exactly one. A ready-to-use "
            f"old_str for each place follows. {_READY_TO_USE}\n\n{_places(content, spans)}"
        )
    return f"\n\nAlso: this old_str is not in {path}.\n\n{_not_found(content, old)}"


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
    numbered = _view_numbers(old)
    old_lines = old.split("\n")
    # Rows that start with a number and a tab (a TSV file) are the file's own text when the file has them.
    own = [old_lines[index].strip() for index in numbered if _VIEW_NUMBER_PREFIX.sub("", old_lines[index], count=1).strip()]
    if own and 2 * sum(line in content for line in own) >= len(own):
        numbered = {}
    if numbered:
        cleaned = "\n".join(
            _VIEW_NUMBER_PREFIX.sub("", line, count=1) if index in numbered else line
            for index, line in enumerate(old_lines)
        )
        needle = cleaned if cleaned in content else cleaned.strip()
        count = content.count(needle) if needle.strip() else 0
        # One numbered line could be real text (a table row): it counts only when the text without it is in the file.
        if not (count or len(numbered) > 1 or _same_ignoring_spaces(content, lines, cleaned)):
            numbered = {}
    if numbered:
        text = cleaned
        numbers = list(numbered.values())
        shown = ", ".join(str(number) for number in numbers[:3]) + (", ..." if len(numbers) > 3 else "")
        notes.append(_VIEW_NUMBERS_NOTE.format(numbers=shown))
        if count == 1:
            first, last = _span(content, content.index(needle), needle)
            if _unique(content, needle):
                notes.append(f"Without them, old_str matches {_lines(first, last)}, and only there:\n{_block(needle)}")
            else:  # it would match more places once the text format drops its outer spaces
                notes.append(f"Without them, old_str matches {_lines(first, last)}.\n\n{_places(content, [(first, last)])}")
            return "\n\n".join(notes)
        if count > 1:
            spans = [_span(content, start, needle) for start in _starts(content, needle)]
            notes.append(
                f"Without them, old_str matches {count} places; it must match exactly one. A ready-to-use old_str "
                f"for each place follows. {_READY_TO_USE}\n\n{_places(content, spans)}"
            )
            return "\n\n".join(notes)
    searched_all = len(lines) <= MAX_FULL_SCAN_LINES and len(_trimmed(text.split("\n"))) <= MAX_SIMILARITY_OLD_LINES
    notes.append(
        _same_ignoring_spaces(content, lines, text)
        or _same_ignoring_line_breaks(content, text)
        or _partly_there(content, lines, text)
        or _most_similar(content, lines, text)
        or (_NOTHING_SIMILAR if searched_all else _NOT_SEARCHED)
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


def _same_ignoring_line_breaks(content: str, text: str) -> str | None:
    """old_str is in the file if all spaces and line breaks are ignored (a wrapped statement, a blank line more or
    less): point to the whole lines."""
    words = text.split()
    if not words:
        return None
    pattern = r"\s+".join(map(re.escape, words))
    spans = [_span(content, match.start(), match.group()) for match in re.finditer(pattern, content)]
    if not spans:
        return None
    first, last = spans[0]
    more = f" (and {len(spans) - 1} more places)" if len(spans) > 1 else ""
    return (
        f"Your old_str matches {_lines(first, last)} of the file{more} if spaces, blank lines and line breaks are "
        f"ignored. A ready-to-use old_str follows. {_READY_TO_USE}\n\n{_places(content, spans)}"
    )


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
    # Where old_str would start for each of its lines that the file has, and how many of its lines agree there.
    hits: dict[int, int] = {}
    for index, line in enumerate(norm_old):
        for position in positions.get(line, ()):
            if 0 <= position - index <= last_start:
                hits[position - index] = hits.get(position - index, 0) + 1
    starts = list(hits)
    if not starts and len(lines) <= MAX_FULL_SCAN_LINES:
        starts = list(range(last_start + 1))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2("\n".join(norm_old))
    ranked = []
    for start in starts:
        matcher.set_seq1("\n".join(norm_file[start:start + size]))
        if matcher.real_quick_ratio() >= SIMILAR_ENOUGH:
            quick = matcher.quick_ratio()
            if quick >= SIMILAR_ENOUGH:
                ranked.append((-hits.get(start, 0), -quick, start))
    best_ratio, best = 0.0, None
    # The exact ratio is slow on big files (minutes for every place): only for the likeliest places.
    for _, _, start in sorted(ranked)[:MAX_SIMILARITY_CANDIDATES]:
        matcher.set_seq1("\n".join(norm_file[start:start + size]))
        ratio = matcher.ratio()
        if ratio > best_ratio or (ratio == best_ratio and best is not None and start < best):
            best_ratio, best = ratio, start
    if best is None or best_ratio < SIMILAR_ENOUGH:
        return None
    span = _lines(best + 1, best + size)
    differing = [index for index in range(size) if norm_file[best + index] != norm_old[index]]
    parts = [
        f"Nothing in the file matches your old_str, even with other spaces. The most similar text is {span} "
        f"({min(99, int(best_ratio * 100))}% alike); it differs here:"
    ]
    for index in differing[:MAX_DIFF_LINES]:
        theirs, mine = _shown_pair(lines[best + index], old[index])
        parts.append(f"- line {best + index + 1} of the file: {_code(theirs)}\n  in your old_str: {_code(mine)}")
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
        where = f"Place {number} ({_lines(first, last)})"
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
            parts.append(
                f"{where}: this old_str ({_lines(start, end)}) matches only there.{_added(first, last, start, end)}\n"
                f"{_block(text)}"
            )
    rest = spans[MAX_PLACES:]
    if rest:
        shown = ", ".join(str(first) for first, _ in rest[:MAX_LISTED_LINES])
        more = f" and {len(rest) - MAX_LISTED_LINES} more" if len(rest) > MAX_LISTED_LINES else ""
        if len(rest) == 1:
            parts.append(f"1 more place starts at line {shown}.")
        else:
            parts.append(f"{len(rest)} more places start at lines {shown}{more}.")
    return "\n\n".join(parts)


def _added(first: int, last: int, start: int, end: int) -> str:
    """Names the lines of a ready-to-use old_str (start..end) around its place (first..last): a model that deletes or
    changes "the place" must keep them."""
    added = [number for number in range(start, end + 1) if not first <= number <= last]
    if not added:
        return ""
    named = _numbers(added)
    them = "it" if len(added) == 1 else "them"
    return (
        f" {named[0].upper()}{named[1:]} {_is(added)} added only so that it matches one place: keep {them} unchanged "
        "in new_str."
    )


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
    stripped = text.strip()
    return bool(stripped) and _count(content, text) == 1 and _count(content, stripped) == 1


def _count(content: str, text: str, limit: int = 2) -> int:
    """How many places `text` matches, overlapping ones included (counting stops at `limit`): "a\\na" matches
    "a\\na\\na" twice, which str.count (one count per non-overlapping match) misses."""
    count, index = 0, content.find(text)
    while index != -1 and count < limit:
        count += 1
        index = content.find(text, index + 1)
    return count


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
    tabs_differ: list[tuple[int, str, str]] = []
    for index, mine in enumerate(old):
        number = start + index + 1
        theirs = lines[start + index]
        if index > 0:  # the first line of old_str may begin in the middle of a file line
            file_indent, old_indent = _indent(theirs), _indent(mine)
            if "\t" in file_indent and "\t" not in old_indent:
                tabs_in_file.append(number)
            elif "\t" in old_indent and "\t" not in file_indent:
                tabs_in_old.append(number)
            elif "\t" in file_indent + old_indent and file_indent != old_indent:
                tabs_differ.append((number, file_indent, old_indent))
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
    for number, in_file, in_old in tabs_differ:
        notes.append(f"line {number} starts with {_describe(in_file)} in the file but {_describe(in_old)} in your old_str")
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


def _describe(indent: str) -> str:
    """e.g. "2 tabs", "1 tab and 4 spaces" or "8 spaces"."""
    tabs = indent.count("\t")
    spaces = len(indent) - tabs
    parts = [f"{tabs} tab{'s' * (tabs != 1)}" if tabs else "", f"{spaces} space{'s' * (spaces != 1)}" if spaces else ""]
    return " and ".join(part for part in parts if part) or "no indentation"


def _view_numbers(old: str) -> dict[int, int]:
    """The line numbers of `view` at the start of old_str's lines ({line index: number}), or {} when old_str does not
    look like that: at least half of its lines numbered, one line only in view's own format (number, tab), several
    lines counting up by one per line."""
    lines = old.split("\n")
    found = []
    for index, line in enumerate(lines):
        match = _VIEW_NUMBER.match(line)
        if match:
            found.append((index, int(match.group(1)), match.group(2) == "\t"))
    # A last line that is only the next number: an empty line of `view`, its tab dropped.
    bare = _BARE_NUMBER.fullmatch(lines[-1])
    if found and bare and int(bare.group(1)) - found[-1][1] == len(lines) - 1 - found[-1][0]:
        found.append((len(lines) - 1, int(bare.group(1)), False))
    if not found or 2 * len(found) < sum(1 for line in lines if line.strip()):
        return {}
    if len(found) == 1:
        return {found[0][0]: found[0][1]} if found[0][2] else {}
    if all(later[1] - earlier[1] == later[0] - earlier[0] for earlier, later in zip(found, found[1:])):
        return {index: number for index, number, _ in found}
    return {}


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
    gap = _gap(norm_file, norm_old, from_start, start, longest)
    if gap:  # the rest of old_str is there too, after one changed line or a few lines more or less
        first, last, extra = gap
        if extra == 0:
            at = start + longest + 1 if from_start else start
            mine = old[longest] if from_start else old[len(old) - longest - 1]
            said = f"line {at}: the file has {_quoted(lines[at - 1])} where your old_str has {_quoted(mine)}"
        elif extra > 0:
            at = start + longest + 1 if from_start else start - extra + 1
            said = f"that the file has {extra} more line{'s' * (extra != 1)} there (from line {at}: {_quoted(lines[at - 1])})"
        else:
            said = f"that your old_str has {-extra} line{'s' * (extra != -1)} the file does not have"
        where = f"Your old_str matches {_lines(first, last)} of the file (spaces ignored), except {said}."
        window = "\n".join(lines[first - 1:last])
        if _unique(content, window) and last - first + 1 <= MAX_BLOCK_LINES and len(window) <= MAX_BLOCK_CHARS:
            return (
                f"{where}\nThe exact text of {_lines(first, last)}, ready to use as old_str (it matches only there):\n"
                f"{_block(window)}\n{_READY_TO_USE}"
            )
        return f"{where}\n\n{_places(content, [(first, last)])}"
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
        part, rest = ("start", "after") if from_start else ("end", "before")
        return (
            f"{where}\nThe exact text of {span} (it matches only there) is only the {part} of your old_str: if you use "
            f"it as old_str, write new_str for these lines only (the lines {rest} them stay as they are).\n"
            f"{_block(window)}"
        )
    return where


def _gap(
    norm_file: list[str], norm_old: list[str], from_start: bool, start: int, longest: int, max_gap: int = 5
) -> tuple[int, int, int] | None:
    """When the part of old_str that _partly_there found is followed (or preceded) in the file by the rest of old_str
    after one changed line (extra == 0), after `extra` more file lines (extra > 0) or with -extra lines of old_str
    skipped (extra < 0): (first, last, extra), lines 1-based. Else None."""
    size = len(norm_old)

    def fits(window: list[str], wanted: list[str], first_is_end: bool, last_is_start: bool) -> bool:
        if len(window) != len(wanted) or not wanted:
            return False
        for index, (theirs, mine) in enumerate(zip(window, wanted)):
            if index == 0 and first_is_end:
                ok = theirs.endswith(mine)
            elif index == len(wanted) - 1 and last_is_start:
                ok = theirs.startswith(mine)
            else:
                ok = theirs == mine
            if not ok:
                return False
        return True

    if from_start:
        rest, after = norm_old[longest:], start + longest
        if len(rest) > 1 and fits(norm_file[after + 1:after + len(rest)], rest[1:], False, True):
            return start + 1, after + len(rest), 0
        for extra in range(1, max_gap + 1):
            if fits(norm_file[after + extra:after + extra + len(rest)], rest, False, True):
                return start + 1, after + extra + len(rest), extra
        for skip in range(1, min(max_gap, len(rest) - 1) + 1):
            if fits(norm_file[after:after + len(rest) - skip], rest[skip:], False, True):
                return start + 1, after + len(rest) - skip, -skip
    else:
        head, before = norm_old[:size - longest], start
        if len(head) > 1 and before - len(head) >= 0 and fits(
            norm_file[before - len(head):before - 1], head[:-1], True, False
        ):
            return before - len(head) + 1, start + longest, 0
        for extra in range(1, max_gap + 1):
            first = before - extra - len(head)
            if first >= 0 and fits(norm_file[first:first + len(head)], head, True, False):
                return first + 1, start + longest, extra
        for skip in range(1, min(max_gap, len(head) - 1) + 1):
            first = before - (len(head) - skip)
            if first >= 0 and fits(norm_file[first:before], head[:len(head) - skip], True, False):
                return first + 1, start + longest, -skip
    return None


def _quoted(line: str) -> str:
    return _code(_short(line))


def _code(text: str) -> str:
    return f"`{text}`" if text.strip() else "an empty line"


def _shown_pair(theirs: str, mine: str) -> tuple[str, str]:
    """Both lines (without outer spaces), cut around their first difference when long, with invisible characters
    written out: otherwise two long locators that differ near their end, or by a zero-width space, look the same."""
    a, b = theirs.strip(), mine.strip()
    same = 0
    while same < min(len(a), len(b)) and a[same] == b[same]:
        same += 1
    start = max(0, same - MAX_LINE_CHARS // 2) if max(len(a), len(b)) > MAX_LINE_CHARS else 0

    def cut(line: str) -> str:
        piece = line[start:start + MAX_LINE_CHARS]
        piece = "".join(char if char.isprintable() or char == "\t" else f"<U+{ord(char):04X}>" for char in piece)
        return ("…" if start else "") + piece + ("…" if start + MAX_LINE_CHARS < len(line) else "")

    return cut(a), cut(b)


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
