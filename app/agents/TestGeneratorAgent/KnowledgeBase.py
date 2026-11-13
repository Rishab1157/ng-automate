"""The user's knowledge base for the test generator: every text file the user gave the project, as context. Plain
code, no LLM.

The test cases come from one test-data file, but what a test needs is often spread over several: a story or manual
test cases without locators, a sheet of locators, a note with accounts, written in any form or language. A person
reads them all and puts them together; so does the agent, but only when it sees them. A real run with the files only
mounted: the agent never opened them and made up the two locators that only the files held. So their text goes into
the generator's first message as it is (no format is parsed or assumed), cut at a size a small model reads well; a
file that is cut or does not fit is named, to be read in the sandbox. Files that are not text (PDF, spreadsheets) are
left out. The files are read-only in the sandbox: the agent uses them, never changes them.
"""

from pathlib import Path

from app.services.sandboxService.SandboxService import SANDBOX_TEST_DATA_DIR

MAX_FILE_CHARS = 6_000
MAX_TOTAL_CHARS = 16_000
MAX_FILE_BYTES = 1_000_000
CUT_NOTE = "[... the rest of this file is in {path}]"
NOT_SHOWN = "Not shown here (too long for this message): {paths}. Read them in the sandbox."


def knowledge_base(test_data_dir: Path | None) -> str:
    """The knowledge base as text for the generator's message, "" when the project has no text file to show."""
    files = _text_files(test_data_dir)
    shown: list[str] = []
    skipped: list[str] = []
    used = 0
    for name, text in files:
        path = f"{SANDBOX_TEST_DATA_DIR}/{name}"
        body = text.strip()
        if len(body) > MAX_FILE_CHARS:
            body = body[:MAX_FILE_CHARS].rstrip() + "\n" + CUT_NOTE.format(path=path)
        if used + len(body) > MAX_TOTAL_CHARS:
            skipped.append(path)
            continue
        shown.append(f"=== {path} ===\n{body}")
        used += len(body)
    if skipped:
        shown.append(NOT_SHOWN.format(paths=", ".join(skipped)))
    return "\n\n".join(shown)


def _text_files(folder: Path | None) -> list[tuple[str, str]]:
    """(name, text) of the folder's text files, by name: hidden files, folders, empty, very large and binary files (a
    NUL byte, or not UTF-8: PDF, spreadsheets, images) are left out."""
    if folder is None or not folder.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith(".") or not path.is_file() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if not data.strip() or b"\0" in data:
            continue
        try:
            # Windows (\r\n) and old Mac (\r) line ends read as plain \n: the same text whatever wrote the file.
            found.append((path.name, data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")))
        except UnicodeDecodeError:
            continue
    return found
