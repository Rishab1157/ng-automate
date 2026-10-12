"""A short, read-only briefing of the project's own tests and helpers, inlined into the generator prompt.

A small local model tends to invent helper methods (open/type/click) instead of reading the base class first,
and then tries to add them, which the guards roll back. Handing it the project's own base class, helpers and
locator/config files in the prompt makes it follow the real API from its first file. The briefing only reads
existing files; it never changes the project.
"""

from pathlib import Path

# Keep the briefing small: a handful of files, a few thousand characters.
MAX_FILES = 8
MAX_TOTAL_CHARS = 10000
MAX_FILE_CHARS = 3000
# Names like a base class, a helper, or where locators/config live: shown first, because the model needs the API.
PRIORITY_HINTS = (
    "base", "abstract", "helper", "common", "shared", "support", "util", "fixture", "conftest",
    "driver", "config", "locator", "selector", "page", "step", "hook", "world",
)
SOURCE_SUFFIXES = {
    ".java", ".kt", ".kts", ".groovy", ".scala", ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".rb", ".cs", ".go", ".rs", ".php", ".robot", ".resource", ".feature", ".properties", ".ini",
    ".cfg", ".conf", ".yaml", ".yml", ".toml", ".json",
}
SKIP_DIRS = {
    "node_modules", "target", "build", "out", "dist", "bin", "obj", "vendor", ".git", ".venv",
    "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache", ".gradle", ".idea", ".vscode",
}


def project_test_briefing(project_dir: Path, test_dirs: list[str], generated_dir: str) -> str:
    """The project's own test files and helpers as text, ready to inline. Empty string when there is nothing."""
    root = Path(project_dir).resolve()
    files = _select_files(root, test_dirs, generated_dir)
    sections: list[str] = []
    used = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        suffix = ""
        if len(text) > MAX_FILE_CHARS:
            text, suffix = text[:MAX_FILE_CHARS], "\n... (truncated)"
        section = f"--- {path.relative_to(root).as_posix()} ---\n{text}{suffix}"
        if used and used + len(section) > MAX_TOTAL_CHARS:
            break
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)


def _select_files(project_dir: Path, test_dirs: list[str], generated_dir: str) -> list[Path]:
    candidates: list[Path] = []
    for test_dir in test_dirs or []:
        root = project_dir / test_dir
        if root.is_dir():
            candidates.extend(_walk(root, generated_dir))
    # Distinct, priority-named first, then smaller files (a base class is usually small and central).
    unique = sorted({path.resolve() for path in candidates}, key=lambda p: (_priority(p.name), _size(p), str(p)))
    return [path for path in unique[:MAX_FILES]]


def _walk(root: Path, generated_dir: str) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part == generated_dir for part in path.parts):
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            found.append(path)
    return found


def _priority(name: str) -> int:
    lower = name.lower()
    return 0 if any(hint in lower for hint in PRIORITY_HINTS) else 1


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
