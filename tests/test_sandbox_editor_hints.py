"""The sandbox's clearer str_replace errors (docker/sandbox-test/python/ngauto_editor_hints.py).

The agent server in the sandbox imports that module at start. Here it wraps the same FileEditor (the vendored SDK,
the version the sandbox image runs) to check that every hint is accurate: its line numbers, and that each
ready-to-use old_str really matches only one place and works when sent back.
"""

import ast
import importlib.util
import re
import time
from pathlib import Path

import pytest
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError

HINTS_FILE = Path(__file__).resolve().parents[1] / "docker" / "sandbox-test" / "python" / "ngauto_editor_hints.py"
# Under another name than the agent server's, so that loading it does not wrap the editor for every test.
_spec = importlib.util.spec_from_file_location("ngauto_editor_hints_under_test", HINTS_FILE)
assert _spec is not None and _spec.loader is not None
hints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hints)

# pom.xml as a real heal left it (2026-10-14): the same dependency added twice, after </dependencies>.
POM_LINES = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<project xmlns="http://maven.apache.org/POM/4.0.0"',
    '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
    '         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">',
    "    <modelVersion>4.0.0</modelVersion>",
    "",
    "    <groupId>com.demo</groupId>",
    "    <artifactId>saucedemo-ui-tests</artifactId>",
    "    <version>1.0-SNAPSHOT</version>",
    "",
    "    <properties>",
    "        <maven.compiler.release>17</maven.compiler.release>",
    "        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>",
    "    </properties>",
    "",
    "    <dependencies>",
    "        <dependency>",
    "            <groupId>org.seleniumhq.selenium</groupId>",
    "            <artifactId>selenium-java</artifactId>",
    "            <version>4.27.0</version>",
    "            <scope>test</scope>",
    "        </dependency>",
    "        <dependency>",
    "            <groupId>org.testng</groupId>",
    "            <artifactId>testng</artifactId>",
    "            <version>7.6.0</version>",
    "            <scope>test</scope>",
    "        </dependency>",
    "    </dependencies>",
    "        <dependency>",
    "            <groupId>org.seleniumhq.selenium</groupId>",
    "            <artifactId>selenium-devtools-v86</artifactId>",
    "            <version>4.27.0</version>",
    "            <scope>test</scope>",
    "        </dependency>",
    "        <dependency>",
    "            <groupId>org.seleniumhq.selenium</groupId>",
    "            <artifactId>selenium-devtools-v86</artifactId>",
    "            <version>4.27.0</version>",
    "            <scope>test</scope>",
    "        </dependency>",
    "    ",
    "",
    "    <build>",
    "        <plugins>",
    "            <plugin>",
    "                <groupId>org.apache.maven.plugins</groupId>",
    "                <artifactId>maven-compiler-plugin</artifactId>",
    "                <version>3.8.1</version>",
    "                <configuration>",
    "                    <source>1.8</source>",
    "                    <target>1.8</target>",
    "                </configuration>",
    "            </plugin>",
    "            <plugin>",
    "                <groupId>org.apache.maven.plugins</groupId>",
    "                <artifactId>maven-surefire-plugin</artifactId>",
    "                <version>3.5.2</version>",
    "            </plugin>",
    "        </plugins>",
    "    </build>",
    "</project>",
]
BROKEN_POM = "\n".join(POM_LINES) + "\n"
# What the model sent in that heal: the copy with 4 spaces of indentation, where the file has 8.
MODELS_OLD_STR = (
    "</dependencies>\n    <dependency>\n        <groupId>org.seleniumhq.selenium</groupId>\n"
    "        <artifactId>selenium-devtools-v86</artifactId>\n        <version>4.27.0</version>\n"
    "        <scope>test</scope>\n    </dependency>"
)
BASE_TEST = """package com.demo.base;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;

/** Opens Chrome on the shop before each test and closes it after. Pass -Dheadless=true to hide the browser. */
public abstract class BaseTest {
    public static final String BASE_URL = "https://www.saucedemo.com/";

    protected WebDriver driver;

    @BeforeMethod
    public void openBrowser() {
        ChromeOptions options = new ChromeOptions();
        if (Boolean.getBoolean("headless")) {
            options.addArguments("--headless=new");
        }
        options.addArguments("--window-size=1366,900");
        driver = new ChromeDriver(options);
        driver.get(BASE_URL);
    }

    @AfterMethod(alwaysRun = true)
    public void closeBrowser() {
        if (driver != null) {
            driver.quit();
        }
    }
}
"""


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> FileEditor:
    monkeypatch.setattr(FileEditor, "__call__", FileEditor.__call__)  # put back as it was after the test
    assert hints.patch_file_editor(FileEditor)
    return FileEditor()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _fail(editor: FileEditor, path: Path, old: str | None, new: str | None = "changed") -> str:
    call = {"command": "str_replace", "path": str(path)}
    if old is not None:
        call["old_str"] = old
    if new is not None:
        call["new_str"] = new
    with pytest.raises(ToolError) as caught:
        editor(**call)
    return caught.value.message


def _blocks(message: str) -> list[str]:
    return re.findall(r"^(?:```|~~~~)\n(.*?)\n(?:```|~~~~)$", message, re.S | re.M)


def test_each_place_gets_an_old_str_that_matches_only_there(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    message = _fail(editor, pom, "<dependency>")

    # The editor's own sentence stays first.
    assert message.startswith(
        "No replacement was performed. Multiple occurrences of old_str `<dependency>` in lines [17, 23, 30, 36]. "
        "Please ensure it is unique.\n\nWhy: old_str must match exactly one place"
    )
    assert "and it matches 4." in message
    assert "Place 1 (starts at line 17): this old_str (lines 16-17) matches only there:" in message
    assert "Place 2 (starts at line 23): this old_str (lines 23-24) matches only there:" in message
    assert "Place 3 (starts at line 30): this old_str (lines 29-30) matches only there:" in message
    assert message.endswith("1 more place starts at line 36.")
    blocks = _blocks(message)
    assert blocks == ["\n".join(POM_LINES[15:17]), "\n".join(POM_LINES[22:24]), "\n".join(POM_LINES[28:30])]
    for block in blocks:
        assert BROKEN_POM.count(block) == 1 and BROKEN_POM.count(block.strip()) == 1


@pytest.mark.parametrize("trim", [False, True])
def test_the_ready_to_use_old_str_works_when_sent_back(editor: FileEditor, tmp_path: Path, trim: bool) -> None:
    # The text format of tool calls drops the spaces at the start and end of each value (trim=True).
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    third = _blocks(_fail(editor, pom, "<dependency>"))[2]
    old = third.strip() if trim else third

    editor(command="str_replace", path=str(pom), old_str=old, new_str=old + " <!-- here -->")

    assert pom.read_text(encoding="utf-8").count("<!-- here -->") == 1
    assert "    </dependencies>\n        <dependency> <!-- here -->" in pom.read_text(encoding="utf-8")


def test_other_indentation_is_named_and_the_exact_text_given(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    message = _fail(editor, pom, MODELS_OLD_STR, "</dependencies>")

    assert message.startswith("No replacement was performed, old_str `</dependencies>")
    assert (
        "Your old_str matches lines 29-35 of the file if spaces are ignored, but not character for character: "
        "lines 30-35 start with 4 more spaces in the file than in your old_str (for example line 30: 8 in the file, "
        "4 in yours).\nThe exact text of lines 29-35, ready to use as old_str (it matches only there):"
    ) in message
    [block] = _blocks(message)
    assert block == "\n".join(POM_LINES[28:35])
    # With it, the edit the model wanted works: drop the first copy.
    editor(command="str_replace", path=str(pom), old_str=block, new_str="    </dependencies>")
    assert pom.read_text(encoding="utf-8").count("selenium-devtools-v86") == 1


def test_the_same_old_and_new_str_is_explained(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    assert _fail(editor, pom, "</dependencies>", "</dependencies>") == hints._SAME  # it matches one place: fine
    not_there = _fail(editor, pom, MODELS_OLD_STR, MODELS_OLD_STR)
    several = _fail(editor, pom, "<scope>test</scope>", "<scope>test</scope>")

    assert not_there == f"{hints._SAME}\n\nAlso: this old_str is not in {pom}: view the file and copy the text exactly."
    assert several.startswith(hints._SAME) and f"Also: this old_str matches 4 places in {pom}" in several


@pytest.mark.parametrize("old", ["", "   \n  "])
def test_an_empty_old_str_is_explained_without_listing_every_line(
    editor: FileEditor, tmp_path: Path, old: str
) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    assert _fail(editor, pom, old, "<!-- new -->") == hints._EMPTY_OLD


def test_missing_values_are_explained(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    assert _fail(editor, pom, None, "x") == "Parameter `old_str` is required for command: str_replace." + hints._MISSING["old_str"]
    assert _fail(editor, pom, "<modelVersion>4.0.0</modelVersion>", None) == (
        "Parameter `new_str` is required for command: str_replace." + hints._MISSING["new_str"]
    )


@pytest.mark.parametrize("trim", [False, True])
def test_line_numbers_copied_from_view_are_named(editor: FileEditor, tmp_path: Path, trim: bool) -> None:
    # As in a real heal: the model copied the line numbers of `view` into old_str.
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)
    lines = BASE_TEST.split("\n")
    old = f"{9:6}\t{lines[8]}\n{10:6}\t{lines[9]}"

    message = _fail(editor, path, old.strip() if trim else old)

    assert hints._VIEW_NUMBERS_NOTE.format(numbers="9, 10") in message
    assert "Without them, old_str matches lines 9-10, and only there:" in message
    assert _blocks(message) == [f"{lines[8]}\n{lines[9]}"]


def test_line_numbers_copied_with_spaces_instead_of_the_tab_are_named(editor: FileEditor, tmp_path: Path) -> None:
    # A real call (trial 1, 2026-10-14): `view`'s tab was dropped, the numbers kept, the first line unnumbered.
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    old = "</dependencies>\n" + "\n".join(f"{number:6}{POM_LINES[number - 1]}" for number in range(30, 42))

    message = _fail(editor, pom, old, "</dependencies>")

    assert hints._VIEW_NUMBERS_NOTE.format(numbers="30, 31, 32, ...") in message
    assert "Without them, old_str matches lines 29-41, and only there:" in message
    assert _blocks(message) == ["</dependencies>\n" + "\n".join(POM_LINES[29:41])]


def test_a_table_row_is_not_taken_for_a_line_number(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "people.tsv", "id\tname\n1\talice\n2\tbob\n")

    message = _fail(editor, path, "3\tcarol")

    assert "line numbers" not in message


def test_where_only_the_start_of_old_str_is_in_the_file(editor: FileEditor, tmp_path: Path) -> None:
    # A real call (trial 1): old_str had both copies, but the file had only one left. The most similar text of the same
    # length was the project's own <dependencies> block (lines 16-28): a misleading place to point at.
    one_copy = POM_LINES[:35] + POM_LINES[41:]
    pom = _write(tmp_path, "pom.xml", "\n".join(one_copy) + "\n")
    old = "\n".join(POM_LINES[28:41]).strip()

    message = _fail(editor, pom, old, "</dependencies>")

    assert (
        "Only the first 7 lines of your old_str are in the file (lines 29-35, spaces ignored); then line 36 of the "
        "file is an empty line where your old_str has `<dependency>`."
    ) in message
    assert _blocks(message) == ["\n".join(one_copy[28:35])]
    assert "16-28" not in message


def test_where_only_the_end_of_old_str_is_in_the_file(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)
    lines = BASE_TEST.split("\n")
    old = '        options.addArguments("--start-maximized");\n' + "\n".join(lines[21:23])

    message = _fail(editor, path, old)

    assert (
        "Only the last 2 lines of your old_str are in the file (lines 22-23, spaces ignored); before them line 21 of "
        'the file is `options.addArguments("--window-size=1366,900");` where your old_str has '
        '`options.addArguments("--start-maximized");`.'
    ) in message
    assert _blocks(message) == ["\n".join(lines[21:23])]


def test_tabs_and_spaces_at_the_end_of_a_line_are_named(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "main.py", "def main():\n\tif ready:  \n\t\trun()\n\treturn 0\n")

    message = _fail(editor, path, "if ready:\n        run()\n    return 0")

    assert (
        "Your old_str matches lines 2-4 of the file if spaces are ignored, but not character for character: lines "
        "3-4 are indented with tabs in the file but spaces in yours; line 2 ends with spaces in the file that your "
        "old_str does not have."
    ) in message
    [block] = _blocks(message)
    assert block == "\tif ready:  \n\t\trun()\n\treturn 0"
    editor(command="str_replace", path=str(path), old_str=block, new_str="\tif ready:\n\t\trun()\n\treturn 1")
    assert path.read_text(encoding="utf-8") == "def main():\n\tif ready:\n\t\trun()\n\treturn 1\n"


def test_the_most_similar_text_is_shown(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)

    message = _fail(
        editor, path, '        options.addArguments("--window-size=1280,900");\n        driver = new ChromeDriver(options);'
    )

    assert re.search(r"The most similar text is lines 21-22 \(\d+% alike\); it differs here:", message)
    assert (
        '- line 21 of the file: `options.addArguments("--window-size=1366,900");`\n'
        '  in your old_str: `options.addArguments("--window-size=1280,900");`'
    ) in message
    assert _blocks(message) == ['        options.addArguments("--window-size=1366,900");\n        driver = new ChromeDriver(options);']


def test_nothing_similar_says_to_view_the_file(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)

    assert _fail(editor, path, "SELECT * FROM users WHERE id = 7;").endswith(hints._NOTHING_SIMILAR)


def test_many_places_are_listed_briefly(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "many.py", "start\n" + "x = 1\n" * 30 + "end\n")

    message = _fail(editor, path, "x = 1")

    assert "Place 1 (starts at line 2): this old_str (lines 1-2) matches only there:" in message
    assert "Place 2 (starts at line 3): this old_str (lines 1-3) matches only there:" in message
    assert "Place 3 (starts at line 4): this old_str (lines 1-4) matches only there:" in message
    assert message.endswith(
        "27 more places start at lines 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 "
        "and 7 more."
    )


def test_a_place_without_unique_neighbours_says_so(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "same.txt", "x = 1\n" * 40)

    message = _fail(editor, path, "x = 1")

    assert "Place 1 (starts at line 1): no text of up to 12 lines around it matches only there" in message


def test_windows_line_endings_are_handled(editor: FileEditor, tmp_path: Path) -> None:
    path = tmp_path / "pom.xml"
    path.write_bytes(BROKEN_POM.replace("\n", "\r\n").encode("utf-8"))

    third = _blocks(_fail(editor, path, "<dependency>"))[2]
    editor(command="str_replace", path=str(path), old_str=third, new_str=third + " <!-- here -->")

    assert "<!-- here -->" in path.read_text(encoding="utf-8")


def test_successful_edits_and_other_errors_are_unchanged(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    plain = FileEditor.__call__.__wrapped__  # the editor without hints

    result = editor(command="str_replace", path=str(pom), old_str="<version>7.6.0</version>", new_str="<version>7.10.2</version>")
    with pytest.raises(ToolError) as hinted:
        editor(command="view", path=str(pom), view_range=[0, 5])
    with pytest.raises(ToolError) as original:
        plain(FileEditor(), command="view", path=str(pom), view_range=[0, 5])

    assert not result.is_error and "<version>7.10.2</version>" in pom.read_text(encoding="utf-8")
    assert hinted.value.message == original.value.message


def test_a_failing_hint_keeps_the_editor_message(
    editor: FileEditor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    def broken(*args: object) -> str:
        raise RuntimeError("bug in the hint")

    monkeypatch.setattr(hints, "explain", broken)

    assert _fail(editor, pom, "<dependency>") == (
        "No replacement was performed. Multiple occurrences of old_str `<dependency>` in lines [17, 23, 30, 36]. "
        "Please ensure it is unique."
    )


def test_the_editor_is_wrapped_once(editor: FileEditor) -> None:
    wrapped = FileEditor.__call__

    assert hints.patch_file_editor(FileEditor)

    assert FileEditor.__call__ is wrapped


def test_a_big_file_is_answered_quickly(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "big.txt", "".join(f"line {number}\n" for number in range(20000)))

    started = time.monotonic()
    message = _fail(editor, path, "no such line\nnor this one")

    assert time.monotonic() - started < 5
    assert message.endswith(hints._NOTHING_SIMILAR)


def test_only_modules_the_frozen_agent_server_bundles_are_imported() -> None:
    # The agent server is a PyInstaller program: it can import its own modules and the standard library it bundles.
    tree = ast.parse(HINTS_FILE.read_text(encoding="utf-8"))
    top_level = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    in_functions = {
        sub.module
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for sub in ast.walk(node)
        if isinstance(sub, ast.ImportFrom)
    }

    assert top_level <= {"__future__", "difflib", "functools", "logging", "re", "pathlib"}
    assert in_functions <= {"openhands.tools.file_editor.editor", "openhands.tools.file_editor.exceptions"}
