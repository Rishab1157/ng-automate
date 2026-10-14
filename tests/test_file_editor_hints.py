"""Clearer str_replace errors of the vendored file editor (openhands-tools .../file_editor/utils/str_replace_hints.py).

The sandbox's agent server runs this same vendored code. These tests check that every hint is accurate: its line
numbers, and that each ready-to-use old_str really matches only one place and works when sent back.
"""

import re
import time
from pathlib import Path

import pytest
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError
from openhands.tools.file_editor.utils import str_replace_hints as hints

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
def editor() -> FileEditor:
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
    added = "is added only so that it matches one place: keep it unchanged in new_str."
    assert f"Place 1 (line 17): this old_str (lines 16-17) matches only there. Line 16 {added}" in message
    assert f"Place 2 (line 23): this old_str (lines 23-24) matches only there. Line 24 {added}" in message
    assert f"Place 3 (line 30): this old_str (lines 29-30) matches only there. Line 29 {added}" in message
    assert message.endswith("1 more place starts at line 36.")
    blocks = _blocks(message)
    assert blocks == ["\n".join(POM_LINES[15:17]), "\n".join(POM_LINES[22:24]), "\n".join(POM_LINES[28:30])]
    for block in blocks:
        assert BROKEN_POM.count(block) == 1 and BROKEN_POM.count(block.strip()) == 1


def test_deleting_a_duplicate_as_the_hint_says_keeps_the_file_whole(editor: FileEditor, tmp_path: Path) -> None:
    # The pom.xml case: the same block twice. The ready-to-use old_str of a copy has a line of its neighbour in it,
    # and the hint names it, so that deleting "the copy" does not delete that line too.
    import xml.etree.ElementTree as ElementTree

    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    copy = "\n".join(POM_LINES[35:41]).strip()

    message = _fail(editor, pom, copy, "")

    assert message.endswith(
        "Place 2 (lines 36-41): this old_str (lines 35-41) matches only there. Line 35 is added only so that it "
        f"matches one place: keep it unchanged in new_str.\n```\n{chr(10).join(POM_LINES[34:41])}\n```"
    )
    # Done as told (text format: values without outer spaces): line 35 stays, the second copy goes.
    second = _blocks(message)[1]
    editor(command="str_replace", path=str(pom), old_str=second.strip(), new_str=POM_LINES[34].strip())
    text = pom.read_text(encoding="utf-8")
    assert text.count("selenium-devtools-v86") == 1
    ElementTree.fromstring(text.encode("utf-8"))  # still well-formed


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


def test_a_long_block_with_other_indentation_gets_its_exact_text(editor: FileEditor, tmp_path: Path) -> None:
    # Trial 2 (2026-10-14): a 26-line old_str, its lines 4 spaces short. Told to view and copy the lines, the model
    # retyped them 8 times with the same mistake.
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    old = "\n".join(line[4:] for line in POM_LINES[15:41])

    message = _fail(editor, pom, old, "</dependencies>")

    assert "Your old_str matches lines 16-41 of the file if spaces are ignored" in message
    assert _blocks(message) == ["\n".join(POM_LINES[15:41])]


def test_the_same_old_and_new_str_is_explained(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    # A real loop (trial 2): the model "fixed" the indentation of text that already had it, again and again.
    already = "    </dependencies>\n        <dependency>".strip()
    assert _fail(editor, pom, already, already) == (
        f"{hints._SAME}\n\nLines 29-30 of the file already read exactly like your new_str: if that is how they should "
        "look, there is nothing left to change there."
    )
    assert _fail(editor, pom, "<modelVersion>4.0.0</modelVersion>", "<modelVersion>4.0.0</modelVersion>").endswith(
        "\n\nLine 5 of the file already reads exactly like your new_str: if that is how it should look, there is "
        "nothing left to change there."
    )
    not_there = _fail(editor, pom, MODELS_OLD_STR, MODELS_OLD_STR)
    several = _fail(editor, pom, "<scope>test</scope>", "<scope>test</scope>")

    # The second problem gets the same help as on its own: the exact text, or a ready-to-use old_str per place.
    assert not_there.startswith(f"{hints._SAME}\n\nAlso: this old_str is not in {pom}.\n\nYour old_str matches lines 29-35")
    assert _blocks(not_there) == ["\n".join(POM_LINES[28:35])]
    assert several.startswith(f"{hints._SAME}\n\nAlso: this old_str matches 4 places in {pom}; it must match exactly one.")
    assert len(_blocks(several)) == 3 and several.endswith("1 more place starts at line 40.")


def test_the_same_text_names_the_dropped_spaces_and_the_line_to_add(editor: FileEditor, tmp_path: Path) -> None:
    # A real loop (12 calls): the model changed only the first line's indentation, the text format dropped it.
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)

    message = _fail(editor, path, "driver.get(BASE_URL);", "driver.get(BASE_URL);")

    assert "those were dropped from both values before they reached the editor" in message
    assert "To change the indentation of the first line, add the line above it to both old_str and new_str" in message
    # Done as told: the line above goes into both values, and the indentation change is kept.
    lines = BASE_TEST.split("\n")
    old = f"{lines[21]}\n{lines[22]}".strip()
    editor(command="str_replace", path=str(path), old_str=old, new_str=f"{lines[21]}\n            driver.get(BASE_URL);".strip())
    assert "\n            driver.get(BASE_URL);\n" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("old", ["", "   \n  "])
def test_an_empty_old_str_is_explained_without_listing_every_line(
    editor: FileEditor, tmp_path: Path, old: str
) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    assert _fail(editor, pom, old, "<!-- new -->") == hints._EMPTY_OLD


def test_adding_lines_as_the_hint_says_keeps_their_indentation(editor: FileEditor, tmp_path: Path) -> None:
    # The hint says to add lines with an anchor line, not `insert`: the text format drops the spaces at the start of
    # each value, and `insert` would write the first new line without its indentation.
    path = _write(tmp_path, "login_page.py", "class LoginPage:\n    def __init__(self, driver):\n        self.driver = driver\n")
    assert "set old_str to the line just above where they go" in hints._EMPTY_OLD

    editor(
        command="str_replace",
        path=str(path),
        old_str="        self.driver = driver".strip(),
        new_str="        self.driver = driver\n        self.timeout = 10".strip(),
    )

    assert path.read_text(encoding="utf-8").endswith("        self.driver = driver\n        self.timeout = 10\n")


def test_missing_values_are_explained(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    assert _fail(editor, pom, None, "x") == "Parameter `old_str` is required for command: str_replace." + hints._MISSING["old_str"]
    assert _fail(editor, pom, "<modelVersion>4.0.0</modelVersion>", None) == (
        "Parameter `new_str` is required for command: str_replace." + hints._MISSING["new_str"]
    )


def test_an_unclosed_parameter_is_named(editor: FileEditor, tmp_path: Path) -> None:
    # A real call (trial 1, three times): old_str was not closed with </parameter>, so the text format read
    # <parameter=new_str> and the new text into old_str, and new_str was missing.
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    message = _fail(editor, pom, "</dependencies>\n<parameter=new_str>\n</dependencies>\n<dependency>", None)

    assert message == (
        "Parameter `new_str` is required for command: str_replace.\n\n"
        "Your old_str contains `<parameter=new_str>`, which is not in the file: old_str was not closed with "
        "`</parameter>`, so `<parameter=new_str>` and the text after it were read as part of old_str. End every "
        "parameter with `</parameter>` before the next one starts."
    )
    # A file that has such a tag itself (docs about the format): no such claim.
    docs = _write(tmp_path, "format.md", "Write <parameter=name>value</parameter> for each value.\n")
    assert "was not closed" not in _fail(editor, docs, "<parameter=name>value</parameter> for every value")


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


def test_a_copied_view_range_that_ends_on_an_empty_line_is_named(editor: FileEditor, tmp_path: Path) -> None:
    # The text format drops the tab after the last line's number when that line is empty: only "    12" is left.
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)
    lines = BASE_TEST.split("\n")
    old = "\n".join(f"{number:6}\t{lines[number - 1]}" for number in range(10, 13)).strip()
    assert old.endswith("\n    12")

    message = _fail(editor, path, old)

    assert hints._VIEW_NUMBERS_NOTE.format(numbers="10, 11, 12") in message
    assert "Without them, old_str matches lines 10-11, and only there:" in message
    assert [block.strip() for block in _blocks(message)] == [f"{lines[9]}\n{lines[10]}"]


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
    # The text shown is only part of old_str: new_str must then be written for that part only.
    assert (
        "The exact text of lines 29-35 (it matches only there) is only the start of your old_str: if you use it as "
        "old_str, write new_str for these lines only (the lines after them stay as they are)."
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


# Found by the accuracy review (2026-10-14): each was a false or misleading hint before.


def test_a_ready_to_use_old_str_matches_one_place_even_counting_overlaps(editor: FileEditor, tmp_path: Path) -> None:
    # "x\nx" matches "x\nx\nx" twice (overlapping); str.count says once, and the edit landed on another line.
    path = _write(tmp_path, "CartTest.java", "class CartTest {\n    void addThree() {\n        cart.add();\n"
                  "        cart.add();\n        cart.add();\n        check();\n    }\n}\n")

    blocks = _blocks(_fail(editor, path, "cart.add();"))

    assert len(blocks) == 3
    text = path.read_text(encoding="utf-8")
    for block in blocks:
        assert hints._count(text, block) == 1 and hints._count(text, block.strip()) == 1


def test_one_changed_line_gets_the_whole_place(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "BaseTest.java", BASE_TEST)
    lines = BASE_TEST.split("\n")
    old = "\n".join(lines[16:21]).replace('"headless"', '"headles"').strip()

    message = _fail(editor, path, old, "x")

    assert (
        "Your old_str matches lines 17-21 of the file (spaces ignored), except line 18: the file has "
        '`if (Boolean.getBoolean("headless")) {` where your old_str has `if (Boolean.getBoolean("headles")) {`.'
    ) in message
    assert _blocks(message) == ["\n".join(lines[16:21])]


def test_a_line_more_in_the_file_gets_the_whole_place(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "LoginTest.java", "class LoginTest {\n    void login() {\n        open();\n"
                  '        log("start");\n        type(USER);\n        submit();\n    }\n}\n')

    message = _fail(editor, path, "void login() {\n        open();\n        type(USER);\n        submit();\n    }", "x")

    assert "matches lines 2-7 of the file (spaces ignored), except that the file has 1 more line there (from line 4" in message
    assert _blocks(message) == ['    void login() {\n        open();\n        log("start");\n        type(USER);\n'
                                "        submit();\n    }"]


@pytest.mark.parametrize(
    ("text", "old", "block"),
    [
        # a blank line left out
        ("void f() {\n    open();\n\n    check();\n}\n", "open();\n    check();\n}", "    open();\n\n    check();\n}"),
        # a statement wrapped over two lines in the file
        ("class P {\n    void login(String user,\n               String password) {\n    }\n}\n",
         "void login(String user, String password) {", "    void login(String user,\n               String password) {"),
    ],
)
def test_other_line_breaks_get_the_whole_place(editor: FileEditor, tmp_path: Path, text: str, old: str, block: str) -> None:
    path = _write(tmp_path, "P.java", text)

    message = _fail(editor, path, old, "x")

    assert "if spaces, blank lines and line breaks are ignored" in message
    assert _blocks(message) == [block]


def test_a_view_number_block_stays_unique_without_its_outer_spaces(editor: FileEditor, tmp_path: Path) -> None:
    # "    x = 1" is in the file once, but "x = 1" (what the text format sends back) twice.
    path = _write(tmp_path, "f.py", "x = 1\n\ndef f():\n    x = 1\n    return x\n")

    message = _fail(editor, path, "4\t    x = 1", "4\t    x = 2")

    assert hints._VIEW_NUMBERS_NOTE.format(numbers="4") in message
    assert _blocks(message) == ["def f():\n    x = 1"]


def test_rows_of_a_tsv_file_are_not_taken_for_view_numbers(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "users.tsv", "id\tuser\n1\tstandard_user\n2\tlocked_out_user\n3\tproblem_user\n")

    message = _fail(editor, path, "2\tlocked_out_user\n3\tproblem_usr", "x")

    assert "starts its lines with line numbers" not in message
    assert "- line 4 of the file: `3\tproblem_user`\n  in your old_str: `3\tproblem_usr`" in message


def test_tab_indentation_is_named_as_tabs(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "main.go", "func main() {\n\tif ready {\n\t\trun()\n\t}\n}\n")

    message = _fail(editor, path, "if ready {\n\trun()\n\t}", "x")

    assert "line 3 starts with 2 tabs in the file but 1 tab in your old_str" in message


def test_long_lines_are_shown_where_they_differ(editor: FileEditor, tmp_path: Path) -> None:
    locator = "    By add = By.xpath(\"//div[@id='inventory_container']/div/div/div[@class='inventory_item']" \
              "/div[@class='pricebar']//button[@data-test='add-to-cart-sauce-labs-{}']\");"
    path = _write(tmp_path, "P.java", "class P {\n" + locator.format("backpack") + "\n}\n")

    message = _fail(editor, path, locator.format("bike-light"), "x")

    assert "labs-backpack']\");`\n  in your old_str: `" in message and "labs-bike-light']\");`" in message
    assert "100% alike" not in message


def test_a_far_off_old_str_in_a_long_file_is_answered_quickly(editor: FileEditor, tmp_path: Path) -> None:
    # Every line of a 40-line old_str differs a little from the file: difflib on every place took minutes.
    lines = ["    }" if number % 7 == 6 else f'        value{number} = compute("item-{number}");' for number in range(1990)]
    path = _write(tmp_path, "Big.java", "\n".join(lines) + "\n")
    old = "\n".join(line.replace("compute", "computeAll") for line in lines[700:740])

    started = time.monotonic()
    message = _fail(editor, path, old, "x")

    assert time.monotonic() - started < 10
    assert "The most similar text is lines 701-740" in message


def test_many_places_are_listed_briefly(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "many.py", "start\n" + "x = 1\n" * 30 + "end\n")

    message = _fail(editor, path, "x = 1")

    assert "Place 1 (line 2): this old_str (lines 1-2) matches only there. Line 1 is added only" in message
    assert "Place 2 (line 3): this old_str (lines 1-3) matches only there. Lines 1-2 are added only" in message
    assert "Place 3 (line 4): this old_str (lines 1-4) matches only there. Lines 1-3 are added only" in message
    assert message.endswith(
        "27 more places start at lines 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 "
        "and 7 more."
    )


def test_a_place_without_unique_neighbours_says_so(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "same.txt", "x = 1\n" * 40)

    message = _fail(editor, path, "x = 1")

    assert "Place 1 (line 1): no text of up to 12 lines around it matches only there" in message


def test_windows_line_endings_are_handled(editor: FileEditor, tmp_path: Path) -> None:
    path = tmp_path / "pom.xml"
    path.write_bytes(BROKEN_POM.replace("\n", "\r\n").encode("utf-8"))

    third = _blocks(_fail(editor, path, "<dependency>"))[2]
    editor(command="str_replace", path=str(path), old_str=third, new_str=third + " <!-- here -->")

    assert "<!-- here -->" in path.read_text(encoding="utf-8")


def test_successful_edits_and_other_errors_are_unchanged(editor: FileEditor, tmp_path: Path) -> None:
    pom = _write(tmp_path, "pom.xml", BROKEN_POM)

    result = editor(command="str_replace", path=str(pom), old_str="<version>7.6.0</version>", new_str="<version>7.10.2</version>")
    with pytest.raises(ToolError) as view_error:
        editor(command="view", path=str(pom), view_range=[0, 5])
    with pytest.raises(ToolError) as insert_error:
        editor(command="insert", path=str(pom), insert_line=999, new_str="x")

    assert not result.is_error and "<version>7.10.2</version>" in pom.read_text(encoding="utf-8")
    # Only failed str_replace calls get a hint.
    for message in (view_error.value.message, insert_error.value.message):
        assert "Why:" not in message and "old_str" not in message


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


def test_a_big_file_is_answered_quickly(editor: FileEditor, tmp_path: Path) -> None:
    path = _write(tmp_path, "big.txt", "".join(f"line {number}\n" for number in range(20000)))

    started = time.monotonic()
    message = _fail(editor, path, "no such line\nnor this one")

    assert time.monotonic() - started < 5
    # It was not compared with every place, so it does not claim that nothing in the file is like it.
    assert message.endswith(hints._NOT_SEARCHED)


def test_the_executor_hands_the_hint_to_the_model(tmp_path: Path) -> None:
    # The agent gets what FileEditorExecutor makes of the error: the message with the hint.
    from openhands.tools.file_editor.definition import FileEditorAction
    from openhands.tools.file_editor.impl import FileEditorExecutor

    pom = _write(tmp_path, "pom.xml", BROKEN_POM)
    executor = FileEditorExecutor(workspace_root=str(tmp_path))

    observation = executor(FileEditorAction(command="str_replace", path=str(pom), old_str="<dependency>", new_str="x"))

    assert observation.is_error
    assert "Place 3 (line 30): this old_str (lines 29-30) matches only there. Line 29 is added only" in observation.text
