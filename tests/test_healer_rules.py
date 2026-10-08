import pytest

from app.agents.HealerAgent.HealerGuards import (
    check_build_file,
    check_new_file,
    check_only_additions,
    check_test_integrity,
    is_test_file,
)

JAVA_BEFORE = """public class LoginTest {
    @Test
    public void validLogin() {
        login.as("user");
        Assert.assertEquals(page.title(), "Dashboard");
    }
}"""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/test/java/com/demo/LoginTest.java", True),
        ("src/test/java/com/demo/steps/LoginSteps.java", True),
        ("tests/test_search.py", True),
        ("e2e/login.spec.ts", True),
        ("src/test/resources/features/login.feature", True),
        ("pom.xml", False),
        ("src/main/java/com/demo/App.java", False),
        ("requirements.txt", False),
    ],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected


def test_removing_an_assertion_is_rejected() -> None:
    after = JAVA_BEFORE.replace('        Assert.assertEquals(page.title(), "Dashboard");\n', "")

    violations = check_test_integrity("src/test/java/LoginTest.java", JAVA_BEFORE, after)

    assert [v.rule for v in violations] == ["assertion_removed"]


def test_commenting_an_assertion_is_rejected() -> None:
    after = JAVA_BEFORE.replace('Assert.assertEquals', '// Assert.assertEquals')

    rules = {v.rule for v in check_test_integrity("src/test/java/LoginTest.java", JAVA_BEFORE, after)}

    assert "assertion_commented" in rules


@pytest.mark.parametrize(
    ("path", "before", "after"),
    [
        ("src/test/java/LoginTest.java", JAVA_BEFORE, JAVA_BEFORE.replace("@Test", "@Test(enabled = false)")),
        ("tests/test_x.py", "def test_a():\n    assert f() == 1\n", "@pytest.mark.skip\ndef test_a():\n    assert f() == 1\n"),
        ("e2e/a.spec.ts", "it('works', () => { expect(x).toBe(1) })", "it.skip('works', () => { expect(x).toBe(1) })"),
    ],
)
def test_skipping_tests_is_rejected(path: str, before: str, after: str) -> None:
    assert "test_skipped" in {v.rule for v in check_test_integrity(path, before, after)}


def test_deleting_a_test_is_rejected() -> None:
    before = "def test_a():\n    run()\n\ndef test_b():\n    run()\n"
    after = "def test_a():\n    run()\n"

    assert [v.rule for v in check_test_integrity("tests/test_x.py", before, after)] == ["test_removed"]


def test_swallowing_failures_is_rejected() -> None:
    before = "def test_a():\n    assert f()\n"
    after = "def test_a():\n    try:\n        assert f()\n    except Exception: pass\n"

    assert "failure_swallowed" in {v.rule for v in check_test_integrity("tests/test_x.py", before, after)}


def test_real_fixes_are_allowed() -> None:
    fixed_import = "import org.testng.Assert;\n" + JAVA_BEFORE
    moved = JAVA_BEFORE.replace("        login.as(\"user\");\n", "") + '\n// helper\n        login.as("user");'

    assert check_test_integrity("src/test/java/LoginTest.java", JAVA_BEFORE, fixed_import) == []
    assert check_test_integrity("src/test/java/LoginTest.java", JAVA_BEFORE, "    " + JAVA_BEFORE) == []
    assert check_test_integrity("src/test/java/LoginTest.java", JAVA_BEFORE, moved) == []


def test_non_test_files_are_not_checked() -> None:
    assert check_test_integrity("pom.xml", "<assert>", "") == []


POM = "<project>\n  <dependencies>\n    <dependency><version>99.0.0</version></dependency>\n  </dependencies>\n</project>\n"


def test_build_file_with_a_valid_fix_is_allowed() -> None:
    assert check_build_file("pom.xml", POM, POM.replace("99.0.0", "7.10.2")) == []


def test_xml_mangled_by_sed_is_caught() -> None:
    # GNU sed reads "<" in a replacement as "uppercase the next character": "<" became "003c".
    mangled = POM.replace("<dependency><version>99.0.0</version></dependency>", "003cdependency003e003cversion003e7.10.2")
    mangled = mangled.replace("</dependencies>", "")

    [violation] = check_build_file("pom.xml", POM, mangled)

    assert violation.rule == "build_file_broken"
    assert violation.file == "pom.xml"


def test_emptied_build_file_is_caught() -> None:
    [violation] = check_build_file("module/pom.xml", POM, "")

    assert violation.line == "(file emptied)"


def test_broken_package_json_is_caught() -> None:
    before = '{\n  "devDependencies": {"@playwright/test": "^9.9.9"}\n}\n'
    after = '{\n  "devDependencies": {"@playwright/test": "^1.48.0",}\n'

    assert [v.rule for v in check_build_file("package.json", before, after)] == ["build_file_broken"]


def test_file_that_was_already_broken_is_not_blamed_on_the_healer() -> None:
    assert check_build_file("pom.xml", "<project>", "<project><x>") == []


def test_xml_with_a_doctype_is_not_parsed() -> None:
    before = '<?xml version="1.0"?>\n<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.0.dtd">\n<suite name="a"/>\n'

    assert check_build_file("testng.xml", before, before.replace('name="a"/>', 'name="a">')) == []


def test_other_files_are_not_parsed() -> None:
    assert check_build_file("build.gradle", "plugins { id 'java' }", "plugins { id 'java'") == []


@pytest.mark.parametrize(
    ("path", "stray"),
    [
        ("pom.xml.backup", True),
        ("pom.xml.bak", True),
        ("src/Login.java.orig", True),
        ("src/Login.java~", True),
        ("src/test/java/pages/LoginPage.java", False),
        ("requirements.txt", False),
    ],
)
def test_stray_files(path: str, stray: bool) -> None:
    assert bool(check_new_file(path)) is stray



def test_only_additions_allows_new_lines_and_moves() -> None:
    before = "import a;\nimport b;\nclass X {\n}\n"
    after = "import b;\nimport a;\nimport c;\nclass X {\n  void y() {}\n}\n"

    assert check_only_additions("X.java", before, after) == []


def test_only_additions_catches_a_changed_line() -> None:
    [violation] = check_only_additions("LoginPage.java", 'By.id("username");\n', 'By.id("user-name");\n')

    assert (violation.rule, violation.line) == ("existing_code_changed", 'By.id("username");')
