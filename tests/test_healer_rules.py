from pathlib import Path

import pytest

from app.agents.HealerAgent.HealerGuards import (
    check_build_file,
    check_changes,
    check_new_file,
    check_test_integrity,
    is_test_file,
)
from app.utils.ProjectSnapshot import ProjectSnapshot

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


@pytest.mark.parametrize(("path", "comment"), [
    ("src/test/java/LoginTest.java", "// assertThat(cart.count()).isEqualTo(2);"),
    ("src/test/java/LoginTest.java", "/* Assert.assertTrue(page.isOpen()); */"),
    ("tests/test_login.py", "# assert page.title() == 'Dashboard'"),
    ("tests/test_login.py", "# assert response.ok"),
    ("tests/test_login.py", "#self.assertEqual(title, 'Dashboard')"),
    ("e2e/login.spec.ts", "// await expect(page).toHaveTitle('Dashboard');"),
    ("cypress/e2e/login.cy.js", "// cy.get('.title').should('be.visible')"),
    ("src/test/resources/features/login.feature", "# Then the dashboard is shown"),
])
def test_assertion_code_in_a_comment_is_reported(path: str, comment: str) -> None:
    before = "x = 1\n"

    assert [v.rule for v in check_test_integrity(path, before, before + comment + "\n")] == ["assertion_commented"]


@pytest.mark.parametrize(("path", "comment"), [
    ("src/test/java/InventoryTest.java", "// Get list of product names and assert first"),  # a real false alarm
    ("src/test/java/InventoryTest.java", "// Then we verify the cart total"),
    ("tests/test_cart.py", "# should show the cart, assert after the reload"),
    ("tests/test_cart.py", "# And now the user logs out"),
    ("e2e/cart.spec.ts", "// expect the badge to update before we assert it"),
])
def test_a_comment_that_only_talks_about_a_check_is_not_reported(path: str, comment: str) -> None:
    before = "x = 1\n"

    assert check_test_integrity(path, before, before + comment + "\n") == []


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


REPOSITORIES = """  <repositories>
    <repository><id>jitpack.io</id><url>https://jitpack.io</url></repository>
    <repository><id>zephyr</id><url>https://repo.example/zephyr</url></repository>
  </repositories>
"""
SELENIUM = "    <dependency><groupId>org.seleniumhq.selenium</groupId><artifactId>selenium-java</artifactId></dependency>\n"
ZEPHYR = "    <dependency><groupId>com.thed.zephyr</groupId><artifactId>zfj-cloud-rest-client</artifactId></dependency>\n"
MAVEN_POM = f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
{REPOSITORIES}  <dependencies>
    <dependency><groupId>io.cucumber</groupId><artifactId>cucumber-java</artifactId><version>7.18.0</version></dependency>
{SELENIUM}{ZEPHYR}  </dependencies>
  <build>
    <plugins>
      <plugin><artifactId>maven-surefire-plugin</artifactId><version>3.2.5</version></plugin>
    </plugins>
  </build>
</project>
"""


def test_a_pom_that_lost_its_repositories_and_dependencies_is_reported(tmp_path: Path) -> None:
    # A real run: on a project that did not build, the generator ran `sed -i` on pom.xml: 590 lines became 258, its
    # repositories and dependencies gone. The file still parsed, so nothing was reported.
    pom = tmp_path / "pom.xml"
    pom.write_text(MAVEN_POM, encoding="utf-8")
    snapshot = ProjectSnapshot.capture(tmp_path)
    pom.write_text(MAVEN_POM.replace(REPOSITORIES, "").replace(SELENIUM + ZEPHYR, ""), encoding="utf-8")

    [violation] = check_changes(snapshot, snapshot.changes())

    assert (violation.file, violation.rule) == ("pom.xml", "build_file_reduced")
    assert violation.line == (
        "removed from dependencies (2), repositories (2): com.thed.zephyr:zfj-cloud-rest-client, "
        "org.seleniumhq.selenium:selenium-java, jitpack.io, zephyr"
    )
    assert pom.read_text(encoding="utf-8") != MAVEN_POM  # only reported: never rolled back


def test_a_pom_that_gained_a_dependency_or_changed_a_version_is_not_reported() -> None:
    testng = "    <dependency><groupId>org.testng</groupId><artifactId>testng</artifactId></dependency>\n"
    grown = MAVEN_POM.replace(SELENIUM, SELENIUM + testng).replace("7.18.0", "7.20.1")
    # The same dependency moved into dependencyManagement is not lost either.
    moved = MAVEN_POM.replace(ZEPHYR, "").replace(
        "  <build>", f"  <dependencyManagement>\n  <dependencies>\n{ZEPHYR}  </dependencies>\n  </dependencyManagement>\n  <build>"
    )

    assert check_build_file("pom.xml", MAVEN_POM, grown) == []
    assert check_build_file("pom.xml", MAVEN_POM, moved) == []


def test_a_package_json_that_lost_dependencies_is_reported() -> None:
    before = '{"dependencies": {"dotenv": "^16.4.5"}, "devDependencies": {"@playwright/test": "^1.48.0", "typescript": "^5.6.0"}}'
    # typescript removed; @playwright/test upgraded and dotenv moved to devDependencies are kept.
    after = '{"devDependencies": {"@playwright/test": "^1.49.0", "dotenv": "^16.4.5"}}'

    assert [(v.rule, v.line) for v in check_build_file("package.json", before, after)] == [
        ("build_file_reduced", "removed from devDependencies (1): typescript")
    ]
    assert check_build_file("package.json", after, before) == []


@pytest.mark.parametrize(("path", "reported"), [
    ("requirements.txt", True),
    ("requirements-dev.txt", True),
    ("module/build.gradle", True),
    ("build.gradle.kts", True),
    ("pyproject.toml", True),
    ("Tests.csproj", True),
    ("docs/notes.txt", False),  # not a build or dependency file
    ("testng.xml", False),
])
def test_another_build_file_that_lost_many_of_its_lines_is_reported(path: str, reported: bool) -> None:
    before = "".join(f"package{number}==1.{number}\n" for number in range(40))
    cut = before[: before.index("package10==")]

    expected = [("build_file_reduced", "shrank from 40 to 10 lines")] if reported else []
    assert [(v.rule, v.line) for v in check_build_file(path, before, cut)] == expected


def test_small_edits_and_additions_to_a_build_file_are_not_reported() -> None:
    before = "".join(f"package{number}==1.{number}\n" for number in range(40))

    assert check_build_file("requirements.txt", before, before.replace("package3==1.3\n", "")) == []
    assert check_build_file("requirements.txt", before, before.replace("==1.", "==2.")) == []  # every version changed
    assert check_build_file("requirements.txt", before, before + "selenium==4.25.0\n") == []


def test_a_pom_that_no_longer_parses_is_reported_by_its_own_rule() -> None:
    cut = MAVEN_POM.replace(REPOSITORIES, "").replace("<dependencies>", "003cdependencies003e", 1)

    assert [v.rule for v in check_build_file("pom.xml", MAVEN_POM, cut)] == ["build_file_broken"]


def _java(package: str, name: str, *imports: str) -> str:
    lines = [f"package {package};", "", *(f"import {target};" for target in imports), "", f"public class {name} {{}}"]
    return "\n".join(lines) + "\n"


UTILITIES = {
    "src/test/java/utilities/KeywordUtil.java": _java("utilities", "KeywordUtil"),
    "src/test/java/utilities/CommonPage.java": _java("utilities", "CommonPage"),
}


def _write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")


def _missing_imports(root: Path, project: dict[str, str], written: dict[str, str]) -> list[tuple[str, str]]:
    """(file, line) of every import reported after an agent wrote `written` into a project of `project`."""
    _write(root, project)
    snapshot = ProjectSnapshot.capture(root)
    _write(root, written)
    return [(v.file, v.line) for v in check_changes(snapshot, snapshot.changes()) if v.rule == "import_not_found"]


def test_an_import_of_a_project_class_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    # A real run: the project did not build, so the build check was off, and 3 generated tests imported
    # utilities.BrowserUtil, which the project never had (it has utilities.KeywordUtil and utilities.CommonPage).
    steps = "src/test/java/ngauto/LoginSteps.java"
    imports = (
        "utilities.BrowserUtil", "utilities.KeywordUtil", "utilities.CommonPage", "cucumber.api.java.en.Given",
        "org.openqa.selenium.By", "io.restassured.*", "static io.restassured.RestAssured.given",
    )

    assert _missing_imports(tmp_path, UTILITIES, {steps: _java("ngauto", "LoginSteps", *imports)}) == [
        (steps, "import utilities.BrowserUtil;")
    ]


def test_a_class_in_a_file_that_starts_with_a_bom_is_found(tmp_path: Path) -> None:
    # Editors on Windows often save Java files with a UTF-8 BOM before `package`.
    project = {**UTILITIES, "src/test/java/utilities/KeywordUtil.java": "\ufeff" + _java("utilities", "KeywordUtil")}
    steps = "src/test/java/ngauto/LoginSteps.java"

    assert _missing_imports(tmp_path, project, {steps: _java("ngauto", "LoginSteps", "utilities.KeywordUtil")}) == []


def test_a_class_the_agent_wrote_in_another_new_file_is_found(tmp_path: Path) -> None:
    written = {
        "src/test/java/utilities/WaitUtil.java": _java("utilities", "WaitUtil"),
        "src/test/java/ngauto/Helpers.kt": "package ngauto\n\nclass LoginHelper\n\nenum class Role { ADMIN }\n",
        "src/test/java/ngauto/CartTest.java": _java(
            "ngauto", "CartTest", "utilities.WaitUtil", "ngauto.LoginHelper", "ngauto.Role", "static ngauto.Role.ADMIN"
        ),
    }

    assert _missing_imports(tmp_path, UTILITIES, written) == []


@pytest.mark.parametrize(("target", "reported"), [
    ("static utilities.KeywordUtil.click", False),
    ("static utilities.KeywordUtil.*", False),
    ("static utilities.BrowserUtil.open", True),
    ("utilities.*", False),
    ("utilities.KeywordUtil.Keys", False),  # a nested class: its class exists
    ("utilities.helpers.*", True),  # below a package of the project: that package must exist
    ("utilities.helpers.WaitHelper", True),
    ("org.testng.annotations.*", False),
    ("static org.testng.Assert.assertEquals", False),
])
def test_static_and_wildcard_imports(tmp_path: Path, target: str, reported: bool) -> None:
    test = "src/test/java/ngauto/CartTest.java"

    expected = [(test, f"import {target};")] if reported else []
    assert _missing_imports(tmp_path, UTILITIES, {test: _java("ngauto", "CartTest", target)}) == expected


def test_only_the_imports_an_agent_added_are_checked(tmp_path: Path) -> None:
    page = "src/test/java/pages/LoginPage.java"
    project = {**UTILITIES, page: _java("pages", "LoginPage", "utilities.OldHelper")}  # broken before the agent came
    changed = _java("pages", "LoginPage", "utilities.OldHelper", "utilities.KeywordUtil", "utilities.BrowserUtil")

    assert _missing_imports(tmp_path, project, {page: changed}) == [(page, "import utilities.BrowserUtil;")]


PLAYWRIGHT_PROJECT = {
    "fixtures/test.ts": "export const test = base.extend({});\n",
    "tests/pages/LoginPage.ts": "export class LoginPage {}\n",
    "tests/helpers/index.ts": "export async function login() {}\n",
    "tests/data/users.ts": "export const users = [];\n",
    "tests/login.spec.ts": "import { test } from '../fixtures/test';\n",
}


@pytest.mark.parametrize(("line", "reported"), [
    ("import { test } from '../../fixtures/test';", False),
    ("import { LoginPage } from '../pages/LoginPage';", False),
    ("import { login } from '../helpers';", False),  # a folder's index file
    ("import { users } from '../data/users.js';", False),  # TypeScript's ESM imports name the .js it compiles to
    ("import { test, expect } from '@playwright/test';", False),  # a library
    ("import { shared } from '../../../shared/steps';", False),  # out of the project: not checked
    ("// import { OldPage } from '../pages/OldPage';", False),
    # A real run: the import copied as it is from tests/login.spec.ts, one folder up.
    ("import { test } from '../fixtures/test';", True),
    ("const { CartPage } = require('../pages/CartPage');", True),
])
def test_a_relative_js_import_that_leads_to_no_file_is_reported(tmp_path: Path, line: str, reported: bool) -> None:
    spec = "tests/ngauto/td-001-valid-login.spec.ts"

    expected = [(spec, line)] if reported else []
    assert _missing_imports(tmp_path, PLAYWRIGHT_PROJECT, {spec: line + "\n"}) == expected


@pytest.mark.parametrize(("line", "reported"), [
    ("from ..pages.login_page import LoginPage", False),
    ("from ..pages import login_page", False),
    ("from . import helpers", False),
    ("from pages.cart_page import CartPage", False),  # not relative: it may be a library's
    ("from ..pages.cart_page import CartPage", True),
    ("from .pages.login_page import LoginPage", True),  # one dot: tests/ngauto/pages, which does not exist
])
def test_a_relative_python_import_that_leads_to_no_module_is_reported(tmp_path: Path, line: str, reported: bool) -> None:
    project = {"tests/pages/__init__.py": "", "tests/pages/login_page.py": "class LoginPage:\n    pass\n"}
    test = "tests/ngauto/test_td_001.py"

    expected = [(test, line)] if reported else []
    assert _missing_imports(tmp_path, project, {test: line + "\n"}) == expected


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


@pytest.mark.parametrize(("path", "scratch"), [
    ("result.txt", True),  # a real run: `mvn ... | tee result.txt`, left in the project root
    ("results.txt", True),
    ("test-output.txt", True),
    ("build.log", True),
    ("nohup.out", True),
    ("requirements.txt", False),  # a new dependency list is part of the work
    ("about.txt", False),
    ("docs/result.txt", False),  # only the project root: a file in a folder was put there on purpose
    ("src/test/resources/data/output.txt", False),
    ("logging.properties", False),
])
def test_command_output_saved_in_the_project_root_is_reported(path: str, scratch: bool) -> None:
    assert [v.rule for v in check_new_file(path)] == (["scratch_file"] if scratch else [])


@pytest.mark.parametrize(("path", "binary"), [
    ("lib/zfj-cloud-rest-client-1.0.jar", True),  # a real run: an empty jar left in lib/ after a failed download
    ("drivers/chromedriver.exe", True),
    ("vendor/native.so", True),
    ("dist/package.tar.gz", True),
    ("src/test/java/LoginTest.java", False),
    ("src/test/resources/data.json", False),
])
def test_a_new_archive_or_compiled_file_is_reported(path: str, binary: bool) -> None:
    assert [v.rule for v in check_new_file(path)] == (["binary_or_large_file"] if binary else [])


def test_a_binary_file_an_agent_adds_is_reported_though_it_is_not_among_the_changes(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "old.dat").write_bytes(b"\x00old")  # the project's own: not reported
    snapshot = ProjectSnapshot.capture(tmp_path)
    (tmp_path / "lib" / "downloaded.dat").write_bytes(b"\x00\x01new")

    changes = snapshot.changes()

    assert changes == []  # binaries are never diffed
    assert [(v.rule, v.file) for v in check_changes(snapshot, changes)] == [("binary_or_large_file", "lib/downloaded.dat")]


def test_a_log_that_the_projects_own_files_name_is_not_reported() -> None:
    # The project's logging writes test.log whenever its tests run, also when an agent runs them: not the agent's.
    project_files = {"src/test/resources/log4j2.xml": '<File name="file" fileName="test.log"/>'}

    assert check_new_file("test.log", project_files) == []
    assert [v.rule for v in check_new_file("result.txt", project_files)] == ["scratch_file"]
