"""Project test briefing: the project's own tests and the shared code they use, found from the project's own
references wherever that code lives (plain code, no LLM)."""

import importlib
import os
import posixpath
import re
import sys
from pathlib import Path

import pytest

from app.agents.TestGeneratorAgent.ProjectTestBriefing import MAX_FILES, MAX_TOTAL_CHARS, project_test_briefing

briefing_module = importlib.import_module("app.agents.TestGeneratorAgent.ProjectTestBriefing")

GENERATED = "ngauto"
EARLIER_RUN = "WRITTEN-BY-AN-EARLIER-RUN"
OUTSIDE = "OUTSIDE-THE-PROJECT"

PLAYWRIGHT_TS = {
    "package.json": '{"devDependencies": {"@playwright/test": "1.58.0"}}\n',
    "playwright.config.ts": "import { defineConfig } from '@playwright/test';\n\nexport default defineConfig({ testDir: './tests' });\n",
    # The smallest test uses no project code: the one that does is the better example.
    "tests/example.spec.ts": "import { test } from '@playwright/test';\n\ntest('opens', async ({ page }) => {});\n",
    "tests/smoke.spec.ts": (
        "import { test, expect } from '../fixtures/test';\n\n"
        "test('login works', async ({ loginPage }) => {\n"
        "  await loginPage.login('admin', 'secret');\n"
        "  await expect(loginPage.heading).toBeVisible();\n"
        "});\n"
    ),
    "fixtures/test.ts": (
        "import { test as base } from '@playwright/test';\n"
        "import { LoginPage } from '../pages/LoginPage';\n\n"
        "export const test = base.extend<{ loginPage: LoginPage }>({\n"
        "  loginPage: async ({ page }, use) => use(new LoginPage(page)),\n"
        "});\n"
        "export { expect } from '@playwright/test';\n"
    ),
    "pages/LoginPage.ts": (
        "import { Page } from '@playwright/test';\n\n"
        "export class LoginPage {\n"
        "  readonly heading = this.page.getByRole('heading');\n"
        "  constructor(private readonly page: Page) {}\n"
        "  async login(user: string, password: string) {\n"
        "    await this.page.fill('#username', user);\n"
        "    await this.page.fill('#password', password);\n"
        "  }\n"
        "}\n"
    ),
    "tests/ngauto/old.spec.ts": f"import {{ test }} from '../../fixtures/test';\n// {EARLIER_RUN}\n",
}
PYTEST = {
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
    "conftest.py": (
        "import pytest\nfrom selenium import webdriver\n\n\n"
        "@pytest.fixture\ndef driver():\n    browser = webdriver.Chrome()\n    yield browser\n    browser.quit()\n"
    ),
    "pages/__init__.py": "",
    "pages/base_page.py": (
        "class BasePage:\n    def __init__(self, driver):\n        self.driver = driver\n\n"
        "    def type_into(self, locator, text):\n        self.driver.find_element(*locator).send_keys(text)\n"
    ),
    "pages/login_page.py": (
        "from selenium.webdriver.common.by import By\n\nfrom .base_page import BasePage\n\n\n"
        'class LoginPage(BasePage):\n    USERNAME = (By.ID, "username")\n\n'
        "    def log_in(self, user):\n        self.type_into(self.USERNAME, user)\n"
    ),
    "tests/__init__.py": "",
    "tests/test_login.py": (
        "from pages.login_page import LoginPage\n\n\n"
        'def test_valid_login(driver):\n    LoginPage(driver).log_in("admin")\n'
    ),
    "tests/ngauto/test_old.py": f"# {EARLIER_RUN}\n",
}
MAVEN = {
    "pom.xml": "<project/>\n",
    "src/main/java/com/acme/pages/LoginPage.java": (
        "package com.acme.pages;\n\nimport org.openqa.selenium.By;\nimport org.openqa.selenium.WebDriver;\n\n"
        "public class LoginPage {\n    private final WebDriver driver;\n\n"
        "    public LoginPage(WebDriver driver) {\n        this.driver = driver;\n    }\n\n"
        '    public void logIn(String user) {\n        driver.findElement(By.id("username")).sendKeys(user);\n    }\n}\n'
    ),
    "src/main/java/com/acme/utils/Waits.java": (
        "package com.acme.utils;\n\npublic final class Waits {\n    public static void waitForPage() {\n    }\n}\n"
    ),
    "src/test/java/com/acme/tests/BaseTest.java": (
        "package com.acme.tests;\n\nimport org.openqa.selenium.WebDriver;\n\n"
        "public class BaseTest {\n    protected WebDriver driver;\n}\n"
    ),
    "src/test/java/com/acme/tests/LoginTest.java": (
        "package com.acme.tests;\n\nimport com.acme.pages.LoginPage;\nimport org.testng.annotations.Test;\n\n"
        "import static com.acme.utils.Waits.waitForPage;\n\n"
        "public class LoginTest extends BaseTest {\n    @Test\n    public void validLogin() {\n"
        '        new LoginPage(driver).logIn("admin");\n        waitForPage();\n    }\n}\n'
    ),
    "src/test/java/com/acme/tests/ngauto/OldTest.java": f"package com.acme.tests.ngauto;\n\n// {EARLIER_RUN}\nclass OldTest {{}}\n",
}
ROBOT = {
    "tests/login.robot": (
        "*** Settings ***\nResource    ../resources/common.resource\n\n"
        "*** Test Cases ***\nValid Login\n    Open Login Page\n    Log In As    admin\n"
    ),
    "resources/common.resource": (
        "*** Settings ***\nLibrary    SeleniumLibrary\nVariables    ../config/env.py\n\n"
        "*** Keywords ***\nOpen Login Page\n    Open Browser    ${URL}    chrome\n\n"
        "Log In As\n    [Arguments]    ${user}\n    Input Text    id=username    ${user}\n"
    ),
    "config/env.py": 'URL = "https://shop.example/login"\n',
    "tests/ngauto/old.robot": f"*** Test Cases ***\nOld\n    Log    {EARLIER_RUN}\n",
}
# Every test is named after the page it tests, so every test name holds a helper word ('page').
TESTS_NAMED_AFTER_PAGES = {
    "src/main/java/com/qa/base/TestBase.java": (
        "package com.qa.base;\n\nimport org.openqa.selenium.WebDriver;\n\n"
        "public class TestBase {\n    protected static WebDriver driver;\n}\n"
    ),
    "src/main/java/com/qa/pages/LoginPage.java": (
        "package com.qa.pages;\n\nimport com.qa.base.TestBase;\n\n"
        "public class LoginPage extends TestBase {\n    public void login(String user) {\n    }\n}\n"
    ),
    "src/test/java/com/qa/testcases/LoginPageTest.java": (
        "package com.qa.testcases;\n\nimport com.qa.base.TestBase;\nimport com.qa.pages.LoginPage;\n\n"
        "public class LoginPageTest extends TestBase {\n    @Test\n    public void loginTest() {\n"
        '        new LoginPage().login("admin");\n    }\n}\n'
    ),
}
# The support files are smaller than the spec and use project code too: the spec is still the example.
CYPRESS = {
    "cypress.config.ts": "import { defineConfig } from 'cypress';\n\nexport default defineConfig({ e2e: {} });\n",
    "cypress/support/e2e.ts": "import './commands';\n",
    "cypress/support/commands.ts": (
        "import { LoginPage } from '../pages/LoginPage';\n\nCypress.Commands.add('logIn', () => new LoginPage().open());\n"
    ),
    "cypress/pages/LoginPage.ts": "export class LoginPage {\n  open() {\n    cy.visit('/login');\n  }\n}\n",
    "cypress/e2e/login.cy.ts": (
        "import { LoginPage } from '../pages/LoginPage';\n\n"
        "describe('login', () => {\n  it('opens the login page', () => {\n    new LoginPage().open();\n  });\n});\n"
    ),
}


def _project(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _shown(briefing: str) -> list[str]:
    """The files in the briefing, in the order shown."""
    return re.findall(r"^--- (.+) ---$", briefing, re.MULTILINE)


def _checked(briefing: str) -> list[str]:
    """The files in the briefing, in the order shown, once the rules every briefing keeps are checked: within the
    character limit, each file once, and never a file of the generated tests' folder."""
    shown = _shown(briefing)
    assert len(briefing) <= MAX_TOTAL_CHARS
    assert len(shown) == len(set(shown))
    assert EARLIER_RUN not in briefing
    assert not [path for path in shown if GENERATED in path.split("/")]
    return shown


def _link(link: Path, target: Path) -> None:
    """A symbolic link; on Windows without the rights for one, a junction (a folder link anyone may create)."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except OSError:
        if sys.platform != "win32" or not target.is_dir():
            pytest.skip("creating links needs extra rights on this machine")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_playwright_fixtures_and_page_objects_outside_the_tests_folder_are_shown(tmp_path: Path) -> None:
    briefing = project_test_briefing(_project(tmp_path, PLAYWRIGHT_TS), ["tests"], GENERATED)

    shown = _checked(briefing)
    # The fixture the test imports, the page object the fixture imports, then the test itself.
    assert shown[:3] == ["fixtures/test.ts", "pages/LoginPage.ts", "tests/smoke.spec.ts"]
    assert "async login(user: string, password: string)" in briefing
    assert "playwright.config.ts" in shown  # a config file at the project root


@pytest.mark.parametrize("test_dir", ["tests", "tests\\", ".\\tests"])
def test_pytest_page_objects_and_the_root_conftest_are_shown(tmp_path: Path, test_dir: str) -> None:
    briefing = project_test_briefing(_project(tmp_path, PYTEST), [test_dir], GENERATED)

    shown = _checked(briefing)
    # The page object the test imports, its base class ('from .base_page import BasePage'), then the test.
    assert shown[:3] == ["pages/login_page.py", "pages/base_page.py", "tests/test_login.py"]
    assert "conftest.py" in shown  # the project's fixtures, at its root
    assert "def type_into(self, locator, text)" in briefing


def test_maven_page_objects_in_the_main_sources_are_shown(tmp_path: Path) -> None:
    briefing = project_test_briefing(_project(tmp_path, MAVEN), ["src/test"], GENERATED)

    shown = _checked(briefing)
    # 'import com.acme.pages.LoginPage;' and 'import static com.acme.utils.Waits.waitForPage;', then the test.
    assert shown[:3] == [
        "src/main/java/com/acme/pages/LoginPage.java",
        "src/main/java/com/acme/utils/Waits.java",
        "src/test/java/com/acme/tests/LoginTest.java",
    ]
    assert "src/test/java/com/acme/tests/BaseTest.java" in shown  # the test folder's base class, as before


def test_robot_keyword_resources_are_shown(tmp_path: Path) -> None:
    briefing = project_test_briefing(_project(tmp_path, ROBOT), ["tests"], GENERATED)

    # The resource file the suite loads, the variables file that resource loads, then the suite.
    assert _checked(briefing) == ["resources/common.resource", "config/env.py", "tests/login.robot"]
    assert "Log In As\n    [Arguments]    ${user}" in briefing


def test_tests_named_after_a_page_are_still_examples(tmp_path: Path) -> None:
    briefing = project_test_briefing(_project(tmp_path, TESTS_NAMED_AFTER_PAGES), ["src/test"], GENERATED)

    assert _checked(briefing) == [
        "src/main/java/com/qa/base/TestBase.java",
        "src/main/java/com/qa/pages/LoginPage.java",
        "src/test/java/com/qa/testcases/LoginPageTest.java",
    ]


def test_a_support_file_in_the_test_folder_is_not_taken_for_the_example(tmp_path: Path) -> None:
    briefing = project_test_briefing(_project(tmp_path, CYPRESS), ["cypress"], GENERATED)

    assert _checked(briefing)[:2] == ["cypress/pages/LoginPage.ts", "cypress/e2e/login.cy.ts"]


@pytest.mark.parametrize(
    ("test_file", "line", "shared"),
    [
        ("tests/test_cart.py", "from shop.pages import cart_page", "shop/pages/cart_page.py"),
        ("tests/test_cart.py", "from shop.pages.cart_page import CartPage", "shop/pages/cart_page.py"),
        ("tests/test_cart.py", "from ..shop.pages.cart_page import CartPage", "shop/pages/cart_page.py"),
        ("src/test/java/shop/CartTest.java", "import shop.pages.CartPage;", "src/main/java/shop/pages/CartPage.java"),
        ("Shop.Tests/CartTests.cs", "using static Shop.Helpers.Waits;", "Shop/Helpers/Waits.cs"),
        ("spec/cart_spec.rb", "require 'shop/pages/cart_page'", "lib/shop/pages/cart_page.rb"),
        ("tests/CartTest.php", "use Shop\\Pages\\CartPage;", "src/Shop/Pages/CartPage.php"),
        ("tests/cart.spec.ts", "import { CartPage } from '@pages/CartPage';", "src/pages/CartPage.ts"),
        ("tests/cart.spec.ts", "import { logIn } from '../helpers/session.js';", "helpers/session.ts"),
        ("tests/cart.spec.ts", "import { users } from '../data';", "data/index.ts"),
        # A package or a namespace: the files in it the test names.
        ("Shop.Tests/CartTests.cs", "using Shop.Pages;\n\nnew CartPage(driver).Open();", "Shop/Pages/CartPage.cs"),
        (
            "Shop.Tests/Tests/CartTests.cs",
            "using Shop.Tests.Pages;\n\nnew CartPage(driver).Open();",
            "Shop.Tests/Pages/CartPage.cs",
        ),
        (
            "src/test/java/shop/CartTest.java",
            "import shop.pages.*;\n\nnew CartPage(driver).open();",
            "src/main/java/shop/pages/CartPage.java",
        ),
        ("tests/CartTest.php", "require_once 'pages/CartPage.php';", "pages/CartPage.php"),
        ("spec/cart_spec.rb", "require_relative 'support/cart_page'", "spec/support/cart_page.rb"),
        ("tests/cart.spec.ts", "import { CartPage } from '@pages/CartPage.js';", "src/pages/CartPage.ts"),
        # A byte order mark before the first line, and Windows separators.
        ("tests/test_cart.py", "﻿from shop.pages.cart_page import CartPage", "shop/pages/cart_page.py"),
        ("tests/cart.robot", "*** Settings ***\nResource    ..\\resources\\cart.resource", "resources/cart.resource"),
    ],
)
def test_module_references_are_followed_wherever_the_module_lives(
    tmp_path: Path, test_file: str, line: str, shared: str
) -> None:
    project = _project(tmp_path, {test_file: f"{line}\n\n// the test\n", shared: "// shared code\n"})

    briefing = project_test_briefing(project, [posixpath.dirname(test_file)], GENERATED)

    assert _checked(briefing) == [shared, test_file]


def test_a_namespace_import_shows_only_the_classes_the_test_names(tmp_path: Path) -> None:
    project = _project(tmp_path, {
        "Shop.Tests/CartTests.cs": (
            "using NUnit.Framework;\nusing OpenQA.Selenium;\nusing Shop.Pages;\n\n"
            "public class CartTests : BaseTest\n{\n    [Test]\n"
            "    public void AddsToCart() => new CartPage(Driver).Add();\n}\n"
        ),
        "Shop/Pages/CartPage.cs": "namespace Shop.Pages;\n\npublic class CartPage { }\n",
        "Shop/Pages/AdminPage.cs": "namespace Shop.Pages;\n\npublic class AdminPage { }\n",
    })

    briefing = project_test_briefing(project, ["Shop.Tests"], GENERATED)

    assert _checked(briefing) == ["Shop/Pages/CartPage.cs", "Shop.Tests/CartTests.cs"]


def test_the_briefing_stays_within_its_limits_and_keeps_an_example_test(tmp_path: Path) -> None:
    pages = {f"pages/Page{n:02d}.ts": f"export class Page{n:02d} {{\n{'  // a long page object\n' * 300}}}\n" for n in range(12)}
    imports = "".join(f"import {{ Page{n:02d} }} from '../pages/Page{n:02d}';\n" for n in range(12))
    project = _project(tmp_path, {**pages, "tests/big.spec.ts": f"{imports}\ntest('big', async () => {{}});\n"})

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    shown = _checked(briefing)
    assert shown[0] == "pages/Page00.ts"  # the shared code still comes first
    assert shown[-1] == "tests/big.spec.ts"  # and never pushes the example test out
    assert "... (truncated)" in briefing


def test_the_briefing_shows_at_most_a_handful_of_files(tmp_path: Path) -> None:
    pages = {f"pages/Page{n:02d}.ts": f"export class Page{n:02d} {{}}\n" for n in range(15)}
    imports = "".join(f"import {{ Page{n:02d} }} from '../pages/Page{n:02d}';\n" for n in range(15))
    project = _project(tmp_path, {**pages, "tests/many.spec.ts": imports})

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    shown = _checked(briefing)
    assert len(shown) == MAX_FILES
    assert "tests/many.spec.ts" in shown


def test_the_analyzers_shared_paths_are_shown(tmp_path: Path) -> None:
    # A feature file names no code: the step definitions and the world are found from the analyzer's paths.
    project = _project(tmp_path / "project", {
        "features/login.feature": "Feature: Login\n  Scenario: Valid login\n    Given the login page is open\n",
        "steps/login_steps.js": "Given('the login page is open', async function () {});\n",
        "support/world.js": "class World {}\nmodule.exports = { World };\n",
        "support/ngauto/old_steps.js": f"// {EARLIER_RUN}\n",
    })
    (tmp_path / "secret.js").write_text(f"// {OUTSIDE}\n", encoding="utf-8")
    shared_paths = ["steps\\", "./support/world.js:2", "support/ngauto", "../secret.js", str(tmp_path / "secret.js"), "."]

    briefing = project_test_briefing(project, ["features"], GENERATED, shared_paths)

    assert _checked(briefing) == ["features/login.feature", "steps/login_steps.js", "support/world.js"]
    assert OUTSIDE not in briefing


def test_a_shared_folder_shows_its_helpers_first_then_its_smallest_files(tmp_path: Path) -> None:
    project = _project(tmp_path, {
        "tests/test_cart.py": "def test_cart():\n    pass\n",
        "lib/__init__.py": "",  # empty: it never takes the place of a file worth showing
        "lib/zeta.py": "Z = 1\n",
        "lib/alpha.py": "ALPHA = 'a little longer'\n",
        "lib/omega.py": "OMEGA = 'the longest plain file of the folder'\n",
        "lib/base_client.py": "class BaseClient:\n    '''The biggest file, but named like a base class.'''\n" * 3,
    })

    briefing = project_test_briefing(project, ["tests"], GENERATED, ["lib"])

    assert _checked(briefing) == ["tests/test_cart.py", "lib/base_client.py", "lib/zeta.py", "lib/alpha.py"]


def test_references_never_leave_the_project(tmp_path: Path) -> None:
    (tmp_path / "secret.ts").write_text(f"export const SECRET = '{OUTSIDE}';\n", encoding="utf-8")
    project = _project(tmp_path / "project", {
        "tests/a.spec.ts": "import { SECRET } from '../../secret';\nimport { helper } from './ngauto/helper';\n",
        "tests/ngauto/helper.ts": f"// {EARLIER_RUN}\n",
    })

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    assert _checked(briefing) == ["tests/a.spec.ts"]
    assert OUTSIDE not in briefing


def test_folder_links_are_never_followed(tmp_path: Path) -> None:
    outside = _project(tmp_path / "outside", {"pages/LoginPage.ts": f"export const SECRET = '{OUTSIDE}';\n"})
    project = _project(tmp_path / "project", {
        "tests/login.spec.ts": (
            "import { SECRET } from '../pages/LoginPage';\nimport { OTHER } from './linked/LoginPage';\n\n"
            "test('login', async () => {});\n"
        ),
    })
    _link(project / "pages", outside / "pages")
    _link(project / "tests/linked", outside / "pages")

    briefing = project_test_briefing(project, ["tests"], GENERATED, ["pages", "tests/linked"])

    assert _checked(briefing) == ["tests/login.spec.ts"]
    assert OUTSIDE not in briefing


def test_file_links_are_never_followed(tmp_path: Path) -> None:
    (tmp_path / "secret.ts").write_text(f"export const SECRET = '{OUTSIDE}';\n", encoding="utf-8")
    project = _project(tmp_path / "project", {"tests/login.spec.ts": "import { SECRET } from './secret';\n"})
    _link(project / "tests/secret.ts", tmp_path / "secret.ts")
    _link(project / "conftest.ts", tmp_path / "secret.ts")

    briefing = project_test_briefing(project, ["tests"], GENERATED, ["tests/secret.ts"])

    assert _checked(briefing) == ["tests/login.spec.ts"]
    assert OUTSIDE not in briefing


def test_tool_dependency_and_report_folders_are_never_read(tmp_path: Path) -> None:
    hidden = "SAVED-LOGIN-OR-TOOL-FILE"
    project = _project(tmp_path, {
        "tests/login.spec.ts": (
            "test.use({ storageState: '../playwright/.auth/user.json' });\n\ntest('login', async () => {});\n"
        ),
        "playwright/.auth/user.json": f'{{"cookies": [{{"name": "session", "value": "{hidden}"}}]}}\n',
        "tests/.cache/helper.ts": f"// {hidden}\n",
        "tests/node_modules/helper.ts": f"// {hidden}\n",
        ".tox/py314/lib/conftest.py": f"# {hidden}\n",
        "site-packages/shop/base_page.py": f"# {hidden}\n",
        "ngauto-results/report.json": f'{{"note": "{hidden}"}}\n',
        "allure-results/result.json": f'{{"note": "{hidden}"}}\n',
    })
    shared_paths = [".tox", "site-packages", "ngauto-results", "allure-results", "tests/.cache"]

    briefing = project_test_briefing(project, ["tests"], GENERATED, shared_paths)

    assert _checked(briefing) == ["tests/login.spec.ts"]
    assert hidden not in briefing


def test_binary_files_behind_a_source_suffix_are_never_shown(tmp_path: Path) -> None:
    project = _project(tmp_path, {"tests/login.spec.ts": "import { clip } from './clip';\n\ntest('login', () => {});\n"})
    (project / "tests/clip.ts").write_bytes(b"G@\x00\x10\x00" * 200)  # an MPEG transport stream, not TypeScript
    (project / "tests/Helper.cs").write_bytes("using Shop.Pages;\n".encode("utf-16"))  # UTF-16 text

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    assert _checked(briefing) == ["tests/login.spec.ts"]
    assert "\x00" not in briefing


def test_a_file_name_that_is_not_valid_text_is_left_out(tmp_path: Path) -> None:
    project = _project(tmp_path, {"tests/login.spec.ts": "test('login works', () => {});\n"})
    try:
        (project / "tests/odd\udc80name.spec.ts").write_text("test('odd', () => {});\n", encoding="utf-8")
    except (OSError, UnicodeError):
        pytest.skip("this file system cannot store such a name")

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    assert _checked(briefing) == ["tests/login.spec.ts"]
    briefing.encode("utf-8")  # a prompt must be valid text


def test_huge_files_and_pathological_lines_are_cut_and_never_hang(tmp_path: Path) -> None:
    pathological = "\n".join([
        "import " + "a." * 7_000 + "B;",
        "from " + "x " * 7_000 + "import y",
        "import '" + "a" * 14_000,
        "import " + "'a" * 7_000,
        "./" * 7_000,
        "use " + "\\" * 14_000,
        " " * 14_000 + "x",
    ])
    project = _project(tmp_path, {
        "pages/LoginPage.ts": "export class LoginPage {}\n",
        "tests/login.spec.ts": f"import {{ LoginPage }} from '../pages/LoginPage';\n{pathological}\n",
        "tests/huge.json": "[" + "1," * 2_000_000 + "1]\n",
    })

    briefing = project_test_briefing(project, ["tests"], GENERATED)

    shown = _checked(briefing)
    assert shown[:2] == ["pages/LoginPage.ts", "tests/login.spec.ts"]
    assert "tests/huge.json" in shown
    assert briefing.count("... (truncated)") == 2


def test_a_project_bigger_than_the_walk_limits_is_cut_shallow_files_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, {
        "conftest.py": "import pytest\n",
        "pages/base_page.py": "class BasePage:\n    pass\n",
        "tests/test_login.py": "from pages.base_page import BasePage\n",
    })
    monkeypatch.setattr(briefing_module, "MAX_PROJECT_FILES", 2)

    assert _checked(project_test_briefing(project, ["tests"], GENERATED, ["pages"])) == [
        "pages/base_page.py", "conftest.py",
    ]

    monkeypatch.setattr(briefing_module, "MAX_PROJECT_FILES", 100)
    monkeypatch.setattr(briefing_module, "MAX_PROJECT_FOLDERS", 1)

    assert _checked(project_test_briefing(project, ["tests"], GENERATED, ["pages"])) == ["conftest.py"]


def test_an_empty_project_gives_an_empty_briefing(tmp_path: Path) -> None:
    assert project_test_briefing(tmp_path, ["tests"], GENERATED, ["pages"]) == ""
