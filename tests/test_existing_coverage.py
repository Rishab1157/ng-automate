"""No redundant tests, the plain-code part: which tests count (the 'ngauto' folder only), a test that holds all of a
case's values and checks (never by ids or names), and the check every "already tested" answer must pass. Plain code
on a tmp project, no LLM."""

from pathlib import Path

import pytest

from app.agents.TestGeneratorAgent.ExistingCoverage import (
    DELETED,
    NAME_NOT_FOUND,
    NAME_TOO_SHORT,
    NOT_A_TEST_FILE,
    NOT_EARLIER,
    CoverageClaim,
    case_facts,
    check_claim,
    earlier_test_files,
    is_generated_path,
    parse_path,
    same_values_test,
)
from app.models.testDataModel import TestCaseSpecModel, TestStepModel

URL = "https://www.saucedemo.com/"
ERROR = "Epic sadface: Sorry, this user has been locked out."


def _login(case_id: str, user: str, password: str, locator: str, expected: str) -> TestCaseSpecModel:
    return TestCaseSpecModel(id=case_id, title=f"Login as {user}", steps=[
        TestStepModel(action="open", target=URL),
        TestStepModel(action="type", target="#user-name", value=user),
        TestStepModel(action="type", target="#password", value=password),
        TestStepModel(action="click", target="#login-button"),
        TestStepModel(action="verify_text", target=locator, expected=expected),
    ])


VALID = _login("TC-001", "standard_user", "secret_sauce", "[data-test='title']", "Products")
INVALID = _login("TC-002", "locked_out_user", "secret_sauce", "[data-test='error']", ERROR)

# What an earlier generation wrote for a valid login: named its own way, no case id, locators in the page object.
EARLIER = "src/test/java/com/demo/ngauto/SignInTest.java"
EARLIER_TEST = """package com.demo.ngauto;

public class SignInTest extends BaseTest {
    @Test
    public void signInWorks() {
        new LoginPage(driver).loginAs("standard_user", "secret_sauce");
        Assert.assertEquals(new InventoryPage(driver).title(), "Products");
    }
}
"""
FEATURE = "src/test/resources/features/ngauto/login.feature"
FEATURE_TEXT = (
    "Feature: Login\n  Scenario Outline: Login is refused\n    When I log in as \"<user>\" with \"<password>\"\n"
    "    Then I see \"<message>\"\n    Examples:\n      | user | password | message |\n"
    f"      | locked_out_user | secret_sauce | {ERROR} |\n"
    "      | standard_user | wrong_pass | Epic sadface: Username and password do not match any user in this service |\n"
)


def _project(root: Path, files: dict[str, str]) -> dict[str, str]:
    """Writes the files and returns them as the project before this run (path -> text), as the snapshot keeps it."""
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return dict(files)


def _tests(root: Path, files: dict[str, str]) -> dict[str, str]:
    return earlier_test_files(_project(root, files))


# ---------- which tests count: the 'ngauto' folder only ----------

def test_only_test_files_in_the_ngauto_folder_count(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {
        EARLIER: EARLIER_TEST,
        FEATURE: FEATURE_TEXT,
        "src/test/java/com/demo/tests/LoginTest.java": EARLIER_TEST,  # the user's own test: never compared
        "src/test/java/com/demo/ngautomation/LoginTest.java": EARLIER_TEST,  # a name that only starts with ngauto
        "src/test/java/com/demo/ngauto/pages/LoginPage.java": "public class LoginPage { void loginAs() {} }\n",
    })

    assert list(tests) == [EARLIER, FEATURE]


@pytest.mark.parametrize(("path", "inside"), [
    ("src/test/java/ngauto/A.java", True),
    ("tests/ngauto/sub/a.spec.ts", True),
    ("ngauto/test_a.py", True),
    ("tests/ngauto.spec.ts", False),  # a file named ngauto, not a folder
    ("tests/NGAUTO/a.spec.ts", False),
])
def test_the_ngauto_folder_is_a_folder_of_that_exact_name(path: str, inside: bool) -> None:
    assert is_generated_path(path) is inside


# ---------- the same values and checks, never the same id or name ----------

def test_a_cases_facts_are_its_values_checks_and_data_not_its_locators_or_urls() -> None:
    case = VALID.model_copy(update={"test_data": {"cart": "Sauce Labs Backpack", "qty": "1"}})

    assert case_facts(case) == ["standard_user", "secret_sauce", "products", "sauce labs backpack"]  # "1" proves nothing


def test_the_same_test_under_another_name_and_id_is_found(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {EARLIER: EARLIER_TEST})
    renamed = VALID.model_copy(update={"id": "LOGIN-NEW-7", "title": "User can sign in"})

    assert same_values_test(renamed, tests) == EARLIER


def test_an_earlier_test_named_after_the_case_but_doing_something_else_is_not_the_same(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {"src/test/java/com/demo/ngauto/TC_001_CheckoutTest.java": (
        "public class TC_001_CheckoutTest {\n  @Test public void tc001() {\n"
        "    cart.add(\"Sauce Labs Backpack\"); checkout(\"John\", \"Doe\");\n  }\n}\n"
    )})

    assert same_values_test(VALID, tests) is None


def test_valid_and_invalid_login_are_not_the_same_test(tmp_path: Path) -> None:
    # They share the page, the locators and the password: the values and checks tell them apart.
    tests = _tests(tmp_path, {EARLIER: EARLIER_TEST})

    assert same_values_test(INVALID, tests) is None


def test_most_but_not_all_values_is_not_the_same(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {EARLIER: EARLIER_TEST})

    assert same_values_test(_login("TC-3", "standard_user", "secret_sauce", "h1", "Your Cart"), tests) is None


def test_locators_and_urls_alone_are_not_the_same(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {"tests/ngauto/test_menu.py": (
        f"def test_menu(page):\n    page.goto('{URL}')\n    page.fill('#user-name', 'problem_user')\n"
        "    page.fill('#password', 'other')\n    page.click('#login-button')\n"
        "    assert page.text(\"[data-test='title']\") == 'Menu'\n"
    )})

    assert same_values_test(VALID, tests) is None


def test_values_count_only_whole(tmp_path: Path) -> None:
    case = TestCaseSpecModel(id="A-1", title="Admin", steps=[
        TestStepModel(action="type", target="#u", value="admin"), TestStepModel(action="verify_text", expected="Home"),
    ])
    tests = _tests(tmp_path, {"tests/ngauto/test_x.py": "def test_x():\n    login('administrator')\n    see('Homepage')\n"})

    assert same_values_test(case, tests) is None


def test_quotes_escapes_case_and_spaces_do_not_hide_a_value(tmp_path: Path) -> None:
    case = TestCaseSpecModel(id="Q-1", title="Quote", steps=[
        TestStepModel(action="type", target="#n", value='Say "hi"'),
        TestStepModel(action="verify_text", expected="Hello   WORLD"),
    ])
    path = "src/test/java/x/ngauto/QuoteTest.java"
    tests = _tests(tmp_path, {path: 'class QuoteTest {\n  @Test void q() {\n    type("Say \\"hi\\"");\n'
                                    '    check("hello world");\n  }\n}\n'})

    assert same_values_test(case, tests) == path


def test_one_data_driven_test_is_the_same_for_each_of_its_rows(tmp_path: Path) -> None:
    tests = _tests(tmp_path, {FEATURE: FEATURE_TEXT})
    wrong_password = _login("TC-4", "standard_user", "wrong_pass", "[data-test='error']",
                            "Epic sadface: Username and password do not match any user in this service")

    assert same_values_test(INVALID, tests) == FEATURE and same_values_test(wrong_password, tests) == FEATURE


def test_a_value_too_short_to_count_leaves_the_case_to_the_judge(tmp_path: Path) -> None:
    # Found as a tester: valid login plus "the cart badge shows 1". The valid-login test holds every value that counts
    # (standard_user, secret_sauce, Products) and "1" counts for nothing, so plain code would call it the same test
    # and skip a case that checks the cart. It must not decide: the judge reads both.
    tests = _tests(tmp_path, {EARLIER: EARLIER_TEST})
    with_badge = VALID.model_copy(update={"id": "TC-9", "steps": [
        *VALID.steps,
        TestStepModel(action="click", target="[data-test='add-to-cart-sauce-labs-backpack']"),
        TestStepModel(action="verify_text", target=".shopping_cart_badge", expected="1"),
    ]})

    assert same_values_test(with_badge, tests) is None
    assert same_values_test(VALID, tests) == EARLIER  # without the badge it is the same test


def test_a_case_with_fewer_than_two_values_is_left_to_the_judge(tmp_path: Path) -> None:
    # A story in prose gives no values: nothing here can tell, so the judge compares by meaning.
    tests = _tests(tmp_path, {EARLIER: EARLIER_TEST})
    story = TestCaseSpecModel(id="S-1", title="A shopper signs in", steps=[
        TestStepModel(action="step", text="The shopper signs in with a valid account"),
        TestStepModel(action="verify_text", text="The product list is shown", expected="Products"),
    ])

    assert same_values_test(story, tests) is None


# ---------- the check every "already tested" answer must pass ----------

def test_an_answer_on_a_real_earlier_test_holds(tmp_path: Path) -> None:
    before = _project(tmp_path, {EARLIER: EARLIER_TEST, FEATURE: FEATURE_TEXT})

    assert check_claim(CoverageClaim("TC-001", EARLIER, "signInWorks"), before, tmp_path) is None
    # A scenario title, whatever its case and spaces; one data-driven test may cover several cases.
    assert check_claim(CoverageClaim("TC-002", FEATURE, "login   IS refused"), before, tmp_path) is None


def test_a_test_written_in_this_run_is_no_earlier_test(tmp_path: Path) -> None:
    before = _project(tmp_path, {})
    _project(tmp_path, {EARLIER: EARLIER_TEST})  # written now

    assert check_claim(CoverageClaim("TC-001", EARLIER, "signInWorks"), before, tmp_path) == NOT_EARLIER.format(
        path=EARLIER
    )


def test_a_test_outside_the_ngauto_folder_does_not_count(tmp_path: Path) -> None:
    own = "src/test/java/com/demo/tests/SignInTest.java"
    before = _project(tmp_path, {own: EARLIER_TEST})

    assert check_claim(CoverageClaim("TC-001", own, "signInWorks"), before, tmp_path) == NOT_EARLIER.format(path=own)


@pytest.mark.parametrize("path", ["../outside/ngauto/SignInTest.java", "", "src/test/java/com/demo/ngauto/"])
def test_a_path_that_is_no_project_file_does_not_count(tmp_path: Path, path: str) -> None:
    before = _project(tmp_path, {EARLIER: EARLIER_TEST})

    assert check_claim(CoverageClaim("TC-001", path, "signInWorks"), before, tmp_path) == NOT_EARLIER.format(
        path=path or "the file"
    )


def test_shared_code_is_no_test(tmp_path: Path) -> None:
    page = "src/test/java/com/demo/ngauto/pages/LoginPage.java"
    before = _project(tmp_path, {page: "public class LoginPage {\n  public void loginAs(String u) {}\n}\n"})

    assert check_claim(CoverageClaim("TC-001", page, "loginAs"), before, tmp_path) == NOT_A_TEST_FILE.format(path=page)


@pytest.mark.parametrize("test", ["signInWorksFast", "InWorks", "signIn"])
def test_a_test_name_counts_only_as_a_whole_name(tmp_path: Path, test: str) -> None:
    before = _project(tmp_path, {EARLIER: EARLIER_TEST})

    assert check_claim(CoverageClaim("TC-001", EARLIER, test), before, tmp_path) == NAME_NOT_FOUND.format(
        path=EARLIER, test=test
    )


def test_a_deleted_earlier_test_does_not_count(tmp_path: Path) -> None:
    before = _project(tmp_path, {EARLIER: EARLIER_TEST})
    (tmp_path / EARLIER).unlink()

    assert check_claim(CoverageClaim("TC-001", EARLIER, "signInWorks"), before, tmp_path) == DELETED.format(path=EARLIER)


@pytest.mark.parametrize("test", ["", "si"])
def test_an_answer_must_name_the_test(tmp_path: Path, test: str) -> None:
    before = _project(tmp_path, {EARLIER: EARLIER_TEST})

    assert check_claim(CoverageClaim("TC-001", EARLIER, test), before, tmp_path) == NAME_TOO_SHORT


@pytest.mark.parametrize("written", [
    EARLIER, f"./{EARLIER}", f"/workspace/project/{EARLIER}", f"/{EARLIER}", EARLIER.replace("/", "\\"), f"`{EARLIER}`",
])
def test_a_path_is_read_however_a_model_writes_it(written: str) -> None:
    assert parse_path(written) == EARLIER
