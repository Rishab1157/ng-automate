"""The locators the generator had to assume, and the new test files a plain test run would skip."""

from pathlib import Path

from app.agents.TestGeneratorAgent.GeneratedTestPlacement import unverified_locators
from app.models.healerModel import FileChangeModel


def test_assumed_locators_are_found_with_their_code_line(tmp_path: Path) -> None:
    page = tmp_path / "src/test/java/com/acme/ngauto/CheckoutPage.java"
    page.parent.mkdir(parents=True)
    page.write_bytes(
        b"class CheckoutPage {\n"
        b"    By ok = By.id(\"ok\");\n"
        b"    By checkout = By.id(\"checkout\"); // NGAUTO-UNVERIFIED: assumed locator, not in the test data\n"
        b"    // NGAUTO-UNVERIFIED: assumed locator, not in the test data\n"
        b"    By pay = By.cssSelector(\"button.pay\");\n"
        b"}\n"
    )
    steps = tmp_path / "tests/test_cart.py"
    steps.parent.mkdir()
    steps.write_bytes(b"page.click('#remove')  # NGAUTO-UNVERIFIED: assumed\n")
    files = [
        FileChangeModel(path="src/test/java/com/acme/ngauto/CheckoutPage.java", change="added"),
        FileChangeModel(path="tests/test_cart.py", change="modified"),
        FileChangeModel(path="gone.py", change="deleted"),
    ]

    found = unverified_locators(tmp_path, files)

    assert [(item.path, item.line, item.code) for item in found] == [
        ("src/test/java/com/acme/ngauto/CheckoutPage.java", 3, 'By checkout = By.id("checkout");'),
        ("src/test/java/com/acme/ngauto/CheckoutPage.java", 4, 'By pay = By.cssSelector("button.pay");'),
        ("tests/test_cart.py", 1, "page.click('#remove')"),
    ]
