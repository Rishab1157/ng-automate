"""No redundant tests, the judge: whether an earlier 'ngauto' test already does what a case says, by meaning (test
data in any form), with a fake structured LLM call. Every way its answer can be wrong must leave the case in."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app.agents.TestGeneratorAgent.ExistingCoverage import CoverageClaim, earlier_test_files
from app.agents.TestGeneratorAgent.RedundancyJudge import MAX_TESTS_CHARS, Verdict, describe_case, judge_case
from app.models.llmModel import LlmConfigModel
from app.models.testDataModel import TestCaseSpecModel, TestStepModel

CONFIG = LlmConfigModel(model="ollama_chat/qwen2.5-coder:14b", api_key=SecretStr("not-used"))
EARLIER = "src/test/java/com/demo/ngauto/SignInTest.java"
EARLIER_TEST = (
    "public class SignInTest extends BaseTest {\n    @Test\n    public void signInShowsTheProducts() {\n"
    '        new LoginPage(driver).loginAs("standard_user", "secret_sauce");\n'
    '        Assert.assertEquals(new InventoryPage(driver).title(), "Products");\n    }\n}\n'
)
# A story in prose, read from the user's knowledge base: no id worth anything, no values, no locators.
STORY = TestCaseSpecModel(id="TD-007", title="A registered shopper signs in", steps=[
    TestStepModel(action="step", text="The shopper opens the shop and signs in with a valid account"),
    TestStepModel(action="verify_text", text="The product list is shown"),
])


class FakeLlm:
    """Answers like complete_json: the parsed JSON (through the caller's parser), or raises ValueError."""

    def __init__(self, answer: dict[str, Any] | str | Exception) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, llm_config: LlmConfigModel, instructions: str, text: str, schema_name: str,
                 schema: dict[str, Any], parse: Any, **kwargs: Any) -> Any:
        self.calls.append({"instructions": instructions, "text": text, "schema": schema})
        if isinstance(self.answer, Exception):
            raise self.answer
        return parse(self.answer if isinstance(self.answer, str) else json.dumps(self.answer))


def _project(root: Path, files: dict[str, str]) -> dict[str, str]:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return dict(files)


def _judge(root: Path, answer: dict[str, Any] | str | Exception, files: dict[str, str] | None = None,
           case: TestCaseSpecModel = STORY) -> tuple[Verdict, FakeLlm]:
    before = _project(root, files if files is not None else {EARLIER: EARLIER_TEST})
    llm = FakeLlm(answer)
    return judge_case(CONFIG, case, earlier_test_files(before), before, root, llm), llm


def _yes(file: str = EARLIER, test: str = "signInShowsTheProducts") -> dict[str, Any]:
    return {"covered": True, "file": file, "test": test, "reason": "it signs in with a valid user and checks the products"}


def test_a_story_without_ids_or_values_is_matched_by_meaning(tmp_path: Path) -> None:
    verdict, llm = _judge(tmp_path, _yes())

    assert verdict == Verdict(CoverageClaim("TD-007", EARLIER, "signInShowsTheProducts"),
                              "it signs in with a valid user and checks the products")
    [call] = llm.calls
    assert "Title: A registered shopper signs in" in call["text"] and "signs in with a valid account" in call["text"]
    assert f"File: {EARLIER}\n{EARLIER_TEST}" in call["text"]
    assert "never by names, ids or wording" in call["instructions"]
    assert "a valid login and a refused login are different tests" in call["instructions"]


def test_no_is_no(tmp_path: Path) -> None:
    verdict, _ = _judge(tmp_path, {"covered": False, "file": "", "test": "", "reason": "it checks the cart"})

    assert verdict == Verdict(None, "it checks the cart")


@pytest.mark.parametrize(("answer", "problem"), [
    (_yes(test="testSomethingElse"), f"{EARLIER} has no test named 'testSomethingElse'"),  # a made-up test name
    (_yes(test=""), "it gives no test name"),
    (_yes(file="src/test/java/com/demo/ngauto/Invented.java"),
     "it named src/test/java/com/demo/ngauto/Invented.java, which is not one of the tests shown"),
    (_yes(file="src/test/java/com/demo/tests/LoginTest.java"),
     "it named src/test/java/com/demo/tests/LoginTest.java, which is not one of the tests shown"),
    (_yes(file=""), "it named no file, which is not one of the tests shown"),
])
def test_an_answer_that_does_not_hold_leaves_the_case_in(tmp_path: Path, answer: dict[str, Any], problem: str) -> None:
    verdict, _ = _judge(tmp_path, answer)

    assert verdict.claim is None and verdict.problem == problem


def test_a_path_written_the_sandbox_way_is_the_same_file(tmp_path: Path) -> None:
    verdict, _ = _judge(tmp_path, _yes(file=f"/workspace/project/{EARLIER}"))

    assert verdict.claim == CoverageClaim("TD-007", EARLIER, "signInShowsTheProducts")


@pytest.mark.parametrize("failure", [ValueError("the model call failed (Timeout)"), RuntimeError("boom")])
def test_a_failed_call_never_stops_anything_and_leaves_the_case_in(tmp_path: Path, failure: Exception) -> None:
    verdict, _ = _judge(tmp_path, failure)

    assert verdict.claim is None and verdict.problem is not None and verdict.problem.startswith("the check failed")


@pytest.mark.parametrize("reply", ["not json", '{"file": "x"}', '{"covered": "yes"}'])
def test_an_unusable_reply_leaves_the_case_in(tmp_path: Path, reply: str) -> None:
    verdict, _ = _judge(tmp_path, reply)

    assert verdict.claim is None and verdict.problem is not None


def test_nothing_to_compare_makes_no_call(tmp_path: Path) -> None:
    verdict, llm = _judge(tmp_path, _yes(), files={})

    assert verdict == Verdict(None) and llm.calls == []


def test_only_the_most_alike_tests_are_shown_when_they_do_not_all_fit(tmp_path: Path) -> None:
    filler = "    // " + "x" * 3000 + "\n"
    files = {f"src/test/java/ngauto/Cart{n}Test.java": f"class Cart{n}Test {{ @Test void cart() {{}}\n{filler}}}\n"
             for n in range(8)}
    files[EARLIER] = EARLIER_TEST  # the one that shares words with the story (shopper, signs, products)

    _, llm = _judge(tmp_path, _yes(), files=files)

    text = llm.calls[0]["text"]
    assert text.index(f"File: {EARLIER}") < text.index("File: src/test/java/ngauto/Cart")
    assert len(text) < MAX_TESTS_CHARS + 2000


def test_a_case_is_described_in_its_own_words_with_whatever_it_gives() -> None:
    case = TestCaseSpecModel(id="X", title="Checkout", description="Buy one item", preconditions=["signed in"], steps=[
        TestStepModel(action="type", target="#zip", value="12345", text="Enter the zip code"),
        TestStepModel(action="verify_text", target_hint="Thank you header", expected="Thank you"),
    ], test_data={"item": "Backpack"})

    assert describe_case(case) == (
        "Title: Checkout\nAbout: Buy one item\nGiven: signed in\n"
        '1. Enter the zip code element="#zip" value="12345"\n'
        '2. verify_text element="Thank you header" expected="Thank you"\nData: item=Backpack'
    )
