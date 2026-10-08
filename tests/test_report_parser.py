from pathlib import Path

from app.models.testRunModel import TestCaseStatus
from app.utils.TestReportParser import MAX_REPORT_BYTES, parse_report_file, parse_reports

SUREFIRE = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.demo.LoginTest" tests="3" failures="1" errors="1" skipped="1" time="2.5">
  <testcase name="validLogin" classname="com.demo.LoginTest" time="1.2"/>
  <testcase name="wrongPassword" classname="com.demo.LoginTest" time="0.8">
    <failure message="expected [Dashboard] but found [Login]" type="java.lang.AssertionError">java.lang.AssertionError: expected [Dashboard]
	at com.demo.LoginTest.wrongPassword(LoginTest.java:31)</failure>
  </testcase>
  <testcase name="lockedUser" classname="com.demo.LoginTest" time="0.5">
    <error message="no such element: Unable to locate element" type="org.openqa.selenium.NoSuchElementException"/>
  </testcase>
  <testcase name="later" classname="com.demo.LoginTest" time="0"><skipped/></testcase>
</testsuite>"""

PYTEST = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="2">
<testcase classname="tests.test_search" name="test_ok" time="0.01"/>
<testcase classname="tests.test_search" name="test_bad" time="0.02"><failure message="assert 0 &gt; 0">E assert 0 &gt; 0</failure></testcase>
</testsuite></testsuites>"""

TESTNG = """<?xml version="1.0" encoding="UTF-8"?>
<testng-results skipped="0" failed="1" total="2" passed="1">
  <suite name="Shop UI"><test name="Cucumber"><class name="com.demo.runners.TestRunner">
    <test-method status="PASS" signature="setUp()" name="setUp" is-config="true" duration-ms="5"/>
    <test-method status="PASS" name="runScenario" duration-ms="1500"/>
    <test-method status="FAIL" name="runScenario" duration-ms="900">
      <exception class="org.openqa.selenium.TimeoutException">
        <message><![CDATA[Expected condition failed: waiting for visibility of element located by By.id: cart]]></message>
        <full-stacktrace><![CDATA[org.openqa.selenium.TimeoutException: ...]]></full-stacktrace>
      </exception>
    </test-method>
  </class></test></suite>
</testng-results>"""


def _write(folder: Path, relative: str, text: str) -> Path:
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_surefire_report(tmp_path: Path) -> None:
    cases = parse_report_file(_write(tmp_path, "TEST-com.demo.LoginTest.xml", SUREFIRE))

    assert [(c.name, c.status) for c in cases] == [
        ("validLogin", TestCaseStatus.PASSED),
        ("wrongPassword", TestCaseStatus.FAILED),
        ("lockedUser", TestCaseStatus.ERROR),
        ("later", TestCaseStatus.SKIPPED),
    ]
    assert cases[1].suite == "com.demo.LoginTest"
    assert cases[1].message == "expected [Dashboard] but found [Login]"
    assert "LoginTest.java:31" in cases[1].details
    assert cases[0].duration_seconds == 1.2


def test_pytest_report_with_testsuites_root(tmp_path: Path) -> None:
    cases = parse_report_file(_write(tmp_path, "junit.xml", PYTEST))

    assert [c.status for c in cases] == [TestCaseStatus.PASSED, TestCaseStatus.FAILED]
    assert cases[1].message == "assert 0 > 0"


def test_testng_results_skip_config_methods(tmp_path: Path) -> None:
    cases = parse_report_file(_write(tmp_path, "testng-results.xml", TESTNG))

    assert [c.status for c in cases] == [TestCaseStatus.PASSED, TestCaseStatus.FAILED]
    assert cases[1].suite == "com.demo.runners.TestRunner"
    assert cases[1].duration_seconds == 0.9
    assert "waiting for visibility" in cases[1].message


def test_doctype_is_refused(tmp_path: Path) -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]><testsuite><testcase name="x"/></testsuite>'

    assert parse_report_file(_write(tmp_path, "bomb.xml", bomb)) == []


def test_malformed_and_unknown_files_are_skipped(tmp_path: Path) -> None:
    assert parse_report_file(_write(tmp_path, "broken.xml", "<testsuite><testcase")) == []
    assert parse_report_file(_write(tmp_path, "other.xml", "<project/>")) == []


def test_huge_file_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "huge.xml"
    with path.open("wb") as f:
        f.truncate(MAX_REPORT_BYTES + 1)

    assert parse_report_file(path) == []


def test_parse_reports_collects_globs_once_and_skips_bad_files(tmp_path: Path) -> None:
    _write(tmp_path, "target/surefire-reports/TEST-a.xml", SUREFIRE)
    _write(tmp_path, "target/surefire-reports/TEST-broken.xml", "<oops")
    _write(tmp_path, "target/surefire-reports/testng-results.xml", TESTNG)

    cases, files = parse_reports(
        tmp_path, ["target/surefire-reports/TEST-*.xml", "target/surefire-reports/*.xml"]
    )

    assert files == [
        "target/surefire-reports/TEST-a.xml",
        "target/surefire-reports/TEST-broken.xml",
        "target/surefire-reports/testng-results.xml",
    ]
    assert len(cases) == 6
