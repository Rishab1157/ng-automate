"""Tests for _expand_braces, the {a,b} support shared by the grep and glob tools.

fnmatch and Python's glob module have no brace syntax, so without it a pattern
such as "*.{ts,tsx}" (the grep tool's own example) only matched a file literally
named that way.
"""

import time

import pytest

from openhands.tools.utils import _expand_braces


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("*.java", ["*.java"]),
        ("*.{java,xml}", ["*.java", "*.xml"]),
        ("*.{java}", ["*.java"]),
        ("*.{,java}", ["*.", "*.java"]),
        ("{}", [""]),
        ("", [""]),
        ("{a,b}{c,d}", ["ac", "ad", "bc", "bd"]),
        (
            "src/{main,test}/**/*.{java,kt}",
            [
                "src/main/**/*.java",
                "src/main/**/*.kt",
                "src/test/**/*.java",
                "src/test/**/*.kt",
            ],
        ),
        ("*.{j{ava,son},xml}", ["*.java", "*.json", "*.xml"]),
        (
            "{ünï,cödé}/*.{jävä,txt}",
            [
                "ünï/*.jävä",
                "ünï/*.txt",
                "cödé/*.jävä",
                "cödé/*.txt",
            ],
        ),
        ("my dir/{a b,c}.txt", ["my dir/a b.txt", "my dir/c.txt"]),
    ],
)
def test_expand_braces(pattern, expected):
    assert _expand_braces(pattern) == expected


@pytest.mark.parametrize(
    "pattern",
    [
        "*.{java",  # never closed
        "*.java}",  # never opened
        "}{",
        "[{]*.java",  # a brace inside [...] is a plain character
        "[{,}]",
    ],
)
def test_unbalanced_or_bracketed_braces_stay_literal(pattern):
    assert _expand_braces(pattern) == [pattern]


def test_comma_inside_a_character_class_does_not_split():
    assert _expand_braces("{a,[,]}") == ["a", "[,]"]


def test_unclosed_outer_group_keeps_inner_group_working():
    assert _expand_braces("{a,{b,c}") == ["{a,b", "{a,c"]


def test_brace_bomb_is_refused_quickly():
    started = time.monotonic()
    with pytest.raises(ValueError, match="more than"):
        _expand_braces("{a,b}" * 30)  # 2**30 alternatives
    assert time.monotonic() - started < 2
