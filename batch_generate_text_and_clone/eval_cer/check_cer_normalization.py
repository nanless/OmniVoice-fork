#!/usr/bin/env python3
"""Self-checks for conservative CER normalization; no test framework needed."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batch_generate_text_and_clone.eval_cer.cer_normalization import (  # noqa: E402
    normalize_hypothesis,
    normalize_reference,
    reference_normalization_context,
    reference_normalization_input_fingerprint,
    normalize_safe,
    normalize_strict,
    normalize_text,
)
from batch_generate_text_and_clone.text_generation.text_tn import build_text_tn  # noqa: E402


def _equal(actual: str, expected: str) -> None:
    assert actual == expected, f"expected {expected!r}, got {actual!r}"


def run_checks() -> int:
    strict_cases = {
        "ＡＢＣ　１２．５厘米！": "abc 12.5厘米",
        "８０％，不是八十。": "80%不是八十",
        "Meet me at 8：05, okay?": "meet me at 8:05 okay",
        "uh-huh / we'll": "uh-huh we'll",
        "三点一四": "三点一四",
    }
    safe_cases = {
        "十万": "⟦NUM:100000⟧",
        "十二万三千": "⟦NUM:123000⟧",
        "一亿零三万零五十": "⟦NUM:100030050⟧",
        "一万亿三千万": "⟦NUM:1000030000000⟧",
        "2026/7/20": "⟦DATE:2026-07-20⟧",
        "二〇二六年七月二十日": "⟦DATE:2026-07-20⟧",
        "8:05": "⟦TIME:08:05⟧",
        "下午三点半": "⟦TIME:15:30⟧",
        "八点五分": "⟦TIME:08:05⟧",
        "80%": "⟦PERCENT:80⟧",
        "百分之八十": "⟦PERCENT:80⟧",
        "80 percent": "⟦PERCENT:80⟧",
        "￥１２．５０": "⟦MONEY:CNY:12.5⟧",
        "十二元五角": "⟦MONEY:CNY:12.5⟧",
        "$12.50": "⟦MONEY:USD:12.5⟧",
        "１２．５厘米": "⟦QTY:12.5:cm⟧",
        "十二公里": "⟦QTY:12:km⟧",
        "2 kilograms": "⟦QTY:2:kg⟧",
        "编号 0012": "编号⟦DIGITS:0012⟧",
        "twelve": "⟦NUM:12⟧",
        "one hundred and five": "⟦NUM:105⟧",
        "minus twelve point five": "⟦NUM:-12.5⟧",
        "twenty-one": "⟦NUM:21⟧",
        "eighty percent": "⟦PERCENT:80⟧",
        "twelve centimeters": "⟦QTY:12:cm⟧",
        "第十二": "⟦ORDINAL:12⟧",
        "12th": "⟦ORDINAL:12⟧",
        "fifteenth": "⟦ORDINAL:15⟧",
        "twenty-first": "⟦ORDINAL:21⟧",
        "二分之一": "⟦FRACTION:1/2⟧",
        "七分之三": "⟦FRACTION:3/7⟧",
        "three sevenths": "⟦FRACTION:3/7⟧",
        "3/7": "⟦FRACTION:3/7⟧",
        "½": "⟦FRACTION:1/2⟧",
        "八点零五": "⟦TIME:08:05⟧",
        "8点05": "⟦TIME:08:05⟧",
        "下午3:05": "⟦TIME:15:05⟧",
        "8:05 pm": "⟦TIME:20:05⟧",
        "12:00 am": "⟦TIME:00:00⟧",
        "三点一刻": "⟦TIME:03:15⟧",
        "下午三点": "⟦TIME:15:00⟧",
        "三点钟": "⟦TIME:03:00⟧",
        "three o'clock": "⟦TIME:03:00⟧",
        "3:15 PM": "⟦TIME:15:15⟧",
        "早上8点20响": "⟦TIME:08:20⟧响",
        "下午三点一四": "⟦TIME:15:14⟧",
        "Tom&nbsp;&amp;&nbsp;Jerry": "tom jerry",
        "rock&#39;n": "rock'n",
        "&#xFF11;&#xFF12;": "⟦NUM:12⟧",
        "1,234": "⟦NUM:1234⟧",
        "2/4 kg": "⟦QTY:1/2:kg⟧",
        "&madeup;": "⟦HTML:madeup⟧",
        "the second speaker": "the ⟦ORDINAL:2⟧ speaker",
        "one one": "⟦NUM:1⟧ ⟦NUM:1⟧",
        "one two": "⟦NUM:1⟧ ⟦NUM:2⟧",
        "12,34": "⟦DIGITS:12,34⟧",
        "010-1234567": "⟦DIGITS:0101234567⟧",
        "编号0012": "编号⟦DIGITS:0012⟧",
        "first place": "⟦ORDINAL:1⟧ place",
        "11th": "⟦ORDINAL:11⟧",
        "21st": "⟦ORDINAL:21⟧",
        "three seventeenths": "⟦FRACTION:3/17⟧",
        "two twentieths": "⟦FRACTION:1/10⟧",
        "十二点五厘米": "⟦QTY:12.5:cm⟧",
        "区号是零一零": "区号是⟦DIGITS:010⟧",
        "区号是010": "区号是⟦DIGITS:010⟧",
        "phone number is zero one oh": "phone number is ⟦DIGITS:010⟧",
        "phone number is 010": "phone number is ⟦DIGITS:010⟧",
        "2026-07-20T08:05:30": "⟦DATE:2026-07-20⟧t08:05:30",
        "(3/7, -2)": "⟦FRACTION:3/7⟧ ⟦NUM:-2⟧",
        "比分(3/4)": "比分3/4",
        "version (1.2.3)": "version 1.2.3",
        "03/04,": "03/04",
    }
    protected_cases = (
        "呃那个嗯",
        "啦嘛呀啊哦诶咦",
        "一点儿女儿花儿",
        "不不不我我我",
        "方方芳芳",
        "i miss you miss wang",
        "百分之苹果",
        "三点一四",
        "万一迟到了",
        "千万不要改",
        "一块蛋糕",
        "03/04",
        "version 1.2.3",
        "三月四日",
        "oneplus",
        "someone",
        "one twenty",
        "hundreds of people",
        "give me a second",
        "i second that",
        "11st",
        "12nd",
        "12:60",
        "24:00",
        "比分1:23",
        "比分3/4",
        "1:02.345",
        "tell mom first",
        "give me a third of it",
        "有一点希望",
        "给我两点建议",
        "12:30:45",
        "零点八",
        "second nature",
        "零一零",
    )

    count = 0
    for source, expected in strict_cases.items():
        _equal(normalize_strict(source), expected)
        count += 1
    for source, expected in safe_cases.items():
        actual = normalize_safe(source)
        _equal(actual, expected)
        _equal(normalize_safe(actual), actual)
        count += 2
    for source in protected_cases:
        _equal(normalize_safe(source), source)
        count += 1

    # Entity type must remain evidence: these pairs are intentionally unequal.
    assert normalize_safe("80") != normalize_safe("80%")
    assert normalize_safe("十二") != normalize_safe("十二元")
    assert normalize_safe("8:05") != normalize_safe("805")
    assert normalize_safe("12") != normalize_safe("12th")
    assert normalize_safe("二分之一") != normalize_safe("50%")
    assert normalize_safe("0012") != normalize_safe("12")
    assert normalize_safe("one two") != normalize_safe("twelve")
    assert normalize_safe("zero one oh") != normalize_safe("phone number is zero one oh")
    count += 8

    _equal(normalize_text("十万", profile="strict"), "十万")
    _equal(normalize_text("十万", profile="safe"), "⟦NUM:100000⟧")
    count += 2
    try:
        normalize_text("x", profile="permissive")
    except ValueError:
        count += 1
    else:
        raise AssertionError("unknown profiles must fail closed")

    # The training/reference path must call the same safe profile after its
    # explicit tag-to-spoken-text conversion.
    _equal(build_text_tn("价格是８０％！"), "价格是⟦PERCENT:80⟧")
    _equal(build_text_tn("Well, [confirmation-en] I miss you.", language="en"), "well uh-huh i miss you")
    _equal(
        normalize_reference(
            "Well, [confirmation-en] I miss you.", language="en"
        ),
        "well uh-huh i miss you",
    )
    _equal(normalize_reference("[laughter] 好好笑", language="zh"), "哈哈好好笑")
    _equal(
        normalize_hypothesis("Well, uh-huh I miss you."),
        "well uh-huh i miss you",
    )
    assert normalize_hypothesis("[confirmation-en]") != normalize_reference(
        "[confirmation-en]", language="en"
    )
    _equal(
        normalize_reference("literal [abc] text", language="en"),
        "literal ⟦TAG:abc⟧ text",
    )
    assert reference_normalization_context("en", None)["speech_tag_language"] == "en"
    assert reference_normalization_context("zh", None)["speech_tag_language"] == "zh"
    assert reference_normalization_context(None, None)["speech_tag_language"] == "unknown"
    assert reference_normalization_context(
        None, None, text="你好 hello"
    )["speech_tag_language"] == "zh"
    assert reference_normalization_context(
        None, None, text="hello"
    )["speech_tag_language"] == "en"
    assert reference_normalization_context(
        "en", "pure_zh", text="你好"
    )["speech_tag_language_source"] == "language"
    _equal(normalize_reference("[question-ei]你好"), "诶你好")
    _equal(normalize_reference("[question-oh] hello"), "oh hello")
    assert reference_normalization_input_fingerprint(
        "[confirmation-en]", language="en"
    ) != reference_normalization_input_fingerprint(
        "[confirmation-en]", language="zh"
    )
    count += 16

    tag_cases = {
        "[laughter]": ("哈哈", "haha"),
        "[sigh]": ("哎", ""),
        "[question-ah]": ("啊", "huh"),
        "[question-oh]": ("哦", "oh"),
        "[question-ei]": ("诶", ""),
        "[question-yi]": ("咦", ""),
        "[question-en]": ("嗯", "hmm"),
        "[surprise-ah]": ("啊", "ah"),
        "[surprise-oh]": ("哦", "oh"),
        "[surprise-wa]": ("哇", "wow"),
        "[surprise-yo]": ("哟", "yo"),
        "[dissatisfaction-hnn]": ("嗯", "hmm"),
        "[confirmation-en]": ("嗯", "uh-huh"),
    }
    for tag, (zh_spoken, en_spoken) in tag_cases.items():
        _equal(normalize_reference(tag, language="zh"), normalize_safe(zh_spoken))
        _equal(normalize_reference(tag, language="en"), normalize_safe(en_spoken))
        count += 2
    return count


if __name__ == "__main__":
    checks = run_checks()
    print(f"cer_normalization self-check: {checks} checks passed")
