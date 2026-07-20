#!/usr/bin/env python3
"""Conservative, deterministic text normalization for CER comparison.

Two profiles are intentionally provided:

``strict``
    Surface-only cleanup.  It folds full-width ASCII, lowercases, normalizes
    whitespace, and removes sentence punctuation while retaining punctuation
    that carries lexical or numeric meaning.

``safe``
    Applies ``strict`` cleanup and additionally canonicalizes only explicit,
    deterministic numeric entities.  Canonical tokens keep the entity type so
    that, for example, ``80`` and ``80%`` can never become equal by accident.

The module deliberately has no fuzzy/phonetic rules.  Fillers, particles,
rhotic suffixes, repetitions, and homophones are evidence in a speech error
metric and must remain distinguishable.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import html
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Match, Optional, Tuple

STRICT_PROFILE = "strict"
SAFE_PROFILE = "safe"
PROFILE_VERSIONS = {STRICT_PROFILE: "strict-v2", SAFE_PROFILE: "safe-v2"}
CER_METRIC = "deterministic_char_cer"
EVAL_SCHEMA_VERSION = 4
CER_SCORE_VERSION = 4
NORMALIZATION_VERSION = 4

_CANONICAL_RE = re.compile(
    r"⟦(?:NUM|DIGITS|ORDINAL|FRACTION|PERCENT|DATE|TIME|MONEY|QTY|TAG|HTML):[^⟦⟧]+⟧"
)

_HTML_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_SPEECH_TAG_RE = re.compile(r"\[[^\]]+\]")
_TAG_TO_SPOKEN_ZH = {
    "[laughter]": "哈哈",
    "[sigh]": "哎",
    "[question-ah]": "啊",
    "[question-oh]": "哦",
    "[question-ei]": "诶",
    "[question-yi]": "咦",
    "[question-en]": "嗯",
    "[surprise-ah]": "啊",
    "[surprise-oh]": "哦",
    "[surprise-wa]": "哇",
    "[surprise-yo]": "哟",
    "[dissatisfaction-hnn]": "嗯",
    "[confirmation-en]": "嗯",
}
_TAG_TO_SPOKEN_EN = {
    "[laughter]": "haha",
    "[sigh]": "",
    "[question-ah]": "huh",
    "[question-oh]": "oh",
    "[question-ei]": "",
    "[question-yi]": "",
    "[question-en]": "hmm",
    "[surprise-ah]": "ah",
    "[surprise-oh]": "oh",
    "[surprise-wa]": "wow",
    "[surprise-yo]": "yo",
    "[dissatisfaction-hnn]": "hmm",
    "[confirmation-en]": "uh-huh",
}

_ZH_DIGIT_VALUES = {
    "零": 0,
    "〇": 0,
    "○": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "两": 2,
    "兩": 2,
    "贰": 2,
    "貳": 2,
    "三": 3,
    "叁": 3,
    "參": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "陸": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
_ZH_SMALL_UNITS = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
}
_ZH_LARGE_UNITS = {"万": 10_000, "萬": 10_000, "亿": 100_000_000, "億": 100_000_000}
_ZH_NUMBER_CHARS = "零〇○一壹二两兩贰貳三叁參四肆五伍六陆陸七柒八捌九玖十拾百佰千仟万萬亿億"
_ZH_DIGIT_SEQUENCE_CHARS = "零〇○一壹二两兩贰貳三叁參四肆五伍六陆陸七柒八捌九玖"
_ZH_INTEGER = rf"[{_ZH_NUMBER_CHARS}]+"
_ZH_DECIMAL = rf"[{_ZH_NUMBER_CHARS}]+(?:点[零〇○一壹二两兩贰貳三叁參四肆五伍六陆陸七柒八捌九玖]+)?"
_ARABIC_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_ARABIC_NUMBER = rf"[+-]?(?:{_ARABIC_INTEGER}(?:\.\d+)?|\.\d+)"
_EXPLICIT_NUMBER = rf"(?:{_ARABIC_NUMBER}|{_ZH_DECIMAL})"

_DATE_AR_RE = re.compile(
    r"(?<!\d)(?P<y>\d{4})\s*[-/.]\s*(?P<m>\d{1,2})\s*[-/.]\s*(?P<d>\d{1,2})(?!\d)"
)
_DATE_ZH_RE = re.compile(
    rf"(?P<y>{_ZH_INTEGER}|\d{{4}})年\s*(?P<m>{_ZH_INTEGER}|\d{{1,2}})月\s*(?P<d>{_ZH_INTEGER}|\d{{1,2}})(?:日|号)"
)
_TIME_AR_RE = re.compile(
    r"(?<![\d:])(?:(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)\s*)?"
    r"(?P<h>\d{1,2}):(?P<m>\d{2})"
    r"(?:\s*(?P<ampm>a\.?m\.?|p\.?m\.?))?(?![\d:.])",
    re.IGNORECASE,
)
_TIME_ZH_DETAIL_RE = re.compile(
    rf"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?"
    rf"(?P<h>{_ZH_INTEGER}|\d{{1,2}})点(?:(?P<half>半)|(?P<m>{_ZH_INTEGER}|\d{{1,2}})分)"
)
_TIME_ZH_PERIOD_MINUTE_RE = re.compile(
    rf"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)"
    rf"(?P<h>{_ZH_INTEGER}|\d{{1,2}})点"
    rf"(?P<m>{_ZH_INTEGER}|\d{{1,2}})(?![{_ZH_NUMBER_CHARS}\d])"
)
_TIME_ZH_ZERO_MINUTE_RE = re.compile(
    rf"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?"
    rf"(?P<h>{_ZH_INTEGER}|\d{{1,2}})点"
    rf"(?P<m>(?:零[一二两三四五六七八九]|0\d))"
)
_TIME_ZH_QUARTER_RE = re.compile(
    rf"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?"
    rf"(?P<h>{_ZH_INTEGER}|\d{{1,2}})点(?P<quarter>一刻|三刻)"
)
_TIME_ZH_HOUR_RE = re.compile(
    rf"(?P<period>凌晨|早上|上午|中午|下午|傍晚|晚上)?"
    rf"(?P<h>{_ZH_INTEGER}|\d{{1,2}})点(?P<clock>钟)?(?![{_ZH_NUMBER_CHARS}\d儿兒])"
)
_PERCENT_SYMBOL_RE = re.compile(rf"(?<![A-Za-z0-9_.,/:])(?P<n>{_ARABIC_NUMBER})\s*%")
_PERCENT_ZH_RE = re.compile(rf"百分之\s*(?P<n>{_EXPLICIT_NUMBER})")
_PERCENT_EN_RE = re.compile(rf"(?<![\w.])(?P<n>{_ARABIC_NUMBER})\s+percent\b", re.IGNORECASE)
_MONEY_SYMBOL_RE = re.compile(
    rf"(?P<currency>[¥￥$€£])\s*(?P<n>{_ARABIC_NUMBER})(?![\w.])"
)
_MONEY_ZH_RE = re.compile(
    rf"(?P<yuan>{_EXPLICIT_NUMBER})元"
    rf"(?:(?P<jiao>{_EXPLICIT_NUMBER})角)?"
    rf"(?:(?P<fen>{_EXPLICIT_NUMBER})分)?"
)
_MONEY_KUAI_RE = re.compile(
    rf"(?P<yuan>{_EXPLICIT_NUMBER})块"
    rf"(?:(?P<jiao>{_EXPLICIT_NUMBER})(?:毛|角))?钱"
)

_UNIT_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("千米", "km"),
    ("公里", "km"),
    ("kilometers", "km"),
    ("kilometer", "km"),
    ("厘米", "cm"),
    ("centimeters", "cm"),
    ("centimeter", "cm"),
    ("毫米", "mm"),
    ("millimeters", "mm"),
    ("millimeter", "mm"),
    ("公斤", "kg"),
    ("千克", "kg"),
    ("kilograms", "kg"),
    ("kilogram", "kg"),
    ("毫升", "ml"),
    ("milliliters", "ml"),
    ("milliliter", "ml"),
    ("摄氏度", "degc"),
    ("分钟", "min"),
    ("minutes", "min"),
    ("minute", "min"),
    ("小时", "h"),
    ("hours", "h"),
    ("hour", "h"),
    ("meters", "m"),
    ("meter", "m"),
    ("grams", "g"),
    ("gram", "g"),
    ("liters", "l"),
    ("liter", "l"),
    ("seconds", "s"),
    ("second", "s"),
    ("km", "km"),
    ("cm", "cm"),
    ("mm", "mm"),
    ("kg", "kg"),
    ("ml", "ml"),
    ("min", "min"),
    ("米", "m"),
    ("克", "g"),
    ("升", "l"),
    ("秒", "s"),
    ("°c", "degc"),
)
_UNIT_PATTERN = "|".join(re.escape(alias) for alias, _ in _UNIT_ALIASES)
_UNIT_MAP = {alias.lower(): canonical for alias, canonical in _UNIT_ALIASES}
_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z0-9_.,/:-])(?P<n>{_EXPLICIT_NUMBER})\s*(?P<unit>{_UNIT_PATTERN})(?![a-z])",
    re.IGNORECASE,
)
_FRACTION_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z0-9_.,/:-])(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*"
    rf"(?P<unit>{_UNIT_PATTERN})(?![a-z])",
    re.IGNORECASE,
)
_FRACTION_AR_RE = re.compile(
    r"(?<![A-Za-z0-9_.,/:-])(?P<num>\d+)\s*/\s*(?P<den>\d+)(?![A-Za-z0-9_.,/:-])"
)
_FRACTION_ZH_RE = re.compile(
    rf"(?P<den>{_ZH_DECIMAL}|{_ARABIC_NUMBER})分之(?P<num>{_ZH_DECIMAL}|{_ARABIC_NUMBER})"
)
_FRACTION_UNICODE_RE = re.compile(r"[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]")
_ORDINAL_ZH_RE = re.compile(rf"第\s*(?P<n>{_ZH_DECIMAL}|\d+)")
_ORDINAL_EN_RE = re.compile(r"(?<![A-Za-z0-9_])(?P<n>\d+)(?P<suffix>st|nd|rd|th)\b", re.IGNORECASE)
_GROUPED_DIGITS_RE = re.compile(r"(?<![\d,])(?P<n>\d+(?:,\d+)+)(?![\d,])")
_HYPHENATED_DIGITS_RE = re.compile(
    r"(?<![\d-])(?P<n>\d{2,4}(?:-\d{2,8})+)(?![\d-])"
)
_LONG_DIGITS_RE = re.compile(r"(?<![A-Za-z0-9_.,/:-])(?P<n>\d{7,})(?![A-Za-z0-9_.,/:-])")
_LEADING_ZERO_RE = re.compile(r"(?<![A-Za-z0-9_.,/:-])(0\d+)(?![A-Za-z0-9_.,/:-])")
_LABELED_ZH_DIGITS_RE = re.compile(
    rf"(?P<prefix>(?:编号|号码|电话|手机号|区号)(?:是|为)?\s*)"
    rf"(?P<n>[{_ZH_DIGIT_SEQUENCE_CHARS}](?:[\s-]*[{_ZH_DIGIT_SEQUENCE_CHARS}]){{2,}})"
    rf"(?![{_ZH_DIGIT_SEQUENCE_CHARS}])"
)
_ZH_UNIT_NUMBER_RE = re.compile(
    rf"[{_ZH_NUMBER_CHARS}]*[十拾百佰千仟万萬亿億][{_ZH_NUMBER_CHARS}]*"
)
_ARABIC_STANDALONE_RE = re.compile(
    rf"(?<![A-Za-z0-9_.,/:-])({_ARABIC_NUMBER})(?![A-Za-z0-9_.,/:-])"
)

_EN_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_DIGITS = {word: value for word, value in _EN_SMALL.items() if value < 10}
_EN_SPOKEN_DIGITS = {**_EN_DIGITS, "oh": 0}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_EN_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_EN_NUMBER_WORDS = tuple(
    sorted(
        set(_EN_SMALL) | set(_EN_TENS) | set(_EN_SCALES) | {"hundred", "and", "point"},
        key=len,
        reverse=True,
    )
)
_EN_NUMBER_ATOM = "(?:" + "|".join(re.escape(word) for word in _EN_NUMBER_WORDS) + ")"
_EN_NUMBER_PHRASE = rf"(?:minus[ -]+)?{_EN_NUMBER_ATOM}(?:[ -]+{_EN_NUMBER_ATOM})*"
_EN_NUMBER_PHRASE_RE = re.compile(
    rf"(?<![A-Za-z])(?P<n>{_EN_NUMBER_PHRASE})(?![A-Za-z])",
    re.IGNORECASE,
)
_EN_PERCENT_WORD_RE = re.compile(
    rf"(?<![A-Za-z])(?P<n>{_EN_NUMBER_PHRASE})[ -]+percent\b",
    re.IGNORECASE,
)
_EN_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z])(?P<n>{_EN_NUMBER_PHRASE})[ -]+(?P<unit>{_UNIT_PATTERN})(?![a-z])",
    re.IGNORECASE,
)
_EN_SPOKEN_DIGIT_WORD = "(?:" + "|".join(
    re.escape(word) for word in sorted(_EN_SPOKEN_DIGITS, key=len, reverse=True)
) + ")"
_LABELED_EN_DIGITS_RE = re.compile(
    rf"(?P<prefix>\b(?:phone number|telephone number|area code|number)"
    rf"(?:\s+(?:is|was))?\s+)"
    rf"(?P<n>(?:\d(?:[\s-]*\d){{2,}}|"
    rf"{_EN_SPOKEN_DIGIT_WORD}(?:[\s-]+{_EN_SPOKEN_DIGIT_WORD}){{2,}}))"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EN_ORDINAL_VALUES = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
    "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
    "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_EN_ORDINAL_WORD = "(?:" + "|".join(
    re.escape(word) for word in sorted(_EN_ORDINAL_VALUES, key=len, reverse=True)
) + ")"
_EN_ORDINAL_RE = re.compile(
    rf"(?<![A-Za-z])(?:(?P<tens>{'|'.join(_EN_TENS)})[ -]+)?"
    rf"(?P<ordinal>{_EN_ORDINAL_WORD})(?![A-Za-z])",
    re.IGNORECASE,
)
_EN_FRACTION_DENOMINATORS = {"half": 2, "halves": 2, "quarter": 4, "quarters": 4}
for _ordinal_word, _ordinal_value in _EN_ORDINAL_VALUES.items():
    if 3 <= _ordinal_value <= 20:
        _EN_FRACTION_DENOMINATORS[_ordinal_word] = _ordinal_value
        _EN_FRACTION_DENOMINATORS[f"{_ordinal_word}s"] = _ordinal_value
_EN_FRACTION_RE = re.compile(
    rf"(?<![A-Za-z])(?P<num>{_EN_NUMBER_PHRASE})[ -]+(?P<den>"
    + "|".join(
        re.escape(word)
        for word in sorted(_EN_FRACTION_DENOMINATORS, key=len, reverse=True)
    )
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)
_TIME_EN_OCLOCK_RE = re.compile(
    rf"(?<![A-Za-z])(?P<h>\d{{1,2}}|{'|'.join(_EN_SMALL)})\s+o['’]?clock"
    r"(?:\s*(?P<ampm>a\.?m\.?|p\.?m\.?))?(?![A-Za-z])",
    re.IGNORECASE,
)

_UNICODE_FRACTIONS = {
    "½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4),
    "⅕": (1, 5), "⅖": (2, 5), "⅗": (3, 5), "⅘": (4, 5),
    "⅙": (1, 6), "⅚": (5, 6), "⅛": (1, 8), "⅜": (3, 8),
    "⅝": (5, 8), "⅞": (7, 8),
}

_WIDTH_TRANSLATION = {0x3000: " "}
_WIDTH_TRANSLATION.update({code: chr(code - 0xFEE0) for code in range(0xFF01, 0xFF5F)})
_PUNCT_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "＇": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "：": ":",
        "％": "%",
        "／": "/",
        "．": ".",
    }
)


class _TokenStore:
    def __init__(self) -> None:
        self.tokens: List[str] = []

    def put(self, token: str) -> str:
        index = len(self.tokens)
        if index >= 0x1900:
            raise ValueError("too many canonical entities in one text")
        self.tokens.append(token)
        return chr(0xE000 + index)

    def protect_existing(self, text: str) -> str:
        return _CANONICAL_RE.sub(lambda match: self.put(match.group(0)), text)

    def restore(self, text: str) -> str:
        for index, token in enumerate(self.tokens):
            text = text.replace(chr(0xE000 + index), token)
        return text


def _decode_html_entities(text: str) -> str:
    """Decode only complete semicolon-terminated entities, once.

    Unknown, invalid, or control-producing entities are retained as typed
    evidence instead of leaking their alphabetic entity name into CER text.
    """
    def repl(match: Match[str]) -> str:
        raw = match.group(0)
        decoded = html.unescape(raw)
        if decoded == raw or "\ufffd" in decoded or any(
            unicodedata.category(char) in {"Cc", "Cs"} for char in decoded
        ):
            label = raw[1:-1].lower()
            return f"⟦HTML:{label}⟧"
        return decoded

    return _HTML_ENTITY_RE.sub(repl, text)


def reference_normalization_context(
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
    *,
    text: Optional[str] = None,
) -> dict[str, str]:
    """Return the canonical metadata that controls reference-only rules."""
    lang = (language or "").strip().lower()
    kind = (lang_type or "").strip().lower()
    if lang in {"en", "eng"}:
        tag_language = "en"
        source = "language"
    elif lang in {"zh", "zho", "cmn", "cn"}:
        tag_language = "zh"
        source = "language"
    elif kind == "pure_en":
        tag_language = "en"
        source = "lang_type"
    elif kind in {"pure_zh", "pure_cn", "cn_mostly", "frequent_mix"}:
        tag_language = "zh"
        source = "lang_type"
    else:
        visible = _SPEECH_TAG_RE.sub(" ", text or "")
        if re.search(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]", visible):
            tag_language = "zh"
            source = "text_cjk_fallback"
        elif re.search(r"[A-Za-z]", visible):
            tag_language = "en"
            source = "text_latin_fallback"
        else:
            tag_language = "unknown"
            source = "unknown"
    return {
        "language": lang or "unknown",
        "lang_type": kind or "unknown",
        "speech_tag_language": tag_language,
        "speech_tag_language_source": source,
    }


def replace_speech_tags(
    text: str,
    *,
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
) -> str:
    """Expand only whitelisted authoring tags using explicit metadata."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    context = reference_normalization_context(language, lang_type, text=text)
    table = (
        _TAG_TO_SPOKEN_EN
        if context["speech_tag_language"] == "en"
        else _TAG_TO_SPOKEN_ZH
        if context["speech_tag_language"] == "zh"
        else None
    )

    def repl(match: Match[str]) -> str:
        tag = match.group(0)
        spoken = table.get(tag) if table is not None else None
        if spoken is None:
            label = tag[1:-1]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                label = "sha256-" + hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16]
            return f" ⟦TAG:{label.lower()}⟧ "
        return f" {spoken} " if spoken else " "

    return re.sub(r"\s+", " ", _SPEECH_TAG_RE.sub(repl, text)).strip()


def reference_normalization_input_fingerprint(
    text: str,
    *,
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    payload = {
        "text": text,
        "context": reference_normalization_context(language, lang_type, text=text),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _unicode_and_width(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_WIDTH_TRANSLATION).translate(_PUNCT_TRANSLATION)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    text = text.replace("\u00a0", " ")
    return text.lower()


def _chinese_integer(raw: str, *, digit_sequence: bool = False) -> Optional[int]:
    if not raw or any(
        char not in _ZH_DIGIT_VALUES and char not in _ZH_SMALL_UNITS and char not in _ZH_LARGE_UNITS
        for char in raw
    ):
        return None
    if not any(char in _ZH_SMALL_UNITS or char in _ZH_LARGE_UNITS for char in raw):
        if not digit_sequence:
            return _ZH_DIGIT_VALUES.get(raw) if len(raw) == 1 else None
        return int("".join(str(_ZH_DIGIT_VALUES[char]) for char in raw))

    # 万亿 is a conventional compound unit (10^12), not a descending pair of
    # section delimiters.  Handle it explicitly before the normal section walk.
    compound = next((marker for marker in ("万亿", "万億", "萬亿", "萬億") if marker in raw), None)
    if compound is not None:
        if raw.count(compound) != 1:
            return None
        left, right = raw.split(compound)
        left_value = 1 if not left else _chinese_integer(left, digit_sequence=digit_sequence)
        right_value = 0 if not right else _chinese_integer(right, digit_sequence=digit_sequence)
        if left_value is None or right_value is None or right_value >= 10**12:
            return None
        return left_value * 10**12 + right_value

    total = 0
    section = 0
    pending: Optional[int] = None
    last_small_unit = 10_000
    last_large_unit = 10**20
    for char in raw:
        if char in _ZH_DIGIT_VALUES:
            if pending is not None and pending != 0:
                return None
            pending = _ZH_DIGIT_VALUES[char]
            continue
        if char in _ZH_SMALL_UNITS:
            unit = _ZH_SMALL_UNITS[char]
            if unit >= last_small_unit:
                return None
            section += (1 if pending is None else pending) * unit
            pending = None
            last_small_unit = unit
            continue
        unit = _ZH_LARGE_UNITS[char]
        if unit >= last_large_unit:
            return None
        section += 0 if pending is None else pending
        if section == 0:
            section = 1
        total += section * unit
        section = 0
        pending = None
        last_small_unit = 10_000
        last_large_unit = unit
    return total + section + (0 if pending is None else pending)


def _number_string(raw: str, *, digit_sequence: bool = True) -> Optional[str]:
    raw = re.sub(r"[\s,]", "", raw)
    if re.fullmatch(_ARABIC_NUMBER, raw):
        try:
            return _format_decimal(Decimal(raw))
        except InvalidOperation:
            return None
    if "点" in raw:
        if raw.count("点") != 1:
            return None
        left, right = raw.split("点")
        integer = _chinese_integer(left, digit_sequence=digit_sequence)
        if integer is None or not right or any(char not in _ZH_DIGIT_VALUES for char in right):
            return None
        fraction = "".join(str(_ZH_DIGIT_VALUES[char]) for char in right)
        return _format_decimal(Decimal(f"{integer}.{fraction}"))
    integer = _chinese_integer(raw, digit_sequence=digit_sequence)
    return None if integer is None else str(integer)


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise InvalidOperation
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _replace_valid(
    text: str,
    pattern: re.Pattern[str],
    store: _TokenStore,
    builder: Callable[[Match[str]], Optional[str]],
) -> str:
    def repl(match: Match[str]) -> str:
        token = builder(match)
        return match.group(0) if token is None else store.put(token)

    return pattern.sub(repl, text)


def _date_token(year: int, month: int, day: int) -> Optional[str]:
    try:
        date = _datetime.date(year, month, day)
    except ValueError:
        return None
    return f"⟦DATE:{date:%Y-%m-%d}⟧"


def _date_ar_builder(match: Match[str]) -> Optional[str]:
    return _date_token(int(match.group("y")), int(match.group("m")), int(match.group("d")))


def _date_zh_builder(match: Match[str]) -> Optional[str]:
    values = [_number_string(match.group(name), digit_sequence=True) for name in ("y", "m", "d")]
    if any(value is None for value in values):
        return None
    return _date_token(*(int(value) for value in values if value is not None))


def _adjust_hour(hour: int, period: Optional[str]) -> Optional[int]:
    period = period.lower().replace(".", "") if period else None
    if period in {"am", "pm"}:
        if not 1 <= hour <= 12:
            return None
        if period == "am":
            return 0 if hour == 12 else hour
        return 12 if hour == 12 else hour + 12
    if not 0 <= hour <= 23:
        return None
    if period in ("凌晨",):
        return 0 if hour == 12 else hour if hour <= 11 else None
    if period in ("早上", "上午"):
        return hour if hour <= 11 else None
    if period == "中午":
        return 12 if hour == 12 else hour if hour == 11 else None
    if period in ("下午", "傍晚", "晚上"):
        if hour == 12:
            return 12
        return hour + 12 if 1 <= hour <= 11 else None
    return hour


def _time_token(hour: int, minute: int, period: Optional[str] = None) -> Optional[str]:
    hour = _adjust_hour(hour, period)
    if hour is None or not 0 <= minute <= 59:
        return None
    return f"⟦TIME:{hour:02d}:{minute:02d}⟧"


def _time_ar_builder(match: Match[str]) -> Optional[str]:
    prefix = match.string[max(0, match.start() - 12) : match.start()].lower()
    if re.search(
        r"(?:比分(?:为)?|score|ratio|version)\s*[\(\[（【]?\s*$", prefix
    ):
        return None
    period = match.groupdict().get("period")
    ampm = match.groupdict().get("ampm")
    if period and ampm:
        return None
    return _time_token(int(match.group("h")), int(match.group("m")), period or ampm)


def _time_zh_builder(match: Match[str]) -> Optional[str]:
    groups = match.groupdict()
    if "clock" in groups and not groups.get("period") and not groups.get("clock"):
        return None
    hour = _number_string(match.group("h"), digit_sequence=True)
    minute = "30" if groups.get("half") else _number_string(groups.get("m") or "0")
    if hour is None or minute is None:
        return None
    return _time_token(int(hour), int(minute), groups.get("period"))


def _time_zh_quarter_builder(match: Match[str]) -> Optional[str]:
    hour = _number_string(match.group("h"), digit_sequence=True)
    if hour is None:
        return None
    minute = 15 if match.group("quarter") == "一刻" else 45
    return _time_token(int(hour), minute, match.groupdict().get("period"))


def _time_en_oclock_builder(match: Match[str]) -> Optional[str]:
    raw_hour = match.group("h").lower()
    hour = int(raw_hour) if raw_hour.isdigit() else _EN_SMALL.get(raw_hour)
    if hour is None:
        return None
    return _time_token(hour, 0, match.groupdict().get("ampm"))


def _percent_builder(match: Match[str]) -> Optional[str]:
    value = _number_string(match.group("n"), digit_sequence=True)
    return None if value is None else f"⟦PERCENT:{value}⟧"


def _money_symbol_builder(match: Match[str]) -> Optional[str]:
    currencies = {"¥": "CNY", "￥": "CNY", "$": "USD", "€": "EUR", "£": "GBP"}
    value = _number_string(match.group("n"))
    return None if value is None else f"⟦MONEY:{currencies[match.group('currency')]}:{value}⟧"


def _money_zh_builder(match: Match[str]) -> Optional[str]:
    yuan = _number_string(match.group("yuan"), digit_sequence=True)
    jiao = _number_string(match.groupdict().get("jiao") or "0", digit_sequence=True)
    fen = _number_string(match.groupdict().get("fen") or "0", digit_sequence=True)
    if yuan is None or jiao is None or fen is None:
        return None
    if not (0 <= int(jiao) <= 9 and 0 <= int(fen) <= 9):
        return None
    value = Decimal(yuan) + Decimal(jiao) / 10 + Decimal(fen) / 100
    return f"⟦MONEY:CNY:{_format_decimal(value)}⟧"


def _quantity_builder(match: Match[str]) -> Optional[str]:
    value = _number_string(match.group("n"), digit_sequence=True)
    unit = _UNIT_MAP.get(match.group("unit").lower())
    return None if value is None or unit is None else f"⟦QTY:{value}:{unit}⟧"


def _fraction_token(numerator: int, denominator: int) -> Optional[str]:
    if denominator == 0:
        return None
    value = Fraction(numerator, denominator)
    if value.denominator < 0:
        value = -value
    return f"⟦FRACTION:{value.numerator}/{value.denominator}⟧"


def _fraction_quantity_builder(match: Match[str]) -> Optional[str]:
    denominator = int(match.group("den"))
    if denominator == 0:
        return None
    value = Fraction(int(match.group("num")), denominator)
    unit = _UNIT_MAP.get(match.group("unit").lower())
    return None if unit is None else f"⟦QTY:{value.numerator}/{value.denominator}:{unit}⟧"


def _fraction_zh_builder(match: Match[str]) -> Optional[str]:
    numerator = _number_string(match.group("num"), digit_sequence=True)
    denominator = _number_string(match.group("den"), digit_sequence=True)
    if numerator is None or denominator is None:
        return None
    try:
        num_decimal, den_decimal = Decimal(numerator), Decimal(denominator)
    except InvalidOperation:
        return None
    if num_decimal != int(num_decimal) or den_decimal != int(den_decimal):
        return None
    return _fraction_token(int(num_decimal), int(den_decimal))


def _fraction_ar_builder(match: Match[str]) -> Optional[str]:
    numerator_raw = match.group("num")
    denominator_raw = match.group("den")
    prefix = match.string[max(0, match.start() - 12) : match.start()].lower()
    if (
        (len(numerator_raw) > 1 and numerator_raw.startswith("0"))
        or (len(denominator_raw) > 1 and denominator_raw.startswith("0"))
        or re.search(
            r"(?:比分(?:为)?|score|ratio|version)\s*[\(\[（【]?\s*$", prefix
        )
    ):
        return None
    return _fraction_token(int(numerator_raw), int(denominator_raw))


def _fraction_unicode_builder(match: Match[str]) -> Optional[str]:
    numerator, denominator = _UNICODE_FRACTIONS[match.group(0)]
    return _fraction_token(numerator, denominator)


def _ordinal_zh_builder(match: Match[str]) -> Optional[str]:
    value = _number_string(match.group("n"), digit_sequence=True)
    return None if value is None or "." in value else f"⟦ORDINAL:{value}⟧"


def _ordinal_en_builder(match: Match[str]) -> Optional[str]:
    value = int(match.group("n"))
    suffix = match.group("suffix").lower()
    expected = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"⟦ORDINAL:{value}⟧" if suffix == expected else None


def _parse_english_cardinal(raw: str) -> Optional[int]:
    tokens = [token for token in re.split(r"[ -]+", raw.lower().strip()) if token]
    if not tokens:
        return None
    if tokens[0] == "minus":
        sign = -1
        tokens = tokens[1:]
    else:
        sign = 1
    if not tokens or "point" in tokens:
        return None
    if tokens[0] == "and" or tokens[-1] == "and" or any(
        left == right == "and" for left, right in zip(tokens, tokens[1:])
    ):
        return None
    tokens = [token for token in tokens if token != "and"]
    total = 0
    current = 0
    last_scale = 10**15
    last_kind: Optional[str] = None
    for token in tokens:
        if token in _EN_SMALL:
            value = _EN_SMALL[token]
            if last_kind in {"small", "zero"}:
                return None
            if last_kind == "tens" and value >= 10:
                return None
            current += value
            last_kind = "zero" if value == 0 else "small"
        elif token in _EN_TENS:
            if last_kind in {"small", "zero", "tens"}:
                return None
            current += _EN_TENS[token]
            last_kind = "tens"
        elif token == "hundred":
            if last_kind != "small" or not 1 <= current % 100 <= 9:
                return None
            current *= 100
            last_kind = "hundred"
        elif token in _EN_SCALES:
            scale = _EN_SCALES[token]
            if current <= 0 or scale >= last_scale:
                return None
            total += current * scale
            current = 0
            last_scale = scale
            last_kind = "scale"
        else:
            return None
    return sign * (total + current)


def _english_number_token(raw: str) -> Optional[str]:
    tokens = [token for token in re.split(r"[ -]+", raw.lower().strip()) if token]
    sign = ""
    if tokens and tokens[0] == "minus":
        sign, tokens = "-", tokens[1:]
    if not tokens:
        return None
    if all(token in _EN_DIGITS for token in tokens) and len(tokens) > 1:
        return " ".join(f"⟦NUM:{_EN_DIGITS[token]}⟧" for token in tokens)
    if "point" in tokens:
        if tokens.count("point") != 1:
            return None
        index = tokens.index("point")
        left = _parse_english_cardinal(" ".join(tokens[:index]))
        right = tokens[index + 1 :]
        if left is None or not right or not all(token in _EN_DIGITS for token in right):
            return None
        fraction = "".join(str(_EN_DIGITS[token]) for token in right)
        value = Decimal(f"{left}.{fraction}")
        if sign:
            value = -value
        return f"⟦NUM:{_format_decimal(value)}⟧"
    value = _parse_english_cardinal((sign + " " if sign else "") + " ".join(tokens))
    return None if value is None else f"⟦NUM:{value}⟧"


def _english_number_builder(match: Match[str]) -> Optional[str]:
    return _english_number_token(match.group("n"))


def _english_percent_builder(match: Match[str]) -> Optional[str]:
    token = _english_number_token(match.group("n"))
    if token is None or not token.startswith("⟦NUM:") or " ⟦" in token:
        return None
    return token.replace("⟦NUM:", "⟦PERCENT:", 1)


def _english_quantity_builder(match: Match[str]) -> Optional[str]:
    token = _english_number_token(match.group("n"))
    unit = _UNIT_MAP.get(match.group("unit").lower())
    if token is None or unit is None or not token.startswith("⟦NUM:") or " ⟦" in token:
        return None
    value = token[len("⟦NUM:") : -1]
    return f"⟦QTY:{value}:{unit}⟧"


def _english_ordinal_builder(match: Match[str]) -> Optional[str]:
    ordinal_word = match.group("ordinal").lower()
    value = _EN_ORDINAL_VALUES[ordinal_word]
    tens_word = match.groupdict().get("tens")
    if tens_word:
        if value >= 10:
            return None
        value += _EN_TENS[tens_word.lower()]
    full_prefix = match.string[max(0, match.start() - 8) : match.start()].lower()
    if re.search(r"\b(?:a|an)\s+$", full_prefix):
        return None
    if ordinal_word in {"first", "second"} and not tens_word:
        prefix = match.string[max(0, match.start() - 5) : match.start()].lower()
        suffix = match.string[match.end() : match.end() + 16].lower()
        explicit_prefix = bool(
            re.search(r"(?:the|was|am|is|are|came)\s+$", prefix)
        )
        explicit_suffix = bool(re.match(
            r"\s+(?:speaker|question|part|round|grade|chapter|place|prize|time)\b",
            suffix,
        ))
        if not explicit_prefix and not explicit_suffix:
            return None
    return f"⟦ORDINAL:{value}⟧"


def _english_fraction_builder(match: Match[str]) -> Optional[str]:
    numerator = _parse_english_cardinal(match.group("num"))
    denominator = _EN_FRACTION_DENOMINATORS[match.group("den").lower()]
    return None if numerator is None else _fraction_token(numerator, denominator)


def _labeled_zh_digits_builder(match: Match[str]) -> Optional[str]:
    raw = re.sub(r"[\s-]", "", match.group("n"))
    if any(char not in _ZH_DIGIT_VALUES for char in raw):
        return None
    digits = "".join(str(_ZH_DIGIT_VALUES[char]) for char in raw)
    prefix = re.sub(r"\s+", "", match.group("prefix"))
    return f"{prefix}⟦DIGITS:{digits}⟧"


def _labeled_en_digits_builder(match: Match[str]) -> Optional[str]:
    raw = match.group("n").lower()
    if re.fullmatch(r"\d(?:[\s-]*\d){2,}", raw):
        digits = re.sub(r"[\s-]", "", raw)
    else:
        words = [word for word in re.split(r"[\s-]+", raw) if word]
        if len(words) < 3 or any(word not in _EN_SPOKEN_DIGITS for word in words):
            return None
        digits = "".join(str(_EN_SPOKEN_DIGITS[word]) for word in words)
    prefix = re.sub(r"\s+", " ", match.group("prefix").lower()).strip()
    return f"{prefix} ⟦DIGITS:{digits}⟧"


def _general_zh_builder(match: Match[str]) -> Optional[str]:
    raw = match.group(0)
    following = match.string[match.end() : match.end() + 2]
    # A bare unit glyph is highly polysemous in normal prose (e.g. 十分,
    # 百分之).  Safe mode only promotes a multi-character cardinal here;
    # explicit date/time/money/unit contexts above still accept one glyph.
    if len(raw) < 2:
        return None
    if raw in {"万一", "萬一"} or (raw in {"千万", "千萬"} and following.startswith(("别", "不", "要"))):
        return None
    value = _number_string(raw, digit_sequence=False)
    return None if value is None else f"⟦NUM:{value}⟧"


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
    )


def _is_letter(char: str) -> bool:
    return bool(char) and unicodedata.category(char).startswith("L")


def _surface_cleanup(text: str) -> str:
    output: List[str] = []
    length = len(text)
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < length else ""
        category = unicodedata.category(char)

        if 0xE000 <= ord(char) <= 0xF8FF or category[0] in ("L", "N", "M"):
            output.append(char)
        elif char.isspace():
            output.append(" ")
        elif char in ("%", "¥", "$", "€", "£", "°", "²", "³"):
            output.append(char)
        elif char in (".", ":") and previous.isdigit() and following.isdigit():
            output.append(char)
        elif char == "/" and (
            (previous.isdigit() and following.isdigit()) or (_is_letter(previous) and _is_letter(following))
        ):
            output.append(char)
        elif char == "-" and previous.isalnum() and following.isalnum():
            output.append(char)
        elif char == "'" and _is_letter(previous) and _is_letter(following):
            output.append(char)
        elif char in ("+", "-") and following.isdigit() and (not previous or previous.isspace()):
            output.append(char)
        else:
            output.append(" ")

    cleaned = re.sub(r"\s+", " ", "".join(output)).strip()
    # CER treats spacing around CJK as formatting rather than lexical evidence.
    cleaned = re.sub(r"([\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff])", r"\1", cleaned)
    return cleaned


def normalize_strict(text: str) -> str:
    """Apply low-risk surface normalization without semantic rewrites."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    store = _TokenStore()
    normalized = _unicode_and_width(store.protect_existing(_decode_html_entities(text)))
    return store.restore(_surface_cleanup(normalized))


def _normalize_safe_once(text: str) -> str:
    """Run one conservative entity-and-surface normalization pass."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    store = _TokenStore()
    normalized = _unicode_and_width(store.protect_existing(_decode_html_entities(text)))

    for pattern, builder in (
        (_DATE_AR_RE, _date_ar_builder),
        (_DATE_ZH_RE, _date_zh_builder),
        (_TIME_AR_RE, _time_ar_builder),
        (_TIME_EN_OCLOCK_RE, _time_en_oclock_builder),
        (_TIME_ZH_DETAIL_RE, _time_zh_builder),
        (_TIME_ZH_PERIOD_MINUTE_RE, _time_zh_builder),
        (_TIME_ZH_ZERO_MINUTE_RE, _time_zh_builder),
        (_TIME_ZH_QUARTER_RE, _time_zh_quarter_builder),
        (_TIME_ZH_HOUR_RE, _time_zh_builder),
        (_PERCENT_SYMBOL_RE, _percent_builder),
        (_PERCENT_ZH_RE, _percent_builder),
        (_PERCENT_EN_RE, _percent_builder),
        (_EN_PERCENT_WORD_RE, _english_percent_builder),
        (_MONEY_SYMBOL_RE, _money_symbol_builder),
        (_MONEY_ZH_RE, _money_zh_builder),
        (_MONEY_KUAI_RE, _money_zh_builder),
        (_FRACTION_QUANTITY_RE, _fraction_quantity_builder),
        (_FRACTION_AR_RE, _fraction_ar_builder),
        (_QUANTITY_RE, _quantity_builder),
        (_EN_QUANTITY_RE, _english_quantity_builder),
        (_FRACTION_ZH_RE, _fraction_zh_builder),
        (_FRACTION_UNICODE_RE, _fraction_unicode_builder),
        (_EN_FRACTION_RE, _english_fraction_builder),
        (_ORDINAL_ZH_RE, _ordinal_zh_builder),
        (_ORDINAL_EN_RE, _ordinal_en_builder),
        (_EN_ORDINAL_RE, _english_ordinal_builder),
        (_LABELED_ZH_DIGITS_RE, _labeled_zh_digits_builder),
        (_LABELED_EN_DIGITS_RE, _labeled_en_digits_builder),
        (
            _GROUPED_DIGITS_RE,
            lambda match: None
            if re.fullmatch(_ARABIC_INTEGER, match.group("n"))
            else f"⟦DIGITS:{match.group('n')}⟧",
        ),
        (
            _HYPHENATED_DIGITS_RE,
            lambda match: f"⟦DIGITS:{match.group('n').replace('-', '')}⟧",
        ),
        (_LONG_DIGITS_RE, lambda match: f"⟦DIGITS:{match.group('n')}⟧"),
        (_LEADING_ZERO_RE, lambda match: f"⟦DIGITS:{match.group(1)}⟧"),
        (_ZH_UNIT_NUMBER_RE, _general_zh_builder),
        (_EN_NUMBER_PHRASE_RE, _english_number_builder),
    ):
        normalized = _replace_valid(normalized, pattern, store, builder)

    normalized = _ARABIC_STANDALONE_RE.sub(
        lambda match: store.put(f"⟦NUM:{_number_string(match.group(1))}⟧"),
        normalized,
    )
    return store.restore(_surface_cleanup(normalized))


def normalize_safe(text: str) -> str:
    """Normalize explicit entities to a stable semantic representation.

    Surface punctuation can safely expose an entity that was deliberately not
    matched while it touched ambiguous punctuation, for example ``(3/7, -2)``.
    Iterate a small, fixed number of times so the first public call is already
    idempotent.  Failure to converge is an error rather than silent drift.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    current = text
    for _ in range(4):
        normalized = _normalize_safe_once(current)
        if normalized == current:
            return normalized
        current = normalized
    raise ValueError("safe normalization did not converge within four passes")


def normalize_text(text: str, profile: str = SAFE_PROFILE) -> str:
    """Normalize ``text`` with the named profile (``strict`` or ``safe``)."""
    if profile == STRICT_PROFILE:
        return normalize_strict(text)
    if profile == SAFE_PROFILE:
        return normalize_safe(text)
    raise ValueError(f"unknown normalization profile: {profile!r}")


def normalize_reference(
    text: str,
    profile: str = SAFE_PROFILE,
    *,
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
) -> str:
    """Normalize a reference independently using authoring metadata only."""
    spoken = replace_speech_tags(text, language=language, lang_type=lang_type)
    return normalize_text(spoken, profile)


def normalize_hypothesis(text: str, profile: str = SAFE_PROFILE) -> str:
    """Normalize a hypothesis independently; never inspect a reference."""
    return normalize_text(text, profile)


def normalization_fingerprint(profile: str = SAFE_PROFILE) -> str:
    if profile not in PROFILE_VERSIONS:
        raise ValueError(f"unknown normalization profile: {profile!r}")
    payload = {
        "profile": profile,
        "version": NORMALIZATION_VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "PROFILE_VERSIONS",
    "CER_METRIC",
    "EVAL_SCHEMA_VERSION",
    "CER_SCORE_VERSION",
    "NORMALIZATION_VERSION",
    "SAFE_PROFILE",
    "STRICT_PROFILE",
    "normalize_safe",
    "normalize_strict",
    "normalize_text",
    "normalize_reference",
    "normalize_hypothesis",
    "reference_normalization_context",
    "reference_normalization_input_fingerprint",
    "replace_speech_tags",
    "normalization_fingerprint",
]
