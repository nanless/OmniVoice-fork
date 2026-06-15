"""Quality filter for adult OmniVoice tagged texts."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .scenarios import LENGTH_BOUNDS
from .tags import TAG_DEFINITIONS, VALID_TAG_NAMES

_CHILD_GEN = Path(__file__).resolve().parents[2] / "text_generation"
if str(_CHILD_GEN) not in sys.path:
    sys.path.insert(0, str(_CHILD_GEN))

from llm_generate_texts import (  # noqa: E402
    _auto_correct_tags,
    _contains_pinyin_or_phonetic_markup,
    _find_invalid_tags,
    _is_complete_for_asr,
    _validate_language_content,
    _validate_tag_combinations,
    _validate_tag_position,
)

BAD_MARKERS = [
    "股票", "投资", "政治", "战争", "sex", "kill", "die", "porn",
    "no cap", "yyds", "绝绝子", " emo", "emo了",
    " um", " uh", "ugh", "soooo", "sooo ",
    "吃饭饭", "宝宝要", "脑斧", "灰机",
]

_CHILD_MARKERS = ["饭饭", "水水", "觉觉", "要要要", "宝宝不", "明明要", "七饭"]


def _count_tags(text: str) -> Tuple[int, List[str]]:
    pattern = re.compile(rf"\[({VALID_TAG_NAMES})\]")
    tags = pattern.findall(text)
    return len(tags), tags


def _count_speech_units(text: str) -> int:
    clean = re.sub(r"\[[^\]]+\]", "", text).strip()
    cjk = len(re.findall(r"[\u4e00-\u9fff]", clean))
    en = len(re.findall(r"[A-Za-z]+", clean))
    return cjk + en


def _validate_length(text: str, length_type: str) -> bool:
    total = _count_speech_units(text)
    lo, hi = LENGTH_BOUNDS.get(length_type, (0, 9999))
    return lo * 0.35 <= total <= hi * 1.5 + 10


def quality_filter(
    texts: List[Dict],
    max_tags_per_text: int = 4,
    max_same_tag_repeat: int = 2,
) -> List[Dict]:
    filtered: List[Dict] = []
    for item in texts:
        text = item.get("text", "").strip()
        if not text:
            continue
        lower = text.lower()
        if any(m in lower for m in BAD_MARKERS):
            continue
        if any(m in text for m in _CHILD_MARKERS):
            continue

        if item.get("lang_type") not in {
            "pure_cn", "pure_en", "cn_mostly", "en_mostly", "frequent_mix",
        }:
            continue
        if not _validate_language_content(item, text):
            continue
        if _find_invalid_tags(text):
            continue
        if not _is_complete_for_asr(text):
            continue
        if _contains_pinyin_or_phonetic_markup(text):
            continue

        tag_count, tags = _count_tags(text)
        if tag_count > max_tags_per_text:
            continue
        if any(tags.count(t) > max_same_tag_repeat for t in set(tags)):
            continue

        length_type = item.get("length_type", "medium")
        if not _validate_length(text, length_type):
            continue

        text = _auto_correct_tags(text, item.get("emotion", ""))
        item = dict(item)
        item["text"] = text
        tag_count, tags = _count_tags(text)

        pos_issues = _validate_tag_position(text)
        combo_issues = _validate_tag_combinations(text)
        if len(pos_issues) + len(combo_issues) > 5:
            continue

        try:
            from text_tn import attach_text_tn  # noqa: WPS433
        except ImportError:
            sys.path.insert(0, str(_CHILD_GEN))
            from text_tn import attach_text_tn  # noqa: WPS433

        attach_text_tn(item)
        if not item.get("text_tn"):
            continue

        item["tag_count"] = tag_count
        item["tags_used"] = tags
        item["age_tier"] = item.get("age_tier", "adult")
        filtered.append(item)
    return filtered
