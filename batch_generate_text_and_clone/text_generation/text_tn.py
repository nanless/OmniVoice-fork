#!/usr/bin/env python3
"""Build ASR/WER reference text (TN) from OmniVoice tagged child speech prompts."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from omnivoice.eval.wer.text_norm_omni import text_normalize
except Exception:
    _FALLBACK_PUNCT = re.compile(
        r'[，。？！、；：\u201c\u201d\u2018\u2019【】（）《》'
        r',.?!;:"\x27()\[\]{}<>~`@#$^&*_+=|\\]'
    )

    def text_normalize(text, iso_code="*", lower_case=True, remove_numbers=False, remove_brackets=False):
        if lower_case:
            text = text.lower()
        text = _FALLBACK_PUNCT.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

_SPEECH_TAG_RE = re.compile(r"\[[^\]]+\]")
_CJK_RANGE = r"\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f"

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


def replace_speech_tags(text: str, lang_en: bool = False) -> str:
    """Replace OmniVoice non-verbal tags with ASR-expected spoken text."""
    table = _TAG_TO_SPOKEN_EN if lang_en else _TAG_TO_SPOKEN_ZH

    def _repl(m: re.Match) -> str:
        word = table.get(m.group(), "")
        if not word:
            return " "
        return f" {word} "

    cleaned = _SPEECH_TAG_RE.sub(_repl, text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_speech_tags(text: str) -> str:
    """Remove OmniVoice non-verbal tags; keep spoken words only."""
    cleaned = _SPEECH_TAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_cjk_spaces(text: str) -> str:
    """Remove spaces between CJK characters (align with omnivoice WER eval)."""
    text = re.sub(rf"([{_CJK_RANGE}])\s+([{_CJK_RANGE}])", r"\1\2", text)
    text = re.sub(rf"([{_CJK_RANGE}])\s+", r"\1", text)
    text = re.sub(rf"\s+([{_CJK_RANGE}])", r"\1", text)
    return text


def _tn_iso_code(language: Optional[str], lang_type: Optional[str]) -> str:
    lang = (language or "").strip().lower()
    lt = (lang_type or "").strip().lower()
    if lang in ("en", "eng") or lt in ("pure_en",):
        return "eng"
    return "*"


def build_text_tn(
    text: str,
    *,
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
) -> str:
    """
    Normalized transcript for ASR reference / WER.
    Strips paralinguistic tags, then applies omnivoice text_normalize.
    """
    iso_code = _tn_iso_code(language, lang_type)
    spoken = replace_speech_tags(text, lang_en=(iso_code == "eng"))
    if not spoken:
        return ""
    if iso_code == "eng":
        spoken = re.sub(r"(\d+(?:\.\d+)?)%", r"\1 percent", spoken)
    else:
        spoken = re.sub(r"(\d+(?:\.\d+)?)%", r"百分之\1", spoken)
    tn = text_normalize(
        spoken,
        iso_code=iso_code,
        lower_case=True,
        remove_numbers=False,
        remove_brackets=False,
    )
    tn = unicodedata.normalize("NFKC", tn)
    tn = clean_cjk_spaces(tn)
    tn = re.sub(r"\s+", " ", tn).strip()
    return tn


def attach_text_tn(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add or refresh ``text_tn`` on a generated JSONL record."""
    text = (item.get("text") or "").strip()
    if not text:
        item.pop("text_tn", None)
        return item
    item["text_tn"] = build_text_tn(
        text,
        language=item.get("language"),
        lang_type=item.get("lang_type") or item.get("lang_key"),
    )
    return item


def attach_text_tn_batch(items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    for item in items:
        attach_text_tn(item)
    return items
