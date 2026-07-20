#!/usr/bin/env python3
"""Build ASR/WER reference text (TN) from OmniVoice tagged child speech prompts."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batch_generate_text_and_clone.eval_cer.cer_normalization import (
    normalize_safe,
    replace_speech_tags as _shared_replace_speech_tags,
)

_SPEECH_TAG_RE = re.compile(r"\[[^\]]+\]")
_CJK_RANGE = r"\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f"

def replace_speech_tags(text: str, lang_en: bool = False) -> str:
    """Backward-compatible wrapper around the shared whitelist."""
    return _shared_replace_speech_tags(text, language="en" if lang_en else "zh")


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


def build_text_tn(
    text: str,
    *,
    language: Optional[str] = None,
    lang_type: Optional[str] = None,
) -> str:
    """
    Normalized transcript for ASR reference / WER.

    Paralinguistic tags are first converted to their configured spoken form;
    all remaining normalization uses the shared conservative CER safe profile.
    """
    spoken = _shared_replace_speech_tags(
        text, language=language, lang_type=lang_type
    )
    if not spoken:
        return ""
    return normalize_safe(spoken)


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
