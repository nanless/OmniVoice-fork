"""Frequency suppression hints for adult generation."""

import re
from collections import Counter
from typing import Dict, List


def build_suppression_hint(texts: List[Dict], window_size: int = 500) -> str:
    if len(texts) < 40:
        return ""
    recent = texts[-window_size:]
    opening_counter = Counter()
    for item in recent:
        text = re.sub(r"\[[^\]]+\]", "", item.get("text", ""))
        if len(text) >= 4:
            opening_counter[text[:4]] += 1
    hot = [k for k, v in opening_counter.most_common(8) if v >= max(4, len(recent) // 80)]
    if not hot:
        return ""
    return (
        "=== 最近高频开头抑制 ===\n"
        + "\n".join(f"- 避免以「{o}」开头" for o in hot[:5])
        + "\n"
    )
