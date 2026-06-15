"""Output helpers and QA statistics."""

import hashlib
import json
from collections import Counter
from typing import Dict, List, Optional

from .config import GenConfig
from .quality_filter import quality_filter as _quality_filter


def save_jsonl(texts: List[Dict], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in texts:
            clean = {k: v for k, v in item.items() if not str(k).startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def ensure_text_ids(texts: List[Dict]) -> None:
    for item in texts:
        if item.get("id"):
            continue
        identity = {
            "text": item.get("text", ""),
            "lang_type": item.get("lang_type"),
            "scenario": item.get("scenario"),
            "task_id": item.get("task_id"),
            "age_tier": "adult",
        }
        digest = hashlib.md5(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        item["id"] = f"adult_{digest}"


def generation_quality_report(texts: List[Dict], config: Optional[GenConfig] = None) -> Dict:
    total = len(texts)
    with_tags = sum(1 for t in texts if t.get("tags_used") or t.get("tag_count", 0) > 0)
    report = {
        "total": total,
        "by_length": dict(Counter(t.get("length_type", "unknown") for t in texts)),
        "by_lang_type": dict(Counter(t.get("lang_type", "unknown") for t in texts)),
        "by_scenario": dict(Counter(t.get("scenario", "unknown") for t in texts)),
        "by_emotion": dict(Counter(t.get("emotion", "unknown") for t in texts)),
        "tag_coverage": with_tags / total if total else 0.0,
        "empty_text_tn": sum(1 for t in texts if not t.get("text_tn")),
        "warnings": [],
    }
    if config and total < config.total_target:
        report["warnings"].append(f"below_target:{total}/{config.total_target}")
    return report


def print_generation_quality_report(report: Dict) -> None:
    print("\n=== Adult Generation QA ===")
    print(f"Total: {report['total']}")
    print(f"Length: {report['by_length']}")
    print(f"Lang type: {report['by_lang_type']}")
    print(f"Scenario: {report['by_scenario']}")
    print(f"Emotion: {report['by_emotion']}")
    print(f"Tag coverage: {report['tag_coverage'] * 100:.1f}%")
    print(f"Empty text_tn: {report['empty_text_tn']}")
    if report["warnings"]:
        print(f"Warnings: {', '.join(report['warnings'])}")
    else:
        print("Warnings: none")


def print_statistics(texts: List[Dict]) -> None:
    print("\n=== Distribution ===")
    for field in ("length_type", "lang_type", "scenario", "emotion"):
        counter = Counter(t.get(field, "unknown") for t in texts)
        print(f"{field}: {dict(counter)}")


def diagnose_rejections(texts: List[Dict]) -> Counter:
    """Count how many items fail quality filter (for tuning)."""
    passed = {id(t) for t in _quality_filter(texts)}
    reasons = Counter()
    for item in texts:
        if id(item) in passed:
            reasons["pass"] += 1
        else:
            reasons["rejected"] += 1
    return reasons
