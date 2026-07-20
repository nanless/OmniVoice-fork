#!/usr/bin/env python3
"""Read-only corpus audit for deterministic reference normalization."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from cer_normalization import (
    normalize_reference,
    reference_normalization_context,
)

_TOKEN_RE = re.compile(r"⟦(?P<kind>[A-Z]+):[^⟦⟧]+⟧")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all WAVs")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 0 or args.max_examples < 0:
        parser.error("--limit and --max-examples must be non-negative")

    counters: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)

    wav_paths = sorted(args.root.rglob("*.wav"))
    if args.limit:
        wav_paths = wav_paths[: args.limit]
    for wav_path in wav_paths:
        counters["wav_total"] += 1
        metadata_path = wav_path.with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            counters["missing_metadata"] += 1
            continue
        except (json.JSONDecodeError, OSError, TypeError):
            counters["invalid_metadata"] += 1
            continue
        raw = metadata.get("gen_text") if isinstance(metadata, dict) else None
        if not isinstance(raw, str) or not raw:
            counters["missing_gen_text"] += 1
            continue
        language = metadata.get("language")
        lang_type = metadata.get("lang_type") or metadata.get("lang_key")
        context = reference_normalization_context(
            language, lang_type, text=raw
        )
        context_counts[context["speech_tag_language"]] += 1
        if (
            context["speech_tag_language"] == "unknown"
            and len(examples["unknown_language_context"]) < args.max_examples
        ):
            examples["unknown_language_context"].append(
                {"wav": str(wav_path), "raw": raw}
            )
        try:
            normalized = normalize_reference(
                raw, language=language, lang_type=lang_type
            )
            repeated = normalize_reference(
                normalized, language=language, lang_type=lang_type
            )
        except Exception as exc:  # report every bad corpus item, then continue
            counters["normalization_error"] += 1
            if len(examples["normalization_error"]) < args.max_examples:
                examples["normalization_error"].append(
                    {"wav": str(wav_path), "error": repr(exc), "raw": raw}
                )
            continue
        counters["normalized"] += 1
        if normalized != raw:
            counters["surface_or_entity_changed"] += 1
        if not normalized:
            counters["empty_normalized"] += 1
        if repeated != normalized:
            counters["non_idempotent"] += 1
            if len(examples["non_idempotent"]) < args.max_examples:
                examples["non_idempotent"].append(
                    {"wav": str(wav_path), "once": normalized, "twice": repeated}
                )
        kinds = [match.group("kind") for match in _TOKEN_RE.finditer(normalized)]
        token_counts.update(kinds)
        for kind in sorted(set(kinds)):
            key = f"token_{kind}"
            counters[key] += 1
            if len(examples[key]) < args.max_examples:
                examples[key].append(
                    {"wav": str(wav_path), "raw": raw, "normalized": normalized}
                )

    result = {
        "root": str(args.root.resolve()),
        "limit": args.limit,
        "counts": dict(sorted(counters.items())),
        "canonical_token_occurrences": dict(sorted(token_counts.items())),
        "speech_tag_context_items": dict(sorted(context_counts.items())),
        "examples": dict(sorted(examples.items())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    failures = sum(
        counters[key]
        for key in (
            "missing_metadata",
            "invalid_metadata",
            "missing_gen_text",
            "normalization_error",
            "empty_normalized",
            "non_idempotent",
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
