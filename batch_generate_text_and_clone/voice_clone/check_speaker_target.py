#!/usr/bin/env python3
"""Check whether every canonical speaker has the requested accepted duration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from speaker_topup_common import (
    fraction_record,
    inventory_accepted,
    inventory_strict_accepted,
    inventory_generated,
    inventory_original,
    iter_wavs,
    write_json_atomic,
    write_jsonl_atomic,
)
from plan_speaker_topup import DEFAULT_ORIGINAL_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument(
        "--accepted-root", type=Path, action="append", required=True,
        help="Quality-accepted clone root in <dataset>/<speaker>/ layout; repeatable",
    )
    parser.add_argument(
        "--strict-accepted-root", type=Path, action="append", default=[],
        help="Transactional top-up root; only fully committed rounds are counted",
    )
    parser.add_argument(
        "--generated-root", type=Path, action="append", default=[],
        help=(
            "Optional raw plan output root for diagnostics only; generated audio "
            "never contributes to the accepted-duration target"
        ),
    )
    parser.add_argument("--target-seconds", type=Fraction, default=Fraction(1800))
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--scan-workers", type=int, default=16)
    args = parser.parse_args()
    if args.target_seconds <= 0:
        parser.error("--target-seconds must be positive")
    if args.scan_workers <= 0:
        parser.error("--scan-workers must be positive")
    return args


def main() -> int:
    args = parse_args()
    original_root = args.original_root.resolve()
    original, original_counts, target_paths = inventory_original(
        original_root, args.scan_workers
    )
    seen = {
        wav.resolve() for speaker_dir in target_paths.values() for wav in iter_wavs(speaker_dir)
    }
    accepted, accepted_counts, seen = inventory_accepted(
        [path.resolve() for path in args.accepted_root], target_paths, seen,
        args.scan_workers,
    )
    strict, strict_counts, seen = inventory_strict_accepted(
        [path.resolve() for path in args.strict_accepted_root], target_paths, seen,
        args.scan_workers,
    )
    for speaker_key in target_paths:
        accepted[speaker_key] += strict[speaker_key]
        accepted_counts[speaker_key] += strict_counts[speaker_key]
    generated, generated_counts, _ = inventory_generated(
        [path.resolve() for path in args.generated_root], target_paths, seen,
        args.scan_workers,
    )

    rows = []
    for speaker_key in sorted(target_paths):
        total = original[speaker_key] + accepted[speaker_key]
        deficit = max(Fraction(0), args.target_seconds - total)
        rows.append({
            "record_type": "speaker_target_status",
            "speaker_key": speaker_key,
            "target_duration": fraction_record(args.target_seconds),
            "original_duration": fraction_record(original[speaker_key]),
            "accepted_duration": fraction_record(accepted[speaker_key]),
            "diagnostic_generated_duration": fraction_record(generated[speaker_key]),
            "total_duration": fraction_record(total),
            "deficit_duration": fraction_record(deficit),
            "original_file_count": original_counts[speaker_key],
            "accepted_file_count": accepted_counts[speaker_key],
            "diagnostic_generated_file_count": generated_counts[speaker_key],
            "meets_target": deficit == 0,
        })

    deficient = [row for row in rows if not row["meets_target"]]
    summary = {
        "record_type": "speaker_target_summary",
        "schema_version": 1,
        "counting_policy": "original_plus_accepted_only",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "original_root": str(original_root),
        "accepted_roots": [str(path.resolve()) for path in args.accepted_root],
        "strict_accepted_roots": [
            str(path.resolve()) for path in args.strict_accepted_root
        ],
        "generated_roots": [str(path.resolve()) for path in args.generated_root],
        "target_duration": fraction_record(args.target_seconds),
        "scan_workers": args.scan_workers,
        "speaker_count": len(rows),
        "meeting_target_count": len(rows) - len(deficient),
        "deficient_speaker_count": len(deficient),
        "total_deficit_duration": fraction_record(sum(
            (Fraction(
                row["deficit_duration"]["numerator"],
                row["deficit_duration"]["denominator"],
            ) for row in deficient),
            Fraction(0),
        )),
        "all_speakers_meet_target": not deficient,
    }
    if args.output_jsonl:
        write_jsonl_atomic(args.output_jsonl, [summary, *rows])
    if args.summary_json:
        write_json_atomic(args.summary_json, {**summary, "speakers": rows})

    print(
        f"speakers={len(rows)} meeting={len(rows) - len(deficient)} "
        f"deficient={len(deficient)} "
        f"total_deficit_seconds={summary['total_deficit_duration']['seconds']:.6f}"
    )
    for row in sorted(
        deficient, key=lambda item: item["deficit_duration"]["seconds"], reverse=True
    )[:20]:
        print(
            f"  {row['speaker_key']}: total={row['total_duration']['seconds']:.6f}s "
            f"deficit={row['deficit_duration']['seconds']:.6f}s"
        )
    return 0 if not deficient else 1


if __name__ == "__main__":
    raise SystemExit(main())
