#!/usr/bin/env python3
"""Filter cloned audios by CER and/or speaker similarity thresholds.

Outputs plain-text lists of wav paths, one per line.

Usage:
    # Low CER only (<0.1)
    python filter_cloned.py --max-cer 0.1 --output cer_low.txt

    # High raw cosine SIM only (default >0.8)
    python filter_cloned.py --output sim_high.txt

    # Both thresholds (AND logic)
    python filter_cloned.py --max-cer 0.1 --min-sim 0.8 --output cer_low_sim_high.txt

    # All combinations at once
    python filter_cloned.py --max-cer 0.1 --min-sim 0.8 --output-dir ./filtered/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "CLONED_VOICES_ROOT",
        "/root/group-shared/voiceprint/data/speech/voice_activity_detection"
        "/batch_cloned_voices_ommivoice_kids_finetuned",
    )
)
DEFAULT_MIN_SIM = 0.8

EVAL_SIM_DIR = Path(__file__).resolve().parent / "eval_sim"
sys.path.insert(0, str(EVAL_SIM_DIR))
from metric_contract import (  # noqa: E402
    SimilarityCollectionValidator,
    validate_current_audio_files,
    validate_current_model_files,
)
EVAL_CER_DIR = Path(__file__).resolve().parent / "eval_cer"
sys.path.insert(0, str(EVAL_CER_DIR))
from cer_normalization import (  # noqa: E402
    CER_SCORE_VERSION,
    EVAL_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    SAFE_PROFILE,
    normalization_fingerprint,
    reference_normalization_input_fingerprint,
)
from eval_contract import (  # noqa: E402
    asr_decode_fingerprint,
    asr_model_fingerprint,
)
from eval_common import iter_clone_records, write_json  # noqa: E402


def load_cer_data(jsonl_path: Path) -> Dict[str, dict]:
    """Load canonical deterministic CER v4 rows keyed by WAV path."""
    t0 = time.time()
    data = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            source = f"{jsonl_path}:{line_no}"
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}: invalid JSON: {exc}") from exc
            if not isinstance(r, dict):
                raise ValueError(f"{source}: CER row must be a JSON object")
            wav = r.get("cloned_audio") or r.get("wav")
            if not isinstance(wav, str) or not wav:
                raise ValueError(f"{source}: missing cloned_audio/wav")
            if (
                r.get("eval_schema_version") != EVAL_SCHEMA_VERSION
                or r.get("cer_metric") != "deterministic_char_cer"
                or r.get("cer_score_version") != CER_SCORE_VERSION
                or r.get("normalization_profile") != SAFE_PROFILE
                or r.get("normalization_version") != NORMALIZATION_VERSION
                or r.get("normalization_fingerprint")
                != normalization_fingerprint(SAFE_PROFILE)
                or r.get("asr_model_fingerprint") != asr_model_fingerprint()
                or r.get("asr_decode_fingerprint") != asr_decode_fingerprint()
                or r.get("stage") != "complete"
            ):
                raise ValueError(f"{source}: not a canonical deterministic CER v4 row")
            value = r.get("cer")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{source}: invalid cer={value!r}")
            try:
                stat = Path(wav).stat()
            except OSError as exc:
                raise ValueError(f"{source}: cannot stat current cloned audio: {exc}") from exc
            current_signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if r.get("cloned_audio_signature") != current_signature:
                raise ValueError(
                    f"{source}: CER cloned_audio_signature does not match current WAV"
                )
            if wav in data:
                raise ValueError(f"{source}: duplicate CER record for {wav}")
            data[wav] = r
    print(f"[CER] loaded {len(data)} records in {time.time() - t0:.1f}s", file=sys.stderr)
    return data


def validate_cer_normalization_inputs(
    cer_data: Dict[str, dict], inventory_records: Dict[str, dict]
) -> None:
    for wav, row in cer_data.items():
        clone = inventory_records.get(wav)
        if clone is None:
            continue
        text = clone.get("gen_text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Current clone metadata has no gen_text: {wav}")
        expected = reference_normalization_input_fingerprint(
            text,
            language=clone.get("language"),
            lang_type=clone.get("lang_type") or clone.get("lang_key"),
        )
        if row.get("reference_normalization_input_fingerprint") != expected:
            raise ValueError(
                f"CER reference normalization input is stale for current clone metadata: {wav}"
            )


def load_sim_data(details_path: Path) -> Dict[str, dict]:
    """Load one explicit full-dataset SIM aggregate."""
    t0 = time.time()
    validator = SimilarityCollectionValidator()
    with open(details_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{details_path}:{line_no}: invalid JSON: {exc}") from exc
            source = f"{details_path}:{line_no}"
            validator.add(r, source)
            validate_current_audio_files(r, source)
    data = dict(validator.records)
    if data:
        validate_current_model_files(next(iter(data.values())), details_path)
    print(f"[SIM] loaded {len(data)} records from {details_path} in {time.time() - t0:.1f}s", file=sys.stderr)
    return data


def extract_dataset(wav: str, out_dir: str) -> str:
    try:
        return Path(wav).relative_to(out_dir).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def filter_and_save(
    cer_data: Dict[str, dict],
    sim_data: Dict[str, dict],
    max_cer: Optional[float],
    min_sim: Optional[float],
    output: Path,
    out_dir: str,
    inventory: Set[str],
    manifest_base: dict,
):
    """Filter wavs by thresholds and write to output file."""
    matched = []
    cer_miss = 0
    sim_miss = 0

    # Determine candidate pool
    if max_cer is not None and min_sim is not None:
        candidates = inventory & set(cer_data.keys()) & set(sim_data.keys())
    elif max_cer is not None:
        candidates = inventory & set(cer_data.keys())
    elif min_sim is not None:
        candidates = inventory & set(sim_data.keys())
    else:
        print("ERROR: at least one threshold required", file=sys.stderr)
        return

    for wav in sorted(candidates):
        ok = True
        tag = []

        if max_cer is not None:
            c = cer_data.get(wav, {}).get("cer")
            if c is None:
                cer_miss += 1
                ok = False
            elif c >= max_cer:
                ok = False
            else:
                tag.append(f"cer={c:.4f}")

        if min_sim is not None:
            s = sim_data.get(wav, {}).get("similarity")
            if s is None:
                sim_miss += 1
                ok = False
            elif s <= min_sim:
                ok = False
            else:
                tag.append(f"sim={s:.4f}")

        if ok:
            matched.append(wav)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(matched) + "\n" if matched else "")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    write_json(output.with_suffix(output.suffix + ".manifest.json"), {
        **manifest_base,
        "output": str(output),
        "max_cer": max_cer,
        "min_raw_cosine_similarity": min_sim,
        "inventory_count": len(inventory),
        "matched_count": len(matched),
        "cer_missing_in_candidates": cer_miss,
        "sim_missing_in_candidates": sim_miss,
    })
    return matched, cer_miss, sim_miss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cer", type=float, default=None, help="Max deterministic CER v4 threshold (exclusive)")
    parser.add_argument(
        "--min-sim", type=float, default=None,
        help=(
            "Min raw cosine similarity threshold in [-1, 1] (exclusive); "
            f"defaults to {DEFAULT_MIN_SIM} when neither threshold is specified"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Single output file path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (auto-names files)")
    parser.add_argument("--sim-details", type=Path, default=None,
                        help="Explicit SIM JSONL (default: OUT/eval_sim_details.jsonl)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow missing metric coverage and invalid clone records")
    args = parser.parse_args()

    if args.max_cer is None and args.min_sim is None:
        args.min_sim = DEFAULT_MIN_SIM

    if args.min_sim is not None and (
        not math.isfinite(args.min_sim) or not -1.0 <= args.min_sim <= 1.0
    ):
        parser.error("--min-sim must be finite and in [-1, 1]")
    if args.max_cer is not None and (
        not math.isfinite(args.max_cer) or args.max_cer < 0
    ):
        parser.error("--max-cer must be finite and non-negative")

    if not args.output and not args.output_dir:
        parser.error("Either --output or --output-dir required")

    out_dir = args.out_dir.resolve()
    inventory_records = {
        str(wav): record
        for wav, _, record in iter_clone_records(
            out_dir, allow_partial=args.allow_partial
        )
    }
    inventory = set(inventory_records)
    if not inventory:
        raise RuntimeError(f"No valid clone inventory found under {out_dir}")
    cer_path = out_dir / "eval_cer_details.jsonl"

    # For SIM data, look in out_dir
    sim_details = args.sim_details or (out_dir / "eval_sim_details.jsonl")

    # Load CER
    cer_data = {}
    if args.max_cer is not None:
        cer_data = load_cer_data(cer_path)
        validate_cer_normalization_inputs(cer_data, inventory_records)

    # Load SIM
    sim_data = {}
    if args.min_sim is not None:
        sim_data = load_sim_data(sim_details)
        null_sim = {wav for wav, record in sim_data.items() if record.get("similarity") is None}
        if null_sim and not args.allow_partial:
            raise RuntimeError(
                f"SIM contains {len(null_sim)} failed/null scores; rerun SIM or use --allow-partial"
            )
        for wav in null_sim:
            sim_data.pop(wav)

    requested_collections = []
    if args.max_cer is not None:
        requested_collections.append(("CER", cer_data))
    if args.min_sim is not None:
        requested_collections.append(("SIM", sim_data))
    for label, records in requested_collections:
        extra = set(records) - inventory
        missing = inventory - set(records)
        if extra:
            raise RuntimeError(f"{label} contains {len(extra)} records outside current clone inventory")
        if missing and not args.allow_partial:
            raise RuntimeError(
                f"{label} coverage incomplete: {len(records)}/{len(inventory)}; "
                "use --allow-partial only for intentional partial screening"
            )

    manifest_base = {
        "schema_version": 1,
        "out_dir": str(out_dir),
        "cer_details": str(cer_path) if args.max_cer is not None else None,
        "sim_details": str(sim_details) if args.min_sim is not None else None,
        "cer_metric": "deterministic_char_cer" if args.max_cer is not None else None,
        "cer_score_version": CER_SCORE_VERSION if args.max_cer is not None else None,
        "cer_normalization_version": NORMALIZATION_VERSION if args.max_cer is not None else None,
        "cer_asr_model_fingerprint": asr_model_fingerprint() if args.max_cer is not None else None,
        "cer_asr_decode_fingerprint": asr_decode_fingerprint() if args.max_cer is not None else None,
        "similarity_metric": "raw_cosine" if args.min_sim is not None else None,
        "allow_partial": args.allow_partial,
        "cer_record_count": len(cer_data),
        "sim_record_count": len(sim_data),
        "cer_missing_from_inventory": len(inventory - set(cer_data)) if args.max_cer is not None else None,
        "sim_missing_from_inventory": len(inventory - set(sim_data)) if args.min_sim is not None else None,
    }

    # Determine outputs
    tasks: List[Tuple[Optional[float], Optional[float], Path]] = []

    if args.output:
        tasks.append((args.max_cer, args.min_sim, args.output))
    else:
        od = args.output_dir
        # Auto-generate named files for each combination
        if args.max_cer is not None and args.min_sim is not None:
            tasks.append((args.max_cer, args.min_sim,
                          od / f"cer_lt{args.max_cer}_sim_gt{args.min_sim}.txt"))
        if args.max_cer is not None and args.min_sim is not None:
            tasks.append((args.max_cer, None,
                          od / f"cer_lt{args.max_cer}.txt"))
            tasks.append((None, args.min_sim,
                          od / f"sim_gt{args.min_sim}.txt"))
        elif args.max_cer is not None:
            tasks.append((args.max_cer, None, od / f"cer_lt{args.max_cer}.txt"))
        elif args.min_sim is not None:
            tasks.append((None, args.min_sim, od / f"sim_gt{args.min_sim}.txt"))

    # Execute
    for mc, ms, outpath in tasks:
        print(f"\n{'='*60}", file=sys.stderr)
        tag_parts = []
        if mc is not None:
            tag_parts.append(f"CER < {mc}")
        if ms is not None:
            tag_parts.append(f"SIM > {ms}")
        tag = " AND ".join(tag_parts)
        print(f"Filtering: {tag}", file=sys.stderr)

        matched, cer_miss, sim_miss = filter_and_save(
            cer_data, sim_data, mc, ms, outpath, str(out_dir), inventory, manifest_base,
        )

        # Stats
        print(f"  Matched: {len(matched):,}", file=sys.stderr)
        if cer_miss:
            print(f"  CER missing: {cer_miss}", file=sys.stderr)
        if sim_miss:
            print(f"  SIM missing: {sim_miss}", file=sys.stderr)

        # Per-dataset breakdown
        by_ds = defaultdict(int)
        for wav in matched:
            by_ds[extract_dataset(wav, str(out_dir))] += 1
        if by_ds:
            print(f"  By dataset:", file=sys.stderr)
            for ds in sorted(by_ds.keys()):
                print(f"    {ds}: {by_ds[ds]:,}", file=sys.stderr)

        # Print a few samples
        if matched:
            print(f"  Sample paths:", file=sys.stderr)
            for p in matched[:5]:
                print(f"    {p}", file=sys.stderr)

        print(f"  Wrote: {outpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
