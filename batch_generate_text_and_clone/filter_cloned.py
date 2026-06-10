#!/usr/bin/env python3
"""Filter cloned audios by CER and/or speaker similarity thresholds.

Outputs plain-text lists of wav paths, one per line.

Usage:
    # Low CER only (<0.1)
    python filter_cloned.py --max-cer 0.1 --output cer_low.txt

    # High SIM only (>0.8)
    python filter_cloned.py --min-sim 0.8 --output sim_high.txt

    # Both thresholds (AND logic)
    python filter_cloned.py --max-cer 0.1 --min-sim 0.8 --output cer_low_sim_high.txt

    # All combinations at once
    python filter_cloned.py --max-cer 0.1 --min-sim 0.8 --output-dir ./filtered/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_OUT_DIR = Path(
    "/root/group-shared/voiceprint/data/speech/voice_activity_detection"
    "/batch_cloned_voices_ommivoice_kids_finetuned"
)


def load_cer_data(jsonl_path: Path) -> Dict[str, dict]:
    """Load CER data from eval_cer_details.jsonl. Returns {wav_path: {manual_cer, llm_cer, ...}}."""
    t0 = time.time()
    data = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav = r.get("wav", "")
            if wav:
                data[wav] = r
    print(f"[CER] loaded {len(data)} records in {time.time() - t0:.1f}s", file=sys.stderr)
    return data


def load_sim_data(sim_dir: Path) -> Dict[str, dict]:
    """Load SIM data from eval_sim_details*.jsonl files, dedup by cloned_audio."""
    t0 = time.time()
    seen = set()
    data = {}
    for fp in sorted(sim_dir.glob("eval_sim_details*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wav = r.get("cloned_audio", "")
                if wav and wav not in seen:
                    seen.add(wav)
                    data[wav] = r
    print(f"[SIM] loaded {len(data)} records from {len(sorted(sim_dir.glob('eval_sim_details*.jsonl')))} files in {time.time() - t0:.1f}s", file=sys.stderr)
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
):
    """Filter wavs by thresholds and write to output file."""
    matched = []
    cer_miss = 0
    sim_miss = 0

    # Determine candidate pool
    if max_cer is not None and min_sim is not None:
        candidates = set(cer_data.keys()) & set(sim_data.keys())
    elif max_cer is not None:
        candidates = set(cer_data.keys())
    elif min_sim is not None:
        candidates = set(sim_data.keys())
    else:
        print("ERROR: at least one threshold required", file=sys.stderr)
        return

    for wav in sorted(candidates):
        ok = True
        tag = []

        if max_cer is not None:
            c = cer_data.get(wav, {}).get("manual_cer")
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
    output.write_text("\n".join(matched) + "\n" if matched else "", encoding="utf-8")
    return matched, cer_miss, sim_miss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cer", type=float, default=None, help="Max manual CER threshold (exclusive)")
    parser.add_argument("--min-sim", type=float, default=None, help="Min similarity threshold (exclusive)")
    parser.add_argument("--use-llm-cer", action="store_true", help="Use LLM CER instead of manual CER")
    parser.add_argument("--output", type=Path, default=None, help="Single output file path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (auto-names files)")
    args = parser.parse_args()

    if args.max_cer is None and args.min_sim is None:
        parser.error("At least one of --max-cer or --min-sim required")
    if not args.output and not args.output_dir:
        parser.error("Either --output or --output-dir required")

    out_dir = args.out_dir
    cer_path = out_dir / "eval_cer_details.jsonl"

    # For SIM data, look in out_dir
    sim_files_dir = out_dir

    # Load CER
    cer_data = {}
    if args.max_cer is not None:
        cer_data = load_cer_data(cer_path)
        if args.use_llm_cer:
            for w, r in cer_data.items():
                if r.get("llm_cer") is not None:
                    r["manual_cer"] = r["llm_cer"]

    # Load SIM
    sim_data = {}
    if args.min_sim is not None:
        sim_data = load_sim_data(sim_files_dir)

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
            cer_data, sim_data, mc, ms, outpath, str(out_dir),
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
