#!/usr/bin/env python3
"""Randomly select audio pairs from each similarity interval for manual inspection.

Copies cloned audio + reference audio + all sidecar JSONs to per-interval subdirs.

Usage:
    python sample_sim_intervals.py
    python sample_sim_intervals.py --n-per-interval 20 --output-dir ./sim_check
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_OUT_DIR = Path(
    "/root/group-shared/voiceprint/data/speech/voice_activity_detection"
    "/batch_cloned_voices_ommivoice_kids_finetuned"
)

SIM_INTERVALS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.75),
    (0.75, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.00),
]

SIDECAR_SUFFIXES = [".eval.json", ".sim.json", ".mos.json"]


def load_sim_data(sim_dir: Path) -> Dict[str, dict]:
    """Load SIM data from all eval_sim_details*.jsonl files, dedup by cloned_audio."""
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
    print(f"[SIM] loaded {len(data)} records in {time.time() - t0:.1f}s", file=sys.stderr)
    return data


def write_meta(record: dict, path: Path):
    """Write a JSON metadata file for a pair."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(src: str, dst: Path, label: str) -> bool:
    """Copy a file, return True if success."""
    if not src or not Path(src).is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Root cloned voices directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./sim_check"),
                        help="Output directory for sampled pairs")
    parser.add_argument("--n-per-interval", type=int, default=20,
                        help="Number of pairs per interval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selections without copying")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir
    output_root = args.output_dir.resolve()

    # Load SIM records
    sim_data = load_sim_data(out_dir)

    # Group by similarity interval
    interval_groups: Dict[Tuple[float, float], List[Tuple[str, dict]]] = {
        iv: [] for iv in SIM_INTERVALS
    }

    for wav, rec in sim_data.items():
        sim = rec.get("similarity")
        if sim is None:
            continue
        ref = rec.get("ref_audio", "")
        if not ref or not Path(ref).is_file():
            continue
        for lo, hi in SIM_INTERVALS:
            if lo <= sim < hi:
                interval_groups[(lo, hi)].append((wav, rec))
                break

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Similarity interval distribution:", file=sys.stderr)
    for (lo, hi), pairs in interval_groups.items():
        print(f"  [{lo:.2f}, {hi:.2f}): {len(pairs):,} pairs", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Sample and copy
    total_copied = 0
    total_failed = 0

    for (lo, hi), pairs in interval_groups.items():
        n = min(args.n_per_interval, len(pairs))
        if n == 0:
            print(f"[{lo:.2f}-{hi:.2f}] NO PAIRS available, skipping", file=sys.stderr)
            continue

        sampled = rng.sample(pairs, n)
        interval_dir = output_root / f"sim_{lo:.2f}_{hi:.2f}".replace(".", "")
        print(f"[{lo:.2f}-{hi:.2f}] Sampling {n}/{len(pairs)} pairs → {interval_dir}/", file=sys.stderr)

        for i, (cloned_wav, rec) in enumerate(sampled, 1):
            pair_dir = interval_dir / f"pair_{i:03d}"
            sim_val = rec["similarity"]
            ref_audio = rec.get("ref_audio", "")
            sidecar_json = rec.get("sidecar_json", "")

            if args.dry_run:
                print(f"  pair_{i:03d}: sim={sim_val:.4f} cloned={Path(cloned_wav).name} ref={Path(ref_audio).name}", file=sys.stderr)
                continue

            ok = 0
            fail = 0

            # Copy cloned audio
            if copy_file(cloned_wav, pair_dir / Path(cloned_wav).name, "cloned"):
                ok += 1
            else:
                fail += 1
                print(f"    MISS: cloned {cloned_wav}", file=sys.stderr)

            # Copy reference audio
            if copy_file(ref_audio, pair_dir / "ref" / Path(ref_audio).name, "ref"):
                ok += 1
            else:
                fail += 1
                print(f"    MISS: ref {ref_audio}", file=sys.stderr)

            # Copy cloned metadata JSON (generation info)
            if sidecar_json:
                if copy_file(sidecar_json, pair_dir / Path(sidecar_json).name, "meta"):
                    ok += 1
                else:
                    fail += 1

            # Copy eval sidecar files
            for suffix in SIDECAR_SUFFIXES:
                src = Path(cloned_wav).with_suffix(suffix)
                if src.is_file():
                    if copy_file(str(src), pair_dir / src.name, suffix):
                        ok += 1
                    else:
                        fail += 1

            # Copy ref text file (look alongside ref_audio)
            ref_path = Path(ref_audio)
            for text_ext in [".txt", ".lab"]:
                text_src = ref_path.with_suffix(text_ext)
                if text_src.is_file():
                    copy_file(str(text_src), pair_dir / "ref" / text_src.name, "ref_text")

            # Write a summary JSON for this pair
            summary = {
                "pair_index": i,
                "interval": f"[{lo:.2f}, {hi:.2f})",
                "similarity": sim_val,
                "cloned_audio": str(cloned_wav),
                "ref_audio": str(ref_audio),
                "gen_text": rec.get("gen_text"),
                "ref_text": rec.get("ref_text"),
                "language": rec.get("language"),
                "speed": rec.get("speed"),
                "dataset": rec.get("dataset"),
            }
            write_meta(summary, pair_dir / "pair_info.json")

            total_copied += ok
            total_failed += fail

            if i <= 3 or i == n:
                print(f"  pair_{i:03d}: sim={sim_val:.4f} ({ok} copied, {fail} failed)", file=sys.stderr)

    if not args.dry_run:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Done: {total_copied} files copied, {total_failed} failed", file=sys.stderr)
        print(f"Output: {output_root}/", file=sys.stderr)
    else:
        print(f"\n[Dry run complete, no files copied]", file=sys.stderr)


if __name__ == "__main__":
    main()
