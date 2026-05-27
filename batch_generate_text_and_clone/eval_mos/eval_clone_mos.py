#!/usr/bin/env python3
"""Evaluate cloned audio with UTMOS (omnivoice/eval MOS predictor).

Scans batch_cloned_voices for text_*.wav (status=generated) and scores each with
UTMOS22Strong — same model/path as omnivoice/eval/mos/utmos.py.

Usage:
    conda activate omnivoice
    cd batch_generate_text_and_clone/eval_mos
    python eval_clone_mos.py
    python eval_clone_mos.py --gpu 0 --sample-size 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from utmos_scorer import UTMosScorer, DEFAULT_MODEL_DIR  # noqa: E402

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "CLONED_VOICES_ROOT",
        "/root/code/github_repos/OmniVoice-fork/batch_cloned_voices",
    )
)


def find_cloned_wavs(out_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (wav_path, sidecar_json) for generated clones."""
    items = []
    for json_path in sorted(out_dir.rglob("text_*.json")):
        if json_path.name.endswith((".eval.json", ".sim.json", ".mos.json")):
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("status") != "generated":
            continue
        wav_path = Path(meta.get("cloned_audio") or json_path.with_suffix(".wav"))
        if wav_path.exists():
            items.append((wav_path, json_path))
    return items


def write_mos_json(json_path: Path, record: dict):
    json_path.with_suffix(".mos.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def summarize(results: list) -> dict:
    scores = [r["utmos"] for r in results if r.get("utmos") is not None]
    by_dataset = defaultdict(list)
    by_language = defaultdict(list)
    for r in results:
        if r.get("utmos") is None:
            continue
        by_dataset[r.get("dataset", "unknown")].append(r["utmos"])
        by_language[r.get("language", "unknown")].append(r["utmos"])

    def stats(vals):
        if not vals:
            return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p10": None, "p90": None}
        a = np.array(vals, dtype=np.float64)
        return {
            "count": int(len(a)),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
        }

    return {
        "overall": stats(scores),
        "by_dataset": {k: stats(v) for k, v in sorted(by_dataset.items())},
        "by_language": {k: stats(v) for k, v in sorted(by_language.items())},
        "failed_count": sum(1 for r in results if r.get("utmos") is None),
        "total_count": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sidecar", action="store_true", help="Do not write text_*.mos.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    items = find_cloned_wavs(args.out_dir)
    if not items:
        print("No cloned audio found.")
        return

    if args.sample_size is not None and args.sample_size < len(items):
        items = random.Random(args.seed).sample(items, args.sample_size)
        print(f"Using sample of {len(items)} files (seed={args.seed})")

    print(f"Found {len(items)} cloned wavs")
    print(f"Loading UTMOS from {args.model_dir} …")
    scorer = UTMosScorer(model_dir=args.model_dir, device="cuda:0")
    print("Model loaded.\n")

    results = []
    for wav_path, json_path in tqdm(items, desc="UTMOS"):
        rel = wav_path.relative_to(args.out_dir)
        dataset = rel.parts[0] if rel.parts else "unknown"
        record = {
            "cloned_audio": str(wav_path),
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": scorer.model_dir,
            "evaluated_at": datetime.now().isoformat(),
        }
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            record.update(
                ref_audio=meta.get("ref_audio"),
                gen_text=meta.get("gen_text"),
                language=meta.get("language"),
                speed=meta.get("speed"),
            )
        except (json.JSONDecodeError, OSError):
            pass

        try:
            record["utmos"] = scorer.score_file(wav_path)
        except Exception as e:
            record["utmos"] = None
            record["error"] = str(e)

        results.append(record)
        if not args.no_sidecar and record.get("utmos") is not None:
            write_mos_json(json_path, record)

    summary = summarize(results)
    tag = f"_{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_mos_summary{tag}.json"
    details_path = args.out_dir / f"eval_mos_details{tag}.jsonl"

    summary.update(
        metric="UTMOS",
        model_dir=scorer.model_dir,
        evaluated_at=datetime.now().isoformat(),
        out_dir=str(args.out_dir),
        sample_size=args.sample_size,
        seed=args.seed if args.sample_size else None,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ov = summary["overall"]
    print(f"\n{'='*60}")
    print(f"Overall UTMOS (n={ov['count']}, failed={summary['failed_count']})")
    if ov["count"]:
        print(f"  mean={ov['mean']:.3f}  p50={ov['p50']:.3f}  p10={ov['p10']:.3f}  p90={ov['p90']:.3f}")
        print(f"  min={ov['min']:.3f}  max={ov['max']:.3f}")
    print(f"\nBy language:")
    for lang, st in summary["by_language"].items():
        if st["count"]:
            print(f"  {lang}: n={st['count']} mean={st['mean']:.3f}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    main()
