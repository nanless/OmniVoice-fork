#!/usr/bin/env python3
"""Evaluate speaker similarity between cloned audio and reference audio.

For each text_*.json under --out-dir (voice_clone output), compare ref_audio vs text_*.wav
using samresnet100 embeddings (eval_sim/model/, self-contained).

Usage:
    conda activate omnivoice
    cd batch_generate_text_and_clone/eval_sim
    python eval_clone_similarity.py
    python eval_clone_similarity.py --gpu 0 --sample-size 200
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
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import speaker_similarity as sim  # noqa: E402

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "CLONED_VOICES_ROOT",
        "/root/code/github_repos/OmniVoice-fork/batch_cloned_voices",
    )
)


def find_clone_pairs(out_dir: Path) -> List[Tuple[Path, Path, Path]]:
    pairs = []
    for json_path in sorted(out_dir.rglob("text_*.json")):
        if json_path.name.endswith((".eval.json", ".sim.json")):
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("status") != "generated":
            continue
        ref_audio = meta.get("ref_audio")
        cloned = Path(meta.get("cloned_audio") or json_path.with_suffix(".wav"))
        if not ref_audio or not cloned.exists() or not Path(ref_audio).exists():
            continue
        pairs.append((cloned, Path(ref_audio), json_path))
    return pairs


def write_sim_json(json_path: Path, record: dict):
    json_path.with_suffix(".sim.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def summarize(results: list) -> dict:
    scores = [r["similarity"] for r in results if r.get("similarity") is not None]
    by_dataset = defaultdict(list)
    for r in results:
        if r.get("similarity") is not None:
            by_dataset[r.get("dataset", "unknown")].append(r["similarity"])

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
        "failed_count": sum(1 for r in results if r.get("similarity") is None),
        "total_count": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=sim.DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--no-sidecar", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pairs = find_clone_pairs(args.out_dir)
    if not pairs:
        print("No cloned audio pairs found.")
        return

    if args.sample_size is not None and args.sample_size < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.sample_size)
        print(f"Using sample of {len(pairs)} pairs (seed={args.seed})")

    cache_dir = args.cache_dir or (args.out_dir / "eval_sim_embedding_cache")
    if args.skip_cache and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    print(f"Found {len(pairs)} clone/ref pairs")
    print(f"Loading samresnet100 from {args.model_dir} …")
    encoder = sim.load_encoder(str(args.model_dir), device="cuda:0")
    cache = sim.EmbeddingCache(cache_dir)
    print("Model loaded.\n")

    results = []
    for cloned_wav, ref_audio, json_path in tqdm(pairs, desc="similarity"):
        rel = cloned_wav.relative_to(args.out_dir)
        dataset = rel.parts[0] if rel.parts else "unknown"
        score = cache.similarity(encoder, str(ref_audio), str(cloned_wav))
        record = {
            "cloned_audio": str(cloned_wav),
            "ref_audio": str(ref_audio),
            "similarity": score,
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": str(args.model_dir),
            "evaluated_at": datetime.now().isoformat(),
        }
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            record.update(
                gen_text=meta.get("gen_text"),
                ref_text=meta.get("ref_text"),
                language=meta.get("language"),
                speed=meta.get("speed"),
            )
        except (json.JSONDecodeError, OSError):
            pass
        results.append(record)
        if not args.no_sidecar:
            write_sim_json(json_path, record)

    summary = summarize(results)
    tag = f"_{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_sim_summary{tag}.json"
    details_path = args.out_dir / f"eval_sim_details{tag}.jsonl"
    summary.update(
        model_dir=str(args.model_dir),
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
    print(f"Overall similarity (n={ov['count']}, failed={summary['failed_count']})")
    if ov["count"]:
        print(f"  mean={ov['mean']:.4f}  p50={ov['p50']:.4f}  p10={ov['p10']:.4f}  p90={ov['p90']:.4f}")
        print(f"  min={ov['min']:.4f}  max={ov['max']:.4f}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    main()
