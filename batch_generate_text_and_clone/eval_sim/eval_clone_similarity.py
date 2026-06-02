#!/usr/bin/env python3
"""Evaluate speaker similarity between cloned audio and reference audio.

Multi-process: split pairs across --workers processes (works on single GPU too).
Each worker loads its own model, writes text_*.sim.json incrementally.

Usage:
    conda activate omnivoice
    cd batch_generate_text_and_clone/eval_sim
    python eval_clone_similarity.py
    python eval_clone_similarity.py --workers 4 --gpus 0
    python eval_clone_similarity.py --workers 2 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(EVAL_DIR))

import speaker_similarity as sim  # noqa: E402
from eval_common import (  # noqa: E402
    append_jsonl,
    list_clone_pairs,
    merge_jsonl_parts,
    parse_gpu_list,
    split_shards,
    write_json,
)

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "CLONED_VOICES_ROOT",
        "/root/code/github_repos/OmniVoice-fork/batch_cloned_voices",
    )
)


def find_clone_pairs(out_dir: Path) -> List[Tuple[Path, Path, Path]]:
    return list_clone_pairs(out_dir, label="sim-scan", scan_workers=8)


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


def _load_results_from_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sim_worker(
    rank: int,
    gpu: str,
    shard: list,
    out_dir: str,
    model_dir: str,
    details_path: str,
    no_sidecar: bool,
):
    """One process: own CUDA context + model on assigned GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out = Path(out_dir)
    details = Path(details_path)

    encoder = sim.load_encoder(model_dir, device="cuda:0")
    print(f"[sim-w{rank}] gpu={gpu} items={len(shard)}", flush=True)

    for cloned_s, ref_s, json_s in tqdm(shard, desc=f"sim-w{rank}", position=rank):
        cloned_wav = Path(cloned_s)
        ref_audio = Path(ref_s)
        json_path = Path(json_s)
        rel = cloned_wav.relative_to(out)
        dataset = rel.parts[0] if rel.parts else "unknown"
        meta = {}
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        score = sim.compute_similarity(encoder, str(ref_audio), str(cloned_wav))
        record = {
            "cloned_audio": str(cloned_wav),
            "ref_audio": str(ref_audio),
            "similarity": score,
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": model_dir,
            "worker": rank,
            "evaluated_at": datetime.now().isoformat(),
        }
        if meta:
            record.update(
                gen_text=meta.get("gen_text"),
                ref_text=meta.get("ref_text"),
                language=meta.get("language"),
                speed=meta.get("speed"),
            )
        if not no_sidecar:
            write_sim_json(json_path, record)
        append_jsonl(details, record)


def _run_single_process(pairs, args, summary_path, details_path):
    gpu_list = parse_gpu_list(args.gpus, args.gpu)
    shard = [(str(c), str(r), str(j)) for c, r, j in pairs]
    _sim_worker(0, gpu_list[0], shard, str(args.out_dir), str(args.model_dir),
                str(details_path), args.no_sidecar)
    results = _load_results_from_jsonl(details_path)
    summary = summarize(results)
    summary.update(
        model_dir=str(args.model_dir),
        evaluated_at=datetime.now().isoformat(),
        out_dir=str(args.out_dir),
        sample_size=args.sample_size,
        seed=args.seed if args.sample_size else None,
        workers=1,
        gpus=gpu_list,
        items_done=len(results),
        items_total=len(pairs),
    )
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=sim.DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU id (legacy)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--workers", type=int, default=4, help="Process count (default 4, ok on 1 GPU)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    gpu_list = parse_gpu_list(args.gpus, args.gpu if args.gpu is not None else 0)
    workers = max(1, args.workers)

    pairs = find_clone_pairs(args.out_dir)
    if not pairs:
        print("No cloned audio pairs found.", flush=True)
        return

    if args.skip_existing:
        before = len(pairs)
        pairs = [(c, r, j) for c, r, j in pairs if not j.with_suffix(".sim.json").exists()]
        print(f"Skip-existing: {before - len(pairs)} already done, {len(pairs)} remaining", flush=True)
        if not pairs:
            print("All pairs already evaluated.", flush=True)
            return

    if args.sample_size is not None and args.sample_size < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.sample_size)
        print(f"Using sample of {len(pairs)} pairs (seed={args.seed})", flush=True)

    tag = f"_{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_sim_summary{tag}.json"
    details_path = args.out_dir / f"eval_sim_details{tag}.jsonl"
    if not args.skip_existing:
        if details_path.exists():
            details_path.unlink()
        for p in args.out_dir.glob(f"eval_sim_details{tag}.w*.jsonl"):
            p.unlink()

    print(
        f"Found {len(pairs)} pairs | workers={workers} gpus={gpu_list}",
        flush=True,
    )

    if workers == 1:
        summary = _run_single_process(pairs, args, summary_path, details_path)
    else:
        shards = split_shards(pairs, workers)
        shard_strs = [[(str(c), str(r), str(j)) for c, r, j in s] for s in shards]
        part_paths = [
            args.out_dir / f"eval_sim_details{tag}.w{i}.jsonl" for i in range(workers)
        ]
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(workers):
            if not shard_strs[i]:
                continue
            gpu = gpu_list[i % len(gpu_list)]
            p = ctx.Process(
                target=_sim_worker,
                args=(
                    i, gpu, shard_strs[i], str(args.out_dir), str(args.model_dir),
                    str(part_paths[i]), args.no_sidecar,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"SIM worker exited with code {p.exitcode}")

        merge_jsonl_parts(part_paths, details_path)
        for p in part_paths:
            p.unlink(missing_ok=True)

        results = _load_results_from_jsonl(details_path)
        summary = summarize(results)
        summary.update(
            model_dir=str(args.model_dir),
            evaluated_at=datetime.now().isoformat(),
            out_dir=str(args.out_dir),
            sample_size=args.sample_size,
            seed=args.seed if args.sample_size else None,
            workers=workers,
            gpus=gpu_list,
            items_done=len(results),
            items_total=len(pairs),
        )
        write_json(summary_path, summary)

    ov = summary["overall"]
    print(f"\n{'='*60}", flush=True)
    print(f"Overall similarity (n={ov['count']}, failed={summary['failed_count']})", flush=True)
    if ov["count"]:
        print(f"  mean={ov['mean']:.4f}  p50={ov['p50']:.4f}  p10={ov['p10']:.4f}  p90={ov['p90']:.4f}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {details_path}", flush=True)


if __name__ == "__main__":
    main()
