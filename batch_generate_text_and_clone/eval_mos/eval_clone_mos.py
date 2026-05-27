#!/usr/bin/env python3
"""Evaluate cloned audio with UTMOS.

Multi-process: split items across --workers processes (works on single GPU too).
Each worker loads its own model, writes text_*.mos.json incrementally.

Usage:
    conda activate omnivoice
    cd batch_generate_text_and_clone/eval_mos
    python eval_clone_mos.py
    python eval_clone_mos.py --workers 4 --gpus 0
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

from utmos_scorer import UTMosScorer, DEFAULT_MODEL_DIR  # noqa: E402
from eval_common import (  # noqa: E402
    append_jsonl,
    list_clone_items,
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


def find_cloned_wavs(out_dir: Path) -> List[Tuple[Path, Path]]:
    return list_clone_items(out_dir, label="mos-scan")


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


def _mos_worker(
    rank: int,
    gpu: str,
    shard: list,
    out_dir: str,
    model_dir: str,
    details_path: str,
    no_sidecar: bool,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    out = Path(out_dir)
    details = Path(details_path)

    scorer = UTMosScorer(model_dir=model_dir, device="cuda:0")
    print(f"[mos-w{rank}] gpu={gpu} items={len(shard)}", flush=True)

    for wav_s, json_s in tqdm(shard, desc=f"mos-w{rank}", position=rank):
        wav_path = Path(wav_s)
        json_path = Path(json_s)
        rel = wav_path.relative_to(out)
        dataset = rel.parts[0] if rel.parts else "unknown"
        record = {
            "cloned_audio": str(wav_path),
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": scorer.model_dir,
            "worker": rank,
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
        if not no_sidecar:
            write_mos_json(json_path, record)
        append_jsonl(details, record)


def _run_single_process(items, args, summary_path, details_path):
    gpu_list = parse_gpu_list(args.gpus, args.gpu)
    shard = [(str(w), str(j)) for w, j in items]
    _mos_worker(0, gpu_list[0], shard, str(args.out_dir), str(args.model_dir),
                str(details_path), args.no_sidecar)
    results = _load_results_from_jsonl(details_path)
    summary = summarize(results)
    summary.update(
        metric="UTMOS",
        model_dir=str(args.model_dir),
        evaluated_at=datetime.now().isoformat(),
        out_dir=str(args.out_dir),
        sample_size=args.sample_size,
        seed=args.seed if args.sample_size else None,
        workers=1,
        gpus=gpu_list,
        items_done=len(results),
        items_total=len(items),
    )
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
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

    items = find_cloned_wavs(args.out_dir)
    if not items:
        print("No cloned audio found.", flush=True)
        return

    if args.skip_existing:
        before = len(items)
        items = [(w, j) for w, j in items if not j.with_suffix(".mos.json").exists()]
        print(f"Skip-existing: {before - len(items)} already done, {len(items)} remaining", flush=True)
        if not items:
            print("All items already evaluated.", flush=True)
            return

    if args.sample_size is not None and args.sample_size < len(items):
        items = random.Random(args.seed).sample(items, args.sample_size)
        print(f"Using sample of {len(items)} files (seed={args.seed})", flush=True)

    tag = f"_{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_mos_summary{tag}.json"
    details_path = args.out_dir / f"eval_mos_details{tag}.jsonl"
    if not args.skip_existing:
        if details_path.exists():
            details_path.unlink()
        for p in args.out_dir.glob(f"eval_mos_details{tag}.w*.jsonl"):
            p.unlink()

    print(f"Found {len(items)} wavs | workers={workers} gpus={gpu_list}", flush=True)

    if workers == 1:
        summary = _run_single_process(items, args, summary_path, details_path)
    else:
        shards = split_shards(items, workers)
        shard_strs = [[(str(w), str(j)) for w, j in s] for s in shards]
        part_paths = [
            args.out_dir / f"eval_mos_details{tag}.w{i}.jsonl" for i in range(workers)
        ]
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(workers):
            if not shard_strs[i]:
                continue
            gpu = gpu_list[i % len(gpu_list)]
            p = ctx.Process(
                target=_mos_worker,
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
                raise SystemExit(f"MOS worker exited with code {p.exitcode}")

        merge_jsonl_parts(part_paths, details_path)
        for p in part_paths:
            p.unlink(missing_ok=True)

        results = _load_results_from_jsonl(details_path)
        summary = summarize(results)
        summary.update(
            metric="UTMOS",
            model_dir=str(args.model_dir),
            evaluated_at=datetime.now().isoformat(),
            out_dir=str(args.out_dir),
            sample_size=args.sample_size,
            seed=args.seed if args.sample_size else None,
            workers=workers,
            gpus=gpu_list,
            items_done=len(results),
            items_total=len(items),
        )
        write_json(summary_path, summary)

    ov = summary["overall"]
    print(f"\n{'='*60}", flush=True)
    print(f"Overall UTMOS (n={ov['count']}, failed={summary['failed_count']})", flush=True)
    if ov["count"]:
        print(f"  mean={ov['mean']:.3f}  p50={ov['p50']:.3f}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {details_path}", flush=True)


if __name__ == "__main__":
    main()
