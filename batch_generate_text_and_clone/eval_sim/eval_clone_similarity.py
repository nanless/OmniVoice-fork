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
import hashlib
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
from metric_contract import (  # noqa: E402
    is_complete_raw_cosine_record,
    similarity_metadata,
    validate_raw_cosine_record,
)
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
        "/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned",
    )
)


def find_clone_pairs(out_dir: Path, allow_partial: bool = False) -> List[Tuple[Path, Path, Path]]:
    return list_clone_pairs(
        out_dir, label="sim-scan", scan_workers=8, allow_partial=allow_partial
    )


def write_sim_json(json_path: Path, record: dict):
    write_json(json_path.with_suffix(".sim.json"), record)


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
        **similarity_metadata(),
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


def _file_signature(path: Path, *, include_sha256: bool = False) -> dict:
    stat = path.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        signature["sha256"] = digest.hexdigest()
    return signature


def _model_signature(model_dir: Path) -> dict:
    """Stable cache identity for the local model config and checkpoint."""
    signature = {}
    for name in ("config.yaml", "avg_model.pt"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required similarity model file not found: {path}")
        signature[name] = _file_signature(path, include_sha256=True)
    return signature


def _load_sidecar_record(
    path: Path,
    model_dir: Path,
    *,
    require_complete: bool,
    expected_cloned: Path | None = None,
    expected_ref: Path | None = None,
    model_signature: dict | None = None,
) -> dict | None:
    """Load a reusable v2 sidecar, rejecting legacy scores and other models."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        validate_raw_cosine_record(record, path)
    except ValueError:
        return None
    if record.get("model_dir") != str(model_dir):
        return None
    if model_signature is not None and record.get("model_signature") != model_signature:
        return None
    if expected_cloned is not None:
        if record.get("cloned_audio") != str(expected_cloned):
            return None
        try:
            if record.get("cloned_audio_signature") != _file_signature(expected_cloned):
                return None
        except OSError:
            return None
    if expected_ref is not None:
        if record.get("ref_audio") != str(expected_ref):
            return None
        try:
            if record.get("ref_audio_signature") != _file_signature(expected_ref):
                return None
        except OSError:
            return None
    if require_complete and not is_complete_raw_cosine_record(record):
        return None
    return record


def _collect_sidecar_results(pairs: list, model_dir: Path, model_signature: dict) -> list:
    """Rebuild aggregate results from the authoritative per-item sidecars."""
    results = []
    missing = []
    for cloned, ref, json_path in pairs:
        sidecar = json_path.with_suffix(".sim.json")
        record = _load_sidecar_record(
            sidecar,
            model_dir,
            require_complete=False,
            expected_cloned=cloned,
            expected_ref=ref,
            model_signature=model_signature,
        )
        if record is None:
            missing.append(str(sidecar))
        else:
            results.append(record)
    if missing:
        preview = "\n  ".join(missing[:5])
        raise RuntimeError(
            f"Missing or incompatible v2 similarity sidecars: {len(missing)}\n  {preview}"
        )
    return results


def _write_jsonl_atomic(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _sim_worker(
    rank: int,
    gpu: str,
    shard: list,
    out_dir: str,
    model_dir: str,
    details_path: str,
    no_sidecar: bool,
    model_signature: dict,
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
            **similarity_metadata(),
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": model_dir,
            "model_signature": model_signature,
            "cloned_audio_signature": _file_signature(cloned_wav),
            "ref_audio_signature": _file_signature(ref_audio),
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


def _run_single_process(pairs, args, details_path, model_signature):
    gpu_list = parse_gpu_list(args.gpus, args.gpu if args.gpu is not None else 0)
    shard = [(str(c), str(r), str(j)) for c, r, j in pairs]
    _sim_worker(0, gpu_list[0], shard, str(args.out_dir), str(args.model_dir),
                str(details_path), args.no_sidecar, model_signature)


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
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.model_dir = args.model_dir.resolve()

    if args.skip_existing and args.no_sidecar:
        parser.error("--skip-existing requires sidecars; remove --no-sidecar")

    gpu_list = parse_gpu_list(args.gpus, args.gpu if args.gpu is not None else 0)
    workers = max(1, args.workers)
    model_signature = _model_signature(args.model_dir)

    all_pairs = find_clone_pairs(args.out_dir, allow_partial=args.allow_partial)
    if not all_pairs:
        print("No cloned audio pairs found.", flush=True)
        return

    selected_pairs = all_pairs
    if args.sample_size is not None and args.sample_size < len(selected_pairs):
        selected_pairs = random.Random(args.seed).sample(selected_pairs, args.sample_size)
        print(f"Using sample of {len(selected_pairs)} pairs (seed={args.seed})", flush=True)

    pairs = selected_pairs
    reused_count = 0
    if args.skip_existing:
        pending = []
        for pair in selected_pairs:
            sidecar = pair[2].with_suffix(".sim.json")
            record = _load_sidecar_record(
                sidecar,
                args.model_dir,
                require_complete=True,
                expected_cloned=pair[0],
                expected_ref=pair[1],
                model_signature=model_signature,
            )
            if record is None:
                pending.append(pair)
            else:
                reused_count += 1
        pairs = pending
        print(
            f"Skip-existing: {reused_count} current raw-cosine v2 results reused, "
            f"{len(pairs)} remaining (legacy scores are recomputed)",
            flush=True,
        )

    tag = f"_{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_sim_summary{tag}.json"
    details_path = args.out_dir / f"eval_sim_details{tag}.jsonl"
    run_details_path = details_path.with_suffix(details_path.suffix + ".current")
    run_details_path.unlink(missing_ok=True)
    for p in args.out_dir.glob(f"eval_sim_details{tag}.w*.jsonl.current"):
        p.unlink()

    print(
        f"Selected {len(selected_pairs)} pairs | evaluate={len(pairs)} "
        f"reuse={reused_count} workers={workers} gpus={gpu_list}",
        flush=True,
    )

    if pairs and workers == 1:
        _run_single_process(pairs, args, run_details_path, model_signature)
    elif pairs:
        shards = split_shards(pairs, workers)
        shard_strs = [[(str(c), str(r), str(j)) for c, r, j in s] for s in shards]
        part_paths = [
            args.out_dir / f"eval_sim_details{tag}.w{i}.jsonl.current" for i in range(workers)
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
                    str(part_paths[i]), args.no_sidecar, model_signature,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"SIM worker exited with code {p.exitcode}")

        merge_jsonl_parts(part_paths, run_details_path)
        for p in part_paths:
            p.unlink(missing_ok=True)


    if args.no_sidecar:
        results = _load_results_from_jsonl(run_details_path)
        for i, record in enumerate(results, 1):
            validate_raw_cosine_record(record, f"{run_details_path}:{i}")
    else:
        results = _collect_sidecar_results(selected_pairs, args.model_dir, model_signature)
    failed_scores = [r for r in results if r.get("similarity") is None]
    if failed_scores and not args.allow_partial:
        raise RuntimeError(
            f"SIM produced {len(failed_scores)} failed/null scores; canonical aggregate not replaced"
        )
    _write_jsonl_atomic(details_path, results)
    run_details_path.unlink(missing_ok=True)

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
        items_total=len(selected_pairs),
        items_evaluated=len(pairs),
        items_reused=reused_count,
    )
    write_json(summary_path, summary)

    ov = summary["overall"]
    print(f"\n{'='*60}", flush=True)
    print(
        f"Overall raw cosine similarity [-1, 1] "
        f"(n={ov['count']}, failed={summary['failed_count']})",
        flush=True,
    )
    if ov["count"]:
        print(f"  mean={ov['mean']:.4f}  p50={ov['p50']:.4f}  p10={ov['p10']:.4f}  p90={ov['p90']:.4f}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {details_path}", flush=True)


if __name__ == "__main__":
    main()
