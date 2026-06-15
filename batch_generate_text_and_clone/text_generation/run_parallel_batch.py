#!/usr/bin/env python3
"""Multi-instance parallel text generation.

Spawns N subprocesses, each targeting a different vLLM port with a different
seed.  After all workers finish, merges results, deduplicates, runs quality
filter, and saves the final JSONL.

Usage:
    python run_parallel_batch.py --ports 8000,8001,8002,8003 --target 100000
    python run_parallel_batch.py --ports 8000,8001 --target 5000 --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_generate_texts as gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ports", default="8000,8001,8002,8003",
        help="Comma-separated vLLM ports (default: 8000,8001,8002,8003)",
    )
    parser.add_argument("--target", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    n_workers = len(ports)
    per_worker = (args.target + n_workers - 1) // n_workers

    config = gen.apply_config_from_env(gen.GenConfig(total_target=args.target))
    output_dir = args.output_dir or config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    worker_dirs = []
    processes = []
    script = str(Path(__file__).resolve().parent / "run_100k_asr_complete.py")

    print(f"Launching {n_workers} workers, {per_worker} texts each", flush=True)

    for i, port in enumerate(ports):
        wdir = os.path.join(output_dir, f".worker_{i}")
        os.makedirs(wdir, exist_ok=True)
        worker_dirs.append(wdir)

        env = os.environ.copy()
        env["GEN_TOTAL_TARGET"] = str(per_worker)
        env["GEN_SEED"] = str(args.seed + i * 10007)
        env["GEN_BATCH_SIZE"] = str(args.batch_size)
        env["GEN_MAX_WORKERS"] = str(args.max_workers)
        env["GEN_OUTPUT_DIR"] = wdir
        env["LLM_BASE_URL"] = f"http://localhost:{port}/v1"
        env.pop("LLM_BASE_URLS", None)

        proc = subprocess.Popen(
            [sys.executable, script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)
        print(f"  Worker {i}: port={port} seed={args.seed + i * 10007} dir={wdir}", flush=True)

    for i, proc in enumerate(processes):
        stdout, _ = proc.communicate()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"Worker {i} finished: {status}", flush=True)
        if proc.returncode != 0:
            log_path = os.path.join(worker_dirs[i], "worker_error.log")
            Path(log_path).write_bytes(stdout or b"")
            print(f"  Error log: {log_path}", flush=True)

    print("Merging results...", flush=True)
    all_texts = []
    for wdir in worker_dirs:
        jsonl = os.path.join(wdir, "llm_children_100k_asr_complete.jsonl")
        if os.path.isfile(jsonl):
            loaded = gen.load_checkpoint(jsonl)
            print(f"  {jsonl}: {len(loaded)} texts", flush=True)
            all_texts.extend(loaded)

    print(f"Total raw: {len(all_texts)}", flush=True)

    processed = gen.deduplicate(all_texts)
    print(f"After exact dedup: {len(processed)}", flush=True)
    processed = gen.semantic_deduplicate(
        processed, threshold=config.semantic_dedup_threshold,
    )
    print(f"After semantic dedup: {len(processed)}", flush=True)
    processed = gen.quality_filter(
        processed, reject_severe_length_mismatch=config.reject_severe_length_mismatch,
    )
    print(f"After quality filter: {len(processed)}", flush=True)

    if len(processed) > args.target:
        processed = processed[: args.target]

    gen.ensure_text_ids(processed)
    output_jsonl = os.path.join(output_dir, "llm_children_100k_asr_complete.jsonl")
    gen.save_checkpoint(processed, output_jsonl)
    print(f"Saved {len(processed)} texts to {output_jsonl}", flush=True)

    report = gen.generation_quality_report(processed, config)
    gen.print_generation_quality_report(report)


if __name__ == "__main__":
    main()
