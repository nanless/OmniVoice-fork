#!/usr/bin/env python3
"""Batch adult text generation for OmniVoice voice cloning."""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omnivoice_adult_text_gen.checkpoint import (
    build_duplicate_index,
    deduplicate,
    filter_incremental_duplicates,
    is_task_complete,
    load_checkpoint,
    load_task_status,
    save_checkpoint,
    save_task_status,
    semantic_deduplicate,
    update_task_status,
)
from omnivoice_adult_text_gen.config import GenConfig, apply_config_from_env
from omnivoice_adult_text_gen.diversity import build_suppression_hint
from omnivoice_adult_text_gen.output import (
    ensure_text_ids,
    generation_quality_report,
    print_generation_quality_report,
    print_statistics,
    save_jsonl,
)
from omnivoice_adult_text_gen.quality_filter import quality_filter
from omnivoice_adult_text_gen.task_generator import generate_task_list
from omnivoice_adult_text_gen.worker import worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--task-status", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = apply_config_from_env(GenConfig(
        total_target=args.total,
        batch_size=args.batch_size,
        max_workers=args.workers,
        temperature=args.temperature,
        seed=args.seed or 42,
    ))

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    output_path = args.output or os.path.join(out_dir, "llm_adult_generated.jsonl")
    checkpoint_path = args.checkpoint or os.path.join(out_dir, ".checkpoint_adult.jsonl")
    task_status_path = args.task_status or os.path.join(out_dir, ".task_status_adult.json")

    api_key = (
        config.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )

    all_texts = load_checkpoint(checkpoint_path) if args.resume else []
    print(f"Loaded {len(all_texts)} from checkpoint")

    task_config = replace(
        config,
        total_target=max(config.total_target, int(config.total_target * config.oversample_ratio)),
    )
    tasks = generate_task_list(task_config)
    task_status = load_task_status(task_status_path) if args.resume else {}
    completed_ids = {
        int(tid)
        for tid, rec in task_status.items()
        if is_task_complete(rec, config.batch_size)
    }
    pending = [t for t in tasks if t["task_id"] not in completed_ids]
    print(f"Tasks: {len(tasks)} total, {len(pending)} pending, target={config.total_target}")

    seen_norm, dup_ctx = build_duplicate_index(all_texts)
    completed = failed = skipped = 0
    start = time.time()

    if pending:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {}
            next_idx = 0

            def submit():
                nonlocal next_idx
                if next_idx >= len(pending):
                    return
                task = dict(pending[next_idx])
                task["suppression_hint"] = build_suppression_hint(
                    all_texts, config.suppression_window_size
                )
                futures[pool.submit(worker, task, config)] = task
                next_idx += 1

            for _ in range(min(config.max_workers, len(pending))):
                submit()

            while futures:
                for future in as_completed(list(futures)):
                    task = futures.pop(future)
                    break
                try:
                    results = future.result(timeout=180)
                    raw_count = len(results or [])
                    if results:
                        results, dup_skipped = filter_incremental_duplicates(
                            results, seen_norm, dup_ctx,
                            same_context_threshold=config.same_context_dup_threshold,
                        )
                        skipped += dup_skipped
                        all_texts.extend(results)
                        update_task_status(
                            task_status, task, "completed",
                            raw_count=raw_count,
                            accepted_count=len(results),
                            skipped_duplicates=dup_skipped,
                        )
                        completed += 1
                    else:
                        update_task_status(task_status, task, "empty")
                        failed += 1
                except Exception as exc:
                    print(f"Task {task['task_id']} failed: {exc}")
                    update_task_status(task_status, task, "failed", error=str(exc))
                    failed += 1

                if (completed + failed) % 5 == 0:
                    elapsed = time.time() - start
                    print(
                        f"Progress: ok={completed} fail={failed} texts={len(all_texts)} "
                        f"skip={skipped} rate={len(all_texts)/max(1,elapsed):.1f}/s"
                    )
                    save_checkpoint(all_texts, checkpoint_path)
                    save_task_status(task_status, task_status_path)
                submit()

    save_checkpoint(all_texts, checkpoint_path)
    save_task_status(task_status, task_status_path)

    if not args.no_postprocess:
        print("\nPost-processing...")
        all_texts = deduplicate(all_texts)
        print(f"After exact dedup: {len(all_texts)}")
        all_texts = semantic_deduplicate(all_texts, config.semantic_dedup_threshold)
        print(f"After semantic dedup: {len(all_texts)}")
        all_texts = quality_filter(
            all_texts,
            max_tags_per_text=config.max_tags_per_text,
            max_same_tag_repeat=config.max_same_tag_repeat,
        )
        print(f"After quality filter: {len(all_texts)}")
        all_texts = all_texts[: config.total_target]

    ensure_text_ids(all_texts)
    print_generation_quality_report(generation_quality_report(all_texts, config))
    save_jsonl(all_texts, output_path)
    print(f"Saved {len(all_texts)} -> {output_path}")
    print_statistics(all_texts)


if __name__ == "__main__":
    main()
