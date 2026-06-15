#!/usr/bin/env python3
"""Production entry: 100k adult ASR-friendly texts for OmniVoice cloning."""

import os
import sys
from collections import Counter
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
    save_jsonl,
)
from omnivoice_adult_text_gen.quality_filter import quality_filter
from omnivoice_adult_text_gen.task_generator import generate_task_list
from omnivoice_adult_text_gen.worker import worker


def main():
    config = apply_config_from_env(GenConfig(total_target=100000, batch_size=10, max_workers=10))
    if os.environ.get("GEN_MODEL"):
        config.model = os.environ["GEN_MODEL"]

    os.makedirs(config.output_dir, exist_ok=True)
    output_jsonl = os.path.join(config.output_dir, "llm_adult_100k_asr_complete.jsonl")
    checkpoint_path = os.path.join(config.output_dir, ".checkpoint_adult_100k.jsonl")
    task_status_path = os.path.join(config.output_dir, ".task_status_adult_100k.json")
    raw_snapshot = os.path.join(config.output_dir, ".raw_adult_before_quality.jsonl")

    api_key = (
        config.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )

    all_texts = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(all_texts)} existing texts", flush=True)

    seen_norm, dup_ctx = build_duplicate_index(all_texts)
    task_config = replace(
        config,
        total_target=max(config.total_target, int(config.total_target * config.oversample_ratio)),
    )
    tasks = generate_task_list(task_config)
    task_status = load_task_status(task_status_path)
    completed_ids = {
        int(tid) for tid, rec in task_status.items() if is_task_complete(rec, config.batch_size)
    }
    pending = [t for t in tasks if t["task_id"] not in completed_ids]

    print(f"Target={config.total_target} tasks={len(tasks)} pending={len(pending)}", flush=True)
    print(f"Output: {output_jsonl}", flush=True)

    completed = failed = skipped = 0
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
                            raw_count=raw_count, accepted_count=len(results),
                            skipped_duplicates=dup_skipped,
                        )
                        completed += 1
                    else:
                        update_task_status(task_status, task, "empty")
                        failed += 1
                except Exception as exc:
                    print(f"Task {task['task_id']} failed: {exc}", flush=True)
                    update_task_status(task_status, task, "failed", error=str(exc))
                    failed += 1

                if (completed + failed) % 1 == 0:
                    print(
                        f"Progress: ok={completed} fail={failed} raw={len(all_texts)} skip={skipped}",
                        flush=True,
                    )
                    save_checkpoint(all_texts, checkpoint_path)
                    save_task_status(task_status, task_status_path)
                submit()

    save_checkpoint(all_texts, checkpoint_path)
    save_task_status(task_status, task_status_path)
    save_checkpoint(all_texts, raw_snapshot)

    print("Post-processing...", flush=True)
    processed = deduplicate(all_texts)
    processed = semantic_deduplicate(processed, config.semantic_dedup_threshold)
    processed = quality_filter(
        processed,
        max_tags_per_text=config.max_tags_per_text,
        max_same_tag_repeat=config.max_same_tag_repeat,
    )
    processed = processed[: config.total_target]

    ensure_text_ids(processed)
    print_generation_quality_report(generation_quality_report(processed, config))
    save_jsonl(processed, output_jsonl)

    print("=== Final Statistics ===", flush=True)
    print(f"Total: {len(processed)}", flush=True)
    print(f"Length: {dict(Counter(t.get('length_type') for t in processed))}", flush=True)
    print(f"Lang: {dict(Counter(t.get('lang_type') for t in processed))}", flush=True)
    print(f"Scenario: {dict(Counter(t.get('scenario') for t in processed))}", flush=True)
    print(f"Saved: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
