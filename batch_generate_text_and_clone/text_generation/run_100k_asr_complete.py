#!/usr/bin/env python3
"""Run a fresh 100k text generation job with ASR/WER-friendly complete text."""

import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing sibling module when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_generate_texts as gen


def main():
    config = gen.GenConfig(total_target=100000, batch_size=10, max_workers=10)
    config = gen.apply_config_from_env(config)
    if os.environ.get("GEN_MODEL"):
        config.model = os.environ["GEN_MODEL"]
    config.truncate_overlength = False

    os.makedirs(config.output_dir, exist_ok=True)

    output_jsonl = os.path.join(config.output_dir, "llm_children_100k_asr_complete.jsonl")
    checkpoint_path = os.path.join(config.output_dir, ".checkpoint_100k_asr_complete.jsonl")
    task_status_path = os.path.join(config.output_dir, ".task_status_100k_asr_complete.json")

    api_key = (
        config.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )

    all_texts = gen.load_checkpoint(checkpoint_path)
    print(f"Loaded {len(all_texts)} existing texts from {checkpoint_path}", flush=True)

    seen_normalized, duplicate_context_index = gen.build_duplicate_index(all_texts)
    task_config = gen.replace(
        config,
        total_target=max(config.total_target, int(config.total_target * config.oversample_ratio)),
    )
    tasks = gen.generate_task_list(task_config)
    task_status = gen.load_task_status(task_status_path)
    completed_task_ids = {
        int(task_id)
        for task_id, rec in task_status.items()
        if gen.is_task_complete(rec, config.batch_size)
    }
    pending_tasks = [t for t in tasks if t["task_id"] not in completed_task_ids]

    print(f"Total tasks: {len(tasks)} batch_size={config.batch_size} target={config.total_target}", flush=True)
    print(f"Pending tasks: {len(pending_tasks)}", flush=True)
    print(f"Oversample ratio: {config.oversample_ratio:.2f}", flush=True)
    print(f"Seed: {config.seed}", flush=True)
    print(f"Output dir: {config.output_dir}", flush=True)
    print(f"Output: {output_jsonl}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Task status: {task_status_path}", flush=True)
    print(f"Truncate overlength: {config.truncate_overlength}", flush=True)
    print(f"Semantic dedup threshold: {config.semantic_dedup_threshold}", flush=True)
    print(f"Max workers: {config.max_workers}", flush=True)
    print(f"Generate text_tn: {config.generate_text_tn}", flush=True)

    completed = 0
    failed = 0
    skipped_duplicates = 0

    if pending_tasks:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_task = {}
            next_task_idx = 0

            def submit_next_task():
                nonlocal next_task_idx
                if next_task_idx >= len(pending_tasks):
                    return
                task_to_submit = dict(pending_tasks[next_task_idx])
                task_to_submit["suppression_hint"] = gen.build_frequency_suppression_hint(
                    all_texts,
                    window_size=config.suppression_window_size,
                )
                future = executor.submit(gen.worker, task_to_submit, config)
                future_to_task[future] = task_to_submit
                next_task_idx += 1

            for _ in range(min(config.max_workers, len(pending_tasks))):
                submit_next_task()

            while future_to_task:
                for future in as_completed(list(future_to_task), timeout=None):
                    task = future_to_task.pop(future)
                    break

                try:
                    results = future.result(timeout=180)
                    raw_count = len(results or [])
                    if results:
                        results, skipped = gen.filter_incremental_duplicates(
                            results,
                            seen_normalized,
                            duplicate_context_index,
                            same_context_threshold=config.same_context_dup_threshold,
                        )
                        skipped_duplicates += skipped
                        all_texts.extend(results)
                        gen.update_task_status(
                            task_status,
                            task,
                            "completed",
                            raw_count=raw_count,
                            accepted_count=len(results),
                            skipped_duplicates=skipped,
                        )
                        completed += 1
                    else:
                        gen.update_task_status(task_status, task, "empty")
                        failed += 1
                except Exception as exc:
                    print(f"Task {task['task_id']} failed: {exc}", flush=True)
                    gen.update_task_status(task_status, task, "failed", error=str(exc))
                    failed += 1

                done = completed + failed
                if done % 1 == 0 or done == len(pending_tasks):
                    print(
                        f"Progress: {completed} succeeded, {failed} failed, "
                        f"total_raw={len(all_texts)}, skipped_duplicates={skipped_duplicates}",
                        flush=True,
                    )
                    gen.save_checkpoint(all_texts, checkpoint_path)
                    gen.save_task_status(task_status, task_status_path)

                submit_next_task()

    gen.save_checkpoint(all_texts, checkpoint_path)
    gen.save_task_status(task_status, task_status_path)
    print(f"Raw checkpoint count: {len(all_texts)}", flush=True)
    print(f"Skipped near duplicates before checkpointing: {skipped_duplicates}", flush=True)

    raw_snapshot = os.path.join(config.output_dir, ".raw_before_quality.jsonl")
    gen.save_checkpoint(all_texts, raw_snapshot)
    print(f"Raw snapshot: {raw_snapshot} ({len(all_texts)} texts)", flush=True)

    print("Post-processing...", flush=True)
    processed = gen.deduplicate(all_texts)
    print(f"After exact dedup: {len(processed)}", flush=True)
    processed = gen.semantic_deduplicate(
        processed,
        threshold=config.semantic_dedup_threshold,
    )
    print(f"After semantic dedup: {len(processed)}", flush=True)
    reject_stats = gen.diagnose_quality_rejections(processed)
    print(f"Quality reject breakdown: {dict(reject_stats)}", flush=True)
    processed = gen.quality_filter(
        processed,
        reject_severe_length_mismatch=config.reject_severe_length_mismatch,
    )
    print(f"After quality filter: {len(processed)}", flush=True)
    processed = gen.refill_to_target(
        processed,
        config,
        checkpoint_path=checkpoint_path,
        future_timeout=180,
    )
    print(f"After refill: {len(processed)}", flush=True)

    gen.ensure_text_ids(processed)
    gen.print_generation_quality_report(gen.generation_quality_report(processed, config))
    gen.save_checkpoint(processed, output_jsonl)

    print("=== Final Statistics ===", flush=True)
    print(f"Total texts: {len(processed)}", flush=True)
    print(f"Length: {dict(Counter(t.get('length_type') for t in processed))}", flush=True)
    print(f"Language: {dict(Counter(t.get('lang_type') for t in processed))}", flush=True)
    print(f"Age: {dict(Counter(t.get('age_tier') for t in processed))}", flush=True)
    print(f"Scenario: {dict(Counter(t.get('scenario') for t in processed))}", flush=True)
    tag_stats = gen.analyze_tags(processed)
    print(
        f"Tag coverage: {tag_stats.get('tag_coverage', 0) * 100:.1f}% "
        f"avg_tags={tag_stats['avg_tags_per_text']:.2f}",
        flush=True,
    )
    print(f"Saved to: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
