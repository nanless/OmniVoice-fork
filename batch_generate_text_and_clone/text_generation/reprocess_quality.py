#!/usr/bin/env python3
"""Re-run quality_filter on a raw snapshot without regenerating LLM batches."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_generate_texts as gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default=".raw_before_quality.jsonl",
        help="Path to raw snapshot (default: .raw_before_quality.jsonl in output dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("GEN_OUTPUT_DIR", ""),
        help="Preview output directory (or set GEN_OUTPUT_DIR)",
    )
    parser.add_argument("--target", type=int, default=int(os.environ.get("GEN_TOTAL_TARGET", "500")))
    args = parser.parse_args()

    out_dir = args.output_dir or gen._DEFAULT_OUTPUT_DIR
    raw_path = args.raw
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(out_dir, raw_path)

    config = gen.apply_config_from_env(gen.GenConfig(total_target=args.target))
    if args.output_dir:
        config.output_dir = args.output_dir

    texts = gen.load_checkpoint(raw_path)
    print(f"Loaded {len(texts)} from {raw_path}")
    deduped = gen.semantic_deduplicate(gen.deduplicate(texts), config.semantic_dedup_threshold)
    print(f"After dedup: {len(deduped)}")
    print(f"Reject breakdown: {dict(gen.diagnose_quality_rejections(deduped))}")
    filtered = gen.quality_filter(deduped, config.reject_severe_length_mismatch)
    print(f"After quality filter: {len(filtered)}")
    if len(filtered) < config.total_target:
        filtered = gen.refill_to_target(filtered, config, checkpoint_path=raw_path)
    print(f"After refill: {len(filtered)}")
    out_jsonl = os.path.join(config.output_dir, "llm_children_100k_asr_complete.jsonl")
    gen.save_checkpoint(filtered[: config.total_target], out_jsonl)
    print(f"Saved: {out_jsonl}")


if __name__ == "__main__":
    main()
