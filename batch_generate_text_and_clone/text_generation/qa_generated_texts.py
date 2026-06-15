#!/usr/bin/env python3
"""Run lightweight QA on generated clone text JSONL."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_generate_texts as gen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="?",
        default="",
        help="Generated JSONL path. Defaults to llm_children_100k_asr_complete.jsonl in output dir.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=int(os.environ.get("GEN_TOTAL_TARGET", "100000")),
        help="Expected final text count for below-target warning.",
    )
    args = parser.parse_args()

    config = gen.apply_config_from_env(gen.GenConfig(total_target=args.target))
    jsonl_path = args.jsonl or os.path.join(
        config.output_dir,
        "llm_children_100k_asr_complete.jsonl",
    )
    texts = gen.load_checkpoint(jsonl_path)
    report = gen.generation_quality_report(texts, config)
    gen.print_generation_quality_report(report)

    if report["warnings"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
