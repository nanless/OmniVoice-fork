#!/usr/bin/env python3
"""QA report for adult generated JSONL."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omnivoice_adult_text_gen.checkpoint import load_checkpoint
from omnivoice_adult_text_gen.config import GenConfig, apply_config_from_env
from omnivoice_adult_text_gen.output import generation_quality_report, print_generation_quality_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="?", default="")
    parser.add_argument("--target", type=int, default=int(os.environ.get("GEN_TOTAL_TARGET", "100000")))
    args = parser.parse_args()

    config = apply_config_from_env(GenConfig(total_target=args.target))
    path = args.jsonl or os.path.join(config.output_dir, "llm_adult_100k_asr_complete.jsonl")
    texts = load_checkpoint(path)
    report = generation_quality_report(texts, config)
    print_generation_quality_report(report)
    if report["warnings"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
