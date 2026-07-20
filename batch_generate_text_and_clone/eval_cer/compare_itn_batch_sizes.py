#!/usr/bin/env python3
"""Disabled legacy LLM-ITN experiment; deterministic CER v4 is authoritative."""

import sys


def main() -> None:
    print(
        "This legacy LLM-ITN experiment is disabled. "
        "Use check_cer_normalization.py and eval_batch_200.py instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
