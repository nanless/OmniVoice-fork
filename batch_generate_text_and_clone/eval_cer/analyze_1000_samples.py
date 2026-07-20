#!/usr/bin/env python3
"""Disabled legacy LLM/manual-CER analyzer; deterministic CER v4 is authoritative."""

import sys


def main() -> None:
    print(
        "This legacy LLM/manual-CER analyzer is disabled. "
        "Use eval_cloned.py outputs and analyze_distributions.py instead.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
