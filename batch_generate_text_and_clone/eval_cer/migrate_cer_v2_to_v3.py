#!/usr/bin/env python3
"""Deprecated compatibility entrypoint; use ``migrate_cer_to_v4.py``."""

from migrate_cer_to_v4 import main


if __name__ == "__main__":
    print("WARNING: migrate_cer_v2_to_v3.py is deprecated; migrating to CER v4.")
    main()
