"""Checkpoint and dedup helpers."""

import sys
from pathlib import Path

_CHILD_GEN = Path(__file__).resolve().parents[2] / "text_generation"
if str(_CHILD_GEN) not in sys.path:
    sys.path.insert(0, str(_CHILD_GEN))

from llm_generate_texts import (  # noqa: E402
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

__all__ = [
    "build_duplicate_index",
    "deduplicate",
    "filter_incremental_duplicates",
    "is_task_complete",
    "load_checkpoint",
    "load_task_status",
    "save_checkpoint",
    "save_task_status",
    "semantic_deduplicate",
    "update_task_status",
]
