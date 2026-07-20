#!/usr/bin/env python3
"""Pure-stdlib provenance contract shared by CER producers and consumers."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

QWEN3_ASR_LOCAL = "/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local"
ASR_DECODE_CONFIG = {
    # The corpus contains pure English and heavy Chinese/English mixing.  A
    # forced language hint is part of the metric and must never be relabeled.
    "language": None,
    "return_time_stamps": False,
    "max_new_tokens": 256,
}


def json_fingerprint(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@lru_cache(maxsize=1)
def asr_model_signature() -> dict[str, dict[str, int]]:
    root = Path(QWEN3_ASR_LOCAL)
    files: list[Path] = []
    for pattern in ("config.json", "*.safetensors", "*.safetensors.index.json"):
        files.extend(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ASR model files found under {root}")
    return {path.name: file_signature(path) for path in sorted(set(files))}


@lru_cache(maxsize=1)
def asr_model_fingerprint() -> str:
    return json_fingerprint(asr_model_signature())


def asr_decode_fingerprint() -> str:
    return json_fingerprint(ASR_DECODE_CONFIG)


__all__ = [
    "ASR_DECODE_CONFIG",
    "QWEN3_ASR_LOCAL",
    "asr_decode_fingerprint",
    "asr_model_fingerprint",
    "asr_model_signature",
    "file_signature",
    "json_fingerprint",
]
