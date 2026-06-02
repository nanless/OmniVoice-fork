#!/usr/bin/env python3
"""Shared helpers for batch clone evaluation (fast scan + incremental I/O)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

SIDEcar_SUFFIXES = (".eval.json", ".sim.json", ".mos.json")
SKIP_DIRS = {"logs", "__pycache__", "eval_sim_embedding_cache"}


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_gpu_list(gpus: str | None = None, gpu: int | None = None) -> List[str]:
    """Parse GPU id list from --gpus string or single --gpu int."""
    if gpus:
        return [g.strip() for g in gpus.split(",") if g.strip()]
    if gpu is not None:
        return [str(gpu)]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [g.strip() for g in env.split(",") if g.strip()] or ["0"]


def split_shards(items: list, num_workers: int) -> List[list]:
    """Round-robin split into num_workers shards."""
    if num_workers <= 1:
        return [items]
    shards: List[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        shards[i % num_workers].append(item)
    return shards


def merge_jsonl_parts(parts: List[Path], out: Path) -> int:
    """Concat worker jsonl shards into one file; return line count."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as dst:
        for part in sorted(parts):
            if not part.exists():
                continue
            with open(part, encoding="utf-8") as src:
                for line in src:
                    dst.write(line)
                    n += 1
    return n


class CerAccumulator:
    """Running char-level CER stats for incremental summary."""

    def __init__(self):
        self.sub = self.ins = self.del_ = self.chars = self.count = 0

    def add(self, substitutions: int, insertions: int, deletions: int, chars: int):
        self.sub += substitutions
        self.ins += insertions
        self.del_ += deletions
        self.chars += chars
        self.count += 1

    def to_dict(self) -> dict:
        weighted = (self.sub + self.ins + self.del_) / self.chars * 100 if self.chars else 0.0
        return {
            "count": self.count,
            "weighted_cer": weighted,
            "total_substitutions": self.sub,
            "total_insertions": self.ins,
            "total_deletions": self.del_,
            "total_chars": self.chars,
        }


def _is_clone_sidecar(name: str) -> bool:
    if not name.startswith("text_") or not name.endswith(".json"):
        return False
    return not any(name.endswith(s) for s in SIDEcar_SUFFIXES)


def _scan_dir(root: str) -> List[Tuple[str, str, dict]]:
    """Scan a single directory tree for clone records."""
    results = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_clone_sidecar(name):
                continue
            json_path = Path(dirpath) / name
            wav_path = json_path.with_suffix(".wav")
            if not wav_path.is_file():
                continue
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("status") != "generated":
                continue
            results.append((str(wav_path), str(json_path), meta))
    return results


def iter_clone_records(out_dir: Path, workers: int = 8) -> Iterator[Tuple[Path, Path, Dict[str, Any]]]:
    """Yield (wav_path, sidecar_json, meta) for status=generated clones."""
    out_dir = Path(out_dir)
    t0 = time.time()

    # Collect immediate subdirs to scan in parallel
    subdirs = [str(p) for p in out_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    total = 0

    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_dir, sd): sd for sd in subdirs}
            for future in as_completed(futures):
                batch = future.result()
                total += len(batch)
                if total % 50000 < len(batch):
                    print(f"[scan] {total} items … {time.time() - t0:.1f}s", flush=True)
                for wav_s, json_s, meta in batch:
                    yield Path(wav_s), Path(json_s), meta
    else:
        # Fallback to single-threaded for small trees
        for wav_s, json_s, meta in _scan_dir(str(out_dir)):
            yield Path(wav_s), Path(json_s), meta


def list_clone_items(out_dir: Path, label: str = "scan", scan_workers: int = 8) -> List[Tuple[Path, Path]]:
    t0 = time.time()
    items = [(w, j) for w, j, _ in iter_clone_records(out_dir, workers=scan_workers)]
    print(f"[{label}] {len(items)} clones in {time.time() - t0:.1f}s", flush=True)
    return items


def list_clone_pairs(out_dir: Path, label: str = "scan", scan_workers: int = 8) -> List[Tuple[Path, Path, Path]]:
    t0 = time.time()
    pairs = []
    for cloned, json_path, meta in iter_clone_records(out_dir, workers=scan_workers):
        ref = meta.get("ref_audio")
        if ref and Path(ref).is_file():
            pairs.append((cloned, Path(ref), json_path))
    print(f"[{label}] {len(pairs)} sim pairs in {time.time() - t0:.1f}s", flush=True)
    return pairs
