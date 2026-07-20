#!/usr/bin/env python3
"""Shared helpers for batch clone evaluation (fast scan + incremental I/O)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

SIDEcar_SUFFIXES = (".eval.json", ".cer.json", ".sim.json", ".mos.json")
SKIP_DIRS = {"logs", "__pycache__", "eval_sim_embedding_cache"}


def write_json(path: Path, data: dict) -> None:
    """Write JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_clone_record(
    wav_path: Path,
    json_path: Path,
    meta: dict,
) -> str | None:
    """Return an error for an unsafe clone record, otherwise None."""
    if meta.get("status") != "generated":
        return f"status={meta.get('status')!r}"
    for field in ("gen_text", "text_id", "model"):
        if not isinstance(meta.get(field), str) or not meta[field]:
            return f"missing/invalid {field}"
    if not isinstance(meta.get("model_signature"), dict) or not meta["model_signature"]:
        return "missing/invalid model_signature"
    if not isinstance(meta.get("generation_config"), dict) or not meta["generation_config"]:
        return "missing/invalid generation_config"
    if meta.get("cloned_audio") != str(wav_path):
        return "cloned_audio path mismatch"
    if not wav_path.is_file():
        return "cloned WAV missing"
    ref = meta.get("ref_audio")
    if not ref or not Path(ref).is_file():
        return "reference audio missing"
    try:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnframes() <= 0:
                return "cloned WAV must be non-empty 16 kHz audio"
    except (OSError, EOFError, wave.Error):
        return "cloned WAV is unreadable"
    schema = meta.get("schema_version")
    if schema in (2, 3):
        if meta.get("cloned_audio_signature") != _file_signature(wav_path):
            return "cloned audio signature mismatch"
        if meta.get("ref_audio_signature") != _file_signature(Path(ref)):
            return "reference audio signature mismatch"
        if schema == 3:
            for field in ("plan_id", "round_id", "task_id", "speaker_key"):
                if not isinstance(meta.get(field), str) or not meta[field]:
                    return f"missing/invalid {field}"
    elif schema not in (None, 1):
        return f"unsupported clone schema_version={schema!r}"
    return None


def _scan_dir(root: str) -> tuple[List[Tuple[str, str, dict]], List[str]]:
    """Scan a single directory tree for clone records."""
    results = []
    errors = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_clone_sidecar(name):
                continue
            json_path = Path(dirpath) / name
            wav_path = json_path.with_suffix(".wav")
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"{json_path}: invalid clone JSON: {exc}")
                continue
            error = validate_clone_record(wav_path, json_path, meta)
            if error:
                errors.append(f"{json_path}: {error}")
                continue
            results.append((str(wav_path), str(json_path), meta))
        file_names = set(files)
        for name in files:
            if name.startswith("text_") and name.endswith(".wav"):
                json_name = name[:-4] + ".json"
                if json_name not in file_names:
                    errors.append(f"{Path(dirpath) / name}: clone JSON sidecar missing")
    return results, errors


def iter_clone_records(
    out_dir: Path,
    workers: int = 8,
    allow_partial: bool = False,
) -> Iterator[Tuple[Path, Path, Dict[str, Any]]]:
    """Yield (wav_path, sidecar_json, meta) for status=generated clones."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Clone output directory not found: {out_dir}")
    t0 = time.time()
    errors: List[str] = []

    # Collect immediate subdirs to scan in parallel
    subdirs = [str(p) for p in out_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    total = 0

    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_dir, sd): sd for sd in subdirs}
            for future in as_completed(futures):
                batch, batch_errors = future.result()
                errors.extend(batch_errors)
                total += len(batch)
                if total % 50000 < len(batch):
                    print(f"[scan] {total} items … {time.time() - t0:.1f}s", flush=True)
                for wav_s, json_s, meta in batch:
                    yield Path(wav_s), Path(json_s), meta
    else:
        # Fallback to single-threaded for small trees
        batch, batch_errors = _scan_dir(str(out_dir))
        errors.extend(batch_errors)
        for wav_s, json_s, meta in batch:
            yield Path(wav_s), Path(json_s), meta

    if errors:
        preview = "\n  ".join(errors[:10])
        message = f"Invalid/incomplete clone records: {len(errors)}\n  {preview}"
        if allow_partial:
            print(f"WARNING: {message}", file=sys.stderr, flush=True)
        else:
            raise RuntimeError(message + "\nUse --allow-partial only for intentional partial evaluation.")


def list_clone_items(out_dir: Path, label: str = "scan", scan_workers: int = 8,
                     allow_partial: bool = False) -> List[Tuple[Path, Path]]:
    t0 = time.time()
    items = [(w, j) for w, j, _ in iter_clone_records(
        out_dir, workers=scan_workers, allow_partial=allow_partial
    )]
    print(f"[{label}] {len(items)} clones in {time.time() - t0:.1f}s", flush=True)
    return items


def list_clone_pairs(out_dir: Path, label: str = "scan", scan_workers: int = 8,
                     allow_partial: bool = False) -> List[Tuple[Path, Path, Path]]:
    t0 = time.time()
    pairs = []
    for cloned, json_path, meta in iter_clone_records(
        out_dir, workers=scan_workers, allow_partial=allow_partial
    ):
        ref = meta.get("ref_audio")
        if ref and Path(ref).is_file():
            pairs.append((cloned, Path(ref), json_path))
    print(f"[{label}] {len(pairs)} sim pairs in {time.time() - t0:.1f}s", flush=True)
    return pairs
