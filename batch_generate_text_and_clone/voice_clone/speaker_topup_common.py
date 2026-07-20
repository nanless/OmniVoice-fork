#!/usr/bin/env python3
"""Shared, dependency-light helpers for speaker-duration top-up tools."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

TARGET_DATASETS = (
    "childmandarin",
    "chineseenglishchildren",
    "kingasr612",
    "king-asr-725",
    "speechocean762",
)
AUDIO_SUFFIXES = {".wav"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: object, length: int = 24) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def duration_fraction(path: Path) -> Fraction:
    """Read exact frames/sample-rate duration without decoding audio."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            samplerate = handle.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"Unreadable audio header: {path}: {exc}") from exc
    if frames <= 0 or samplerate <= 0:
        raise ValueError(
            f"Invalid audio header: {path}: frames={frames}, samplerate={samplerate}"
        )
    return Fraction(frames, samplerate)


def parallel_durations(paths: Iterable[Path], scan_workers: int = 16) -> list[Fraction]:
    """Read WAV durations with bounded concurrency and stable result order."""
    if scan_workers <= 0:
        raise ValueError(f"scan_workers must be positive, got {scan_workers}")
    ordered_paths = list(paths)
    if scan_workers == 1 or len(ordered_paths) <= 1:
        return [duration_fraction(path) for path in ordered_paths]
    results: list[Fraction] = []
    next_index = 0
    window_size = min(len(ordered_paths), scan_workers * 4)
    with ThreadPoolExecutor(max_workers=scan_workers, thread_name_prefix="wav-header") as pool:
        pending = deque()
        while next_index < window_size:
            pending.append(pool.submit(duration_fraction, ordered_paths[next_index]))
            next_index += 1
        while pending:
            # Futures are consumed in submission order, so aggregation is deterministic.
            results.append(pending.popleft().result())
            if next_index < len(ordered_paths):
                pending.append(pool.submit(duration_fraction, ordered_paths[next_index]))
                next_index += 1
    return results


def fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "seconds": float(value),
    }


def fraction_from_record(record: dict[str, Any]) -> Fraction:
    try:
        return Fraction(int(record["numerator"]), int(record["denominator"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid exact duration record: {record!r}") from exc


def iter_wavs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {root}")
    yield from sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def discover_target_speakers(original_root: Path) -> dict[str, Path]:
    """Return authoritative target speaker keys from the five dataset trees."""
    speakers: dict[str, Path] = {}
    for dataset in TARGET_DATASETS:
        dataset_dir = original_root / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Required original dataset directory not found: {dataset_dir}")
        for speaker_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            key = f"{dataset}/{speaker_dir.name}"
            speakers[key] = speaker_dir
        if not any(key.startswith(f"{dataset}/") for key in speakers):
            raise ValueError(f"No speaker directories found under {dataset_dir}")
    return speakers


def inventory_original(
    original_root: Path, scan_workers: int = 16,
) -> tuple[dict[str, Fraction], dict[str, int], dict[str, Path]]:
    speakers = discover_target_speakers(original_root)
    durations = {key: Fraction(0) for key in speakers}
    counts = {key: 0 for key in speakers}
    jobs: list[tuple[str, Path]] = []
    for key, speaker_dir in speakers.items():
        for wav in iter_wavs(speaker_dir):
            jobs.append((key, wav))
            counts[key] += 1
    for (key, _), duration in zip(jobs, parallel_durations(
        (wav for _, wav in jobs), scan_workers
    )):
        durations[key] += duration
    return durations, counts, speakers


def inventory_accepted(
    roots: Iterable[Path],
    target_speakers: dict[str, Path],
    seen_audio: set[Path] | None = None,
    scan_workers: int = 16,
) -> tuple[dict[str, Fraction], dict[str, int], set[Path]]:
    """Inventory accepted roots laid out as <dataset>/<speaker>/... ."""
    durations = {key: Fraction(0) for key in target_speakers}
    counts = {key: 0 for key in target_speakers}
    seen = seen_audio if seen_audio is not None else set()
    jobs: list[tuple[str, Path]] = []
    for root in roots:
        root = root.resolve()
        for wav in iter_wavs(root):
            rel = wav.relative_to(root)
            if len(rel.parts) < 3:
                raise ValueError(
                    f"Accepted audio must be under <dataset>/<speaker>/: {wav}"
                )
            key = f"{rel.parts[0]}/{rel.parts[1]}"
            if key not in target_speakers:
                raise ValueError(f"Accepted audio maps to unknown target speaker {key}: {wav}")
            real = wav.resolve()
            if real in seen:
                raise ValueError(f"Audio counted by more than one inventory root: {real}")
            seen.add(real)
            jobs.append((key, wav))
            counts[key] += 1
    for (key, _), duration in zip(jobs, parallel_durations(
        (wav for _, wav in jobs), scan_workers
    )):
        durations[key] += duration
    return durations, counts, seen


def inventory_generated(
    roots: Iterable[Path],
    target_speakers: dict[str, Path],
    seen_audio: set[Path] | None = None,
    scan_workers: int = 16,
) -> tuple[dict[str, Fraction], dict[str, int], set[Path]]:
    """Count valid plan-generated WAVs using schema-v3 sidecar speaker identity."""
    durations = {key: Fraction(0) for key in target_speakers}
    counts = {key: 0 for key in target_speakers}
    seen = seen_audio if seen_audio is not None else set()
    jobs: list[tuple[str, Path]] = []
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Generated root not found: {root}")
        for sidecar in sorted(root.rglob("text_plan_*.json")):
            try:
                record = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid generated sidecar {sidecar}: {exc}") from exc
            if record.get("schema_version") != 3 or record.get("status") != "generated":
                continue
            key = record.get("speaker_key")
            if key not in target_speakers:
                raise ValueError(f"Generated sidecar has unknown speaker_key {key!r}: {sidecar}")
            wav_value = record.get("cloned_audio")
            if not isinstance(wav_value, str) or not wav_value:
                raise ValueError(f"Generated sidecar has no cloned_audio: {sidecar}")
            wav = Path(wav_value)
            if not wav.is_file():
                raise FileNotFoundError(f"Generated sidecar WAV not found: {wav}")
            real = wav.resolve()
            try:
                real.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Generated sidecar points outside its root: {sidecar} -> {real}") from exc
            stat = real.stat()
            if record.get("cloned_audio_signature") != {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns
            }:
                raise ValueError(f"Generated WAV signature mismatch: {sidecar}")
            if real in seen:
                raise ValueError(f"Audio counted by more than one inventory root: {real}")
            seen.add(real)
            jobs.append((key, wav))
            counts[key] += 1
    for (key, _), duration in zip(jobs, parallel_durations(
        (wav for _, wav in jobs), scan_workers
    )):
        durations[key] += duration
    return durations, counts, seen


def canonical_speaker_key(dataset_root: Path, source_speaker: str, ref_audio: str) -> str:
    """Map source Kaldi speaker IDs to the merged dataset's canonical key."""
    name = dataset_root.name.lower()
    speaker = source_speaker.strip()
    upper = speaker.upper()
    ref_norm = ref_audio.replace("\\", "/")

    if "baai" in name or "childmandarin" in name:
        token = speaker.rsplit("_", 1)[-1]
        if not token.isdigit():
            raise ValueError(f"Unrecognized ChildMandarin speaker {speaker!r}")
        return f"childmandarin/childmandarin_{int(token):03d}"

    if "chinese_english" in name or "scripted_speech_corpus_children" in name:
        token = speaker.rsplit("_", 1)[-1]
        if not token.upper().startswith("G"):
            raise ValueError(f"Unrecognized Chinese-English speaker {speaker!r}")
        return f"chineseenglishchildren/chineseenglishchildren_{token}"

    if "king-asr-en-kid" in name:
        marker = "SPEAKER"
        pos = upper.rfind(marker)
        if pos < 0 or not upper[pos + len(marker):].isdigit():
            raise ValueError(f"Unrecognized King-ASR speaker {speaker!r}")
        number = upper[pos + len(marker):]
        is_612 = "612" in upper[:pos] or "King-ASR-612" in ref_norm
        if is_612:
            return f"kingasr612/kingasr612_{number}"
        return f"king-asr-725/king-asr-725_SPEAKER{number}"

    if "speechocean762" in name:
        token = speaker.rsplit("_", 1)[-1]
        if token.upper().startswith("SPEAKER"):
            token = token[7:]
        if not token.isdigit():
            raise ValueError(f"Unrecognized speechocean762 speaker {speaker!r}")
        return f"speechocean762/speechocean762_{token}"

    raise ValueError(f"Unsupported source dataset: {dataset_root}")


def read_kaldi_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected two fields")
            key, value = parts
            if key in result:
                raise ValueError(f"{path}:{line_no}: duplicate key {key!r}")
            result[key] = value
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
