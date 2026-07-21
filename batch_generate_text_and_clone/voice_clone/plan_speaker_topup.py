#!/usr/bin/env python3
"""Create a deterministic, resumable per-speaker clone top-up plan."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from speaker_topup_common import (
    canonical_json,
    canonical_speaker_key,
    fraction_record,
    inventory_accepted,
    inventory_strict_accepted,
    inventory_original,
    iter_wavs,
    parallel_durations,
    read_kaldi_map,
    stable_hash,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_ORIGINAL_ROOT = Path(
    "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
    "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_"
    "withclone_addlibrilight_1130/audio"
)
DEFAULT_SOURCE_DATASETS = (
    Path("/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth_onlyenhanced"),
    Path("/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth_onlyenhanced"),
    Path("/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth_onlyenhanced"),
    Path("/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth_onlyenhanced"),
)
DEFAULT_TEXTS_PATH = Path(
    "/root/code/github_repos/OmniVoice-fork/batch_generated_text/"
    "llm_children_100k_asr_complete.jsonl"
)
PLAN_SCHEMA_VERSION = 1


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_texts(path: Path) -> list[dict[str, Any]]:
    texts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_no}: missing non-empty text")
            texts.append({
                "id": item.get("id") or item.get("text_id") or f"text_{line_no:06d}",
                "text": text,
                "language": item.get("language", "zh"),
                "lang_type": item.get("lang_type"),
                "length_type": item.get("length_type"),
                "scenario": item.get("scenario") or item.get("scenario_key"),
                "subscene": item.get("subscene"),
                "emotion": item.get("emotion"),
                "age_tier": item.get("age_tier"),
                "text_tn": item.get("text_tn"),
                "source_task_id": item.get("task_id"),
            })
    if not texts:
        raise ValueError(f"No usable texts found in {path}")
    return texts


def load_references(
    dataset_roots: list[Path], target_speakers: set[str], scan_workers: int = 16,
) -> dict[str, list[dict[str, Any]]]:
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duration_jobs: list[tuple[dict[str, Any], Path]] = []
    seen_utterances: set[tuple[str, str]] = set()
    for dataset_root in dataset_roots:
        dataset_root = dataset_root.resolve()
        kaldi_dir = dataset_root / "kaldi_files"
        wav_map = read_kaldi_map(kaldi_dir / "wav.scp")
        text_map = read_kaldi_map(kaldi_dir / "text")
        speaker_map = read_kaldi_map(kaldi_dir / "utt2spk")
        if set(wav_map) - set(speaker_map):
            missing = sorted(set(wav_map) - set(speaker_map))[:5]
            raise ValueError(f"{kaldi_dir}/utt2spk missing utterances: {missing}")
        for utt_id in sorted(wav_map):
            identity = (dataset_root.name, utt_id)
            if identity in seen_utterances:
                raise ValueError(f"Duplicate source utterance identity: {identity}")
            seen_utterances.add(identity)
            raw_path = Path(wav_map[utt_id])
            ref_audio = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            ref_audio = ref_audio.resolve()
            if not ref_audio.is_file():
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
            speaker_key = canonical_speaker_key(
                dataset_root, speaker_map[utt_id], str(ref_audio)
            )
            if speaker_key not in target_speakers:
                raise ValueError(
                    f"Source utterance maps to unknown target speaker {speaker_key}: {utt_id}"
                )
            try:
                relative = ref_audio.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(
                    f"Reference audio must be inside source dataset root: {ref_audio}"
                ) from exc
            transcript = text_map.get(utt_id, "").strip()
            row = {
                "utt_id": utt_id,
                "ref_audio": str(ref_audio),
                "ref_audio_signature": file_signature(ref_audio),
                "ref_text": transcript,
                "source_dataset": dataset_root.name,
                "source_relative_path": relative.as_posix(),
            }
            references[speaker_key].append(row)
            duration_jobs.append((row, ref_audio))

    for (row, _), duration in zip(duration_jobs, parallel_durations(
        (path for _, path in duration_jobs), scan_workers
    )):
        row["ref_duration"] = fraction_record(duration)

    for speaker_key, rows in references.items():
        rows.sort(key=lambda row: (
            not bool(row["ref_text"]),
            not (3.0 <= row["ref_duration"]["seconds"] <= 15.0),
            abs(row["ref_duration"]["seconds"] - 7.0),
            stable_hash(speaker_key, row["utt_id"], row["ref_audio"]),
        ))
    return references


def coprime_step(size: int, speaker_key: str) -> tuple[int, int]:
    if size == 1:
        return 0, 1
    offset = int(stable_hash("text-offset", speaker_key), 16) % size
    step = 1 + int(stable_hash("text-step", speaker_key), 16) % (size - 1)
    while math.gcd(step, size) != 1:
        step = 1 + (step % (size - 1))
    return offset, step


def load_prior_plan_history(
    paths: list[Path],
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, int], list[dict[str, Any]]]:
    used: dict[str, set[tuple[str, str]]] = defaultdict(set)
    next_ordinals: dict[str, int] = defaultdict(int)
    signatures = []
    seen_tasks: set[str] = set()
    for path in paths:
        path = path.resolve(strict=True)
        signatures.append({"path": str(path), "signature": file_signature(path)})
        with open(path, encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                if row.get("record_type") == "plan_meta":
                    continue
                if row.get("record_type") != "clone_task":
                    raise ValueError(f"{path}:{line_no}: unexpected prior-plan record")
                speaker = row.get("speaker_key")
                ref_audio = row.get("ref_audio")
                text_id = row.get("text_id")
                task_id = row.get("task_id")
                if not all(isinstance(value, str) and value for value in (
                    speaker, ref_audio, text_id, task_id
                )):
                    raise ValueError(f"{path}:{line_no}: invalid prior-plan identity")
                if task_id in seen_tasks:
                    raise ValueError(f"Duplicate task across prior plans: {task_id}")
                seen_tasks.add(task_id)
                pair = (str(Path(ref_audio).resolve()), text_id)
                if pair in used[speaker]:
                    raise ValueError(
                        f"Repeated reference/text pair in prior plans for {speaker}: {pair}"
                    )
                used[speaker].add(pair)
                ordinal = row.get("speaker_global_ordinal", row.get("task_ordinal"))
                if not isinstance(ordinal, int) or ordinal < 0:
                    raise ValueError(f"{path}:{line_no}: invalid prior task ordinal")
                next_ordinals[speaker] = max(next_ordinals[speaker], ordinal + 1)
    return used, next_ordinals, signatures


def apply_sim_history_ranking(
    references: dict[str, list[dict[str, Any]]],
    paths: list[Path],
    threshold: float,
    reference_limit: int,
) -> dict[str, Any]:
    ref_to_speaker = {
        row["ref_audio"]: speaker
        for speaker, rows in references.items()
        for row in rows
    }
    stats: dict[str, list[float]] = defaultdict(list)
    seen_clones: set[str] = set()
    rows_loaded = 0
    failed_rows = 0
    for path in paths:
        path = path.resolve(strict=True)
        with open(path, encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                clone = row.get("cloned_audio") or row.get("wav")
                ref_audio = row.get("ref_audio")
                score = row.get("similarity")
                if not isinstance(clone, str) or not clone:
                    raise ValueError(f"{path}:{line_no}: missing cloned audio")
                if clone in seen_clones:
                    raise ValueError(f"Duplicate clone in SIM history: {clone}")
                seen_clones.add(clone)
                if ref_audio not in ref_to_speaker:
                    raise ValueError(f"{path}:{line_no}: unknown reference audio {ref_audio!r}")
                if score is None:
                    failed_rows += 1
                    continue
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not -1.0 <= score <= 1.0
                ):
                    raise ValueError(f"{path}:{line_no}: invalid raw cosine {score!r}")
                stats[ref_audio].append(float(score))
                rows_loaded += 1
    refs_with_pass = 0
    speakers_with_pass = 0
    speakers_with_observed_fallback = 0
    reference_count_before = sum(map(len, references.values()))
    for rows in references.values():
        def rank(row: dict[str, Any]):
            scores = stats.get(row["ref_audio"], [])
            passed = sum(score > threshold for score in scores)
            if passed:
                nonlocal refs_with_pass
                refs_with_pass += 1
            return (
                -(passed / len(scores) if scores else 0.0),
                -passed,
                -max(scores) if scores else 1.0,
                -(sum(scores) / len(scores)) if scores else 1.0,
                not bool(row["ref_text"]),
                not (3.0 <= row["ref_duration"]["seconds"] <= 15.0),
                abs(row["ref_duration"]["seconds"] - 7.0),
                row["ref_audio"],
            )
        rows.sort(key=rank)
        passed_rows = [
            row for row in rows
            if any(score > threshold for score in stats.get(row["ref_audio"], []))
        ]
        if passed_rows:
            speakers_with_pass += 1
            rows[:] = passed_rows[:reference_limit]
            continue
        observed_rows = [row for row in rows if stats.get(row["ref_audio"])]
        if observed_rows:
            speakers_with_observed_fallback += 1
            rows[:] = observed_rows[:reference_limit]
    return {
        "history_files": [str(path.resolve(strict=True)) for path in paths],
        "history_rows": rows_loaded,
        "history_failed_rows_ignored": failed_rows,
        "references_with_sim_pass": refs_with_pass,
        "speakers_with_sim_pass": speakers_with_pass,
        "speakers_with_observed_fallback": speakers_with_observed_fallback,
        "reference_selection_policy": "top_passed_else_top_observed",
        "reference_limit_per_speaker": reference_limit,
        "reference_count_before": reference_count_before,
        "reference_count_after": sum(map(len, references.values())),
        "threshold": threshold,
    }


def task_for(
    round_id: str,
    speaker_key: str,
    round_task_ordinal: int,
    speaker_global_ordinal: int,
    references: list[dict[str, Any]],
    texts: list[dict[str, Any]],
    baseline: Fraction,
    target_seconds: Fraction,
) -> dict[str, Any]:
    ref_index = speaker_global_ordinal % len(references)
    cycle = speaker_global_ordinal // len(references)
    offset, step = coprime_step(len(texts), speaker_key)
    text_index = (offset + ref_index + cycle * step) % len(texts)
    reference = references[ref_index]
    text = texts[text_index]
    task_id = stable_hash(
        "speaker-topup-v2", round_id, speaker_key, speaker_global_ordinal,
        reference["utt_id"],
        text["id"], text["text"],
    )
    rel = Path(reference["source_relative_path"])
    output_relpath = (
        Path(reference["source_dataset"]) / rel.parent / rel.stem /
        f"text_plan_{task_id}.wav"
    ).as_posix()
    return {
        "record_type": "clone_task",
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "task_id": task_id,
        "speaker_key": speaker_key,
        "task_ordinal": round_task_ordinal,
        "round_task_ordinal": round_task_ordinal,
        "speaker_global_ordinal": speaker_global_ordinal,
        "target_duration": fraction_record(target_seconds),
        "baseline_duration": fraction_record(baseline),
        "expected_output_relpath": output_relpath,
        **reference,
        "text_id": text["id"],
        "gen_text": text["text"],
        "language": text["language"],
        "lang_type": text["lang_type"],
        "length_type": text["length_type"],
        "scenario": text["scenario"],
        "subscene": text["subscene"],
        "emotion": text["emotion"],
        "age_tier": text["age_tier"],
        "gen_text_tn": text["text_tn"],
        "source_text_task_id": text["source_task_id"],
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = [
        "speaker_key", "original_seconds", "accepted_seconds", "baseline_seconds",
        "deficit_seconds", "planned_raw_seconds", "generation_target_seconds",
        "speaker_global_ordinal_start", "speaker_global_ordinal_next",
        "planned_tasks", "reference_count",
    ]
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument(
        "--accepted-root", type=Path, action="append", required=True,
        help="Quality-accepted clone root in <dataset>/<speaker>/ layout; repeatable",
    )
    parser.add_argument(
        "--strict-accepted-root", type=Path, action="append", default=[],
        help=(
            "Transactional top-up root; only rounds with valid commits are counted; "
            "repeatable"
        ),
    )
    parser.add_argument(
        "--source-dataset", type=Path, action="append", dest="source_datasets",
        help="Source dataset with kaldi_files; repeat to override the four defaults",
    )
    parser.add_argument("--texts-path", type=Path, default=DEFAULT_TEXTS_PATH)
    parser.add_argument(
        "--prior-plan-jsonl", type=Path, action="append", default=[],
        help="Prior immutable plan; repeatable and used to prevent ref/text reuse",
    )
    parser.add_argument(
        "--history-sim-details", type=Path, action="append", default=[],
        help="Prior raw-cosine aggregate used only to rank productive references",
    )
    parser.add_argument(
        "--history-sim-threshold", type=float, default=0.8,
        help="Strict raw-cosine threshold used for reference-history ranking",
    )
    parser.add_argument(
        "--history-reference-limit", type=int, default=8,
        help=(
            "Maximum productive references retained per speaker when SIM history "
            "is available"
        ),
    )
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument(
        "--round-id", required=True,
        help="Unique stable round name (for example 20260720_r01); reused runs stay idempotent",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--target-seconds", type=Fraction, default=Fraction(1800))
    parser.add_argument(
        "--planning-seconds-per-clone", type=Fraction, default=Fraction(8),
        help="Conservative sizing assumption; workers stop from measured WAV durations",
    )
    parser.add_argument(
        "--generation-multiplier", type=Fraction, default=Fraction(1),
        help=(
            "Raw generation budget as a multiple of the current accepted-duration "
            "deficit; acceptance target remains --target-seconds"
        ),
    )
    parser.add_argument("--reserve-tasks", type=int, default=16)
    parser.add_argument("--max-tasks-per-speaker", type=int, default=10000)
    parser.add_argument(
        "--scan-workers", type=int, default=16,
        help="Concurrent WAV-header readers for inventory and source references",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing report files")
    args = parser.parse_args()
    if args.target_seconds <= 0:
        parser.error("--target-seconds must be positive")
    if args.planning_seconds_per_clone <= 0:
        parser.error("--planning-seconds-per-clone must be positive")
    if args.generation_multiplier < 1:
        parser.error("--generation-multiplier must be at least 1")
    if (
        not math.isfinite(args.history_sim_threshold)
        or not -1.0 <= args.history_sim_threshold <= 1.0
    ):
        parser.error("--history-sim-threshold must be finite and in [-1, 1]")
    if args.reserve_tasks < 0:
        parser.error("--reserve-tasks must be non-negative")
    if args.history_reference_limit <= 0:
        parser.error("--history-reference-limit must be positive")
    if args.max_tasks_per_speaker <= 0:
        parser.error("--max-tasks-per-speaker must be positive")
    if args.scan_workers <= 0:
        parser.error("--scan-workers must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.round_id):
        parser.error("--round-id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    return args


def main() -> int:
    args = parse_args()
    summary_json = args.summary_json or args.plan_jsonl.with_suffix(".summary.json")
    summary_csv = args.summary_csv or args.plan_jsonl.with_suffix(".summary.csv")
    outputs = (args.plan_jsonl, summary_json, summary_csv)
    if not args.force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to replace existing output(s): {existing}")

    print("[1/4] Inventorying canonical original audio...", flush=True)
    original, original_counts, target_paths = inventory_original(
        args.original_root.resolve(), args.scan_workers
    )
    print(
        f"      speakers={len(target_paths)} files={sum(original_counts.values())}",
        flush=True,
    )
    original_audio = {
        wav.resolve() for speaker_dir in target_paths.values() for wav in iter_wavs(speaker_dir)
    }
    print("[2/4] Inventorying quality-accepted clone roots...", flush=True)
    accepted, accepted_counts, seen = inventory_accepted(
        [path.resolve() for path in args.accepted_root], target_paths, original_audio,
        args.scan_workers,
    )
    strict, strict_counts, _ = inventory_strict_accepted(
        [path.resolve() for path in args.strict_accepted_root], target_paths, seen,
        args.scan_workers,
    )
    for speaker_key in target_paths:
        accepted[speaker_key] += strict[speaker_key]
        accepted_counts[speaker_key] += strict_counts[speaker_key]
    print(f"      files={sum(accepted_counts.values())}", flush=True)
    print("[3/4] Loading generation texts and source references...", flush=True)
    texts = load_texts(args.texts_path.resolve())
    source_datasets = [path.resolve() for path in (args.source_datasets or DEFAULT_SOURCE_DATASETS)]
    references = load_references(source_datasets, set(target_paths), args.scan_workers)
    sim_history = None
    if args.history_sim_details:
        sim_history = apply_sim_history_ranking(
            references,
            args.history_sim_details,
            args.history_sim_threshold,
            args.history_reference_limit,
        )
    used_pairs, next_ordinals, prior_plan_signatures = load_prior_plan_history(
        args.prior_plan_jsonl
    )
    print(
        f"      texts={len(texts)} references={sum(map(len, references.values()))}",
        flush=True,
    )

    task_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    for speaker_key in sorted(target_paths):
        baseline = original[speaker_key] + accepted[speaker_key]
        deficit = max(Fraction(0), args.target_seconds - baseline)
        refs = references.get(speaker_key, [])
        if deficit and not refs:
            raise ValueError(f"Deficient target speaker has no source references: {speaker_key}")
        task_count = 0
        raw_budget = deficit * args.generation_multiplier
        generation_target = baseline + raw_budget
        ordinal_start = next_ordinals.get(speaker_key, 0)
        ordinal_next = ordinal_start
        if deficit:
            task_count = math.ceil(
                raw_budget / args.planning_seconds_per_clone
            ) + args.reserve_tasks
            if task_count > args.max_tasks_per_speaker:
                raise ValueError(
                    f"{speaker_key} requires {task_count} planned tasks, above "
                    f"--max-tasks-per-speaker={args.max_tasks_per_speaker}"
                )
            if task_count > len(refs) * len(texts):
                raise ValueError(f"Not enough unique reference/text pairs for {speaker_key}")
            speaker_tasks = []
            candidate_ordinal = ordinal_start
            used_for_speaker = used_pairs[speaker_key]
            while len(speaker_tasks) < task_count:
                if candidate_ordinal - ordinal_start >= len(refs) * len(texts):
                    raise ValueError(
                        f"Exhausted unique reference/text pairs for {speaker_key}"
                    )
                task = task_for(
                    args.round_id,
                    speaker_key,
                    len(speaker_tasks),
                    candidate_ordinal,
                    refs,
                    texts,
                    baseline,
                    generation_target,
                )
                task["acceptance_target_duration"] = fraction_record(
                    args.target_seconds
                )
                task["planned_raw_duration"] = fraction_record(raw_budget)
                pair = (str(Path(task["ref_audio"]).resolve()), task["text_id"])
                candidate_ordinal += 1
                if pair in used_for_speaker:
                    continue
                used_for_speaker.add(pair)
                speaker_tasks.append(task)
            ordinal_next = candidate_ordinal
            task_rows.extend(speaker_tasks)
        speaker_rows.append({
            "speaker_key": speaker_key,
            "original_seconds": float(original[speaker_key]),
            "accepted_seconds": float(accepted[speaker_key]),
            "baseline_seconds": float(baseline),
            "deficit_seconds": float(deficit),
            "planned_raw_seconds": float(raw_budget),
            "generation_target_seconds": float(generation_target),
            "speaker_global_ordinal_start": ordinal_start,
            "speaker_global_ordinal_next": ordinal_next,
            "planned_tasks": task_count,
            "reference_count": len(refs),
        })

    task_ids = [row["task_id"] for row in task_rows]
    output_paths = [row["expected_output_relpath"] for row in task_rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Planner produced duplicate task IDs")
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Planner produced duplicate expected output paths")

    identity = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "round_id": args.round_id,
        "target_duration": fraction_record(args.target_seconds),
        "planning_seconds_per_clone": fraction_record(args.planning_seconds_per_clone),
        "generation_multiplier": fraction_record(args.generation_multiplier),
        "reserve_tasks": args.reserve_tasks,
        "original_root": str(args.original_root.resolve()),
        "accepted_roots": sorted(str(path.resolve()) for path in args.accepted_root),
        "strict_accepted_roots": sorted(
            str(path.resolve()) for path in args.strict_accepted_root
        ),
        "source_datasets": [str(path) for path in source_datasets],
        "texts_path": str(args.texts_path.resolve()),
        "texts_signature": file_signature(args.texts_path.resolve()),
        "prior_plans": prior_plan_signatures,
        "sim_history": sim_history,
        "speakers": speaker_rows,
        "task_ids": task_ids,
    }
    plan_id = stable_hash(canonical_json(identity), length=32)
    for row in task_rows:
        row["plan_id"] = plan_id
    meta = {
        "record_type": "plan_meta",
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in identity.items() if key not in {"speakers", "task_ids"}},
        "scan_workers": args.scan_workers,
        "speaker_count": len(speaker_rows),
        "deficient_speaker_count": sum(row["deficit_seconds"] > 0 for row in speaker_rows),
        "task_count": len(task_rows),
    }
    summary = {
        **meta,
        "record_type": "plan_summary",
        "total_original_seconds": sum(row["original_seconds"] for row in speaker_rows),
        "total_accepted_seconds": sum(row["accepted_seconds"] for row in speaker_rows),
        "total_deficit_seconds": sum(row["deficit_seconds"] for row in speaker_rows),
        "speakers": speaker_rows,
        "original_file_count": sum(original_counts.values()),
        "accepted_file_count": sum(accepted_counts.values()),
    }
    print("[4/4] Writing immutable plan and summaries...", flush=True)
    write_jsonl_atomic(args.plan_jsonl, [meta, *task_rows])
    write_json_atomic(summary_json, summary)
    write_csv_atomic(summary_csv, speaker_rows)
    print(
        f"plan_id={plan_id} speakers={len(speaker_rows)} "
        f"deficient={meta['deficient_speaker_count']} tasks={len(task_rows)}"
    )
    print(f"plan={args.plan_jsonl}\nsummary={summary_json}\ncsv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
