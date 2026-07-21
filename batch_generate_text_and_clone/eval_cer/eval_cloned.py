#!/usr/bin/env python3
"""Evaluate all cloned audios with deterministic, offline CER normalization."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_batch_200 as eb  # noqa: E402
from eval_common import (  # noqa: E402
    CerAccumulator,
    list_clone_items,
    load_inventory_wav_list,
    validate_sim_selection_manifest,
    write_json,
)

CER_METRIC = eb.CER_METRIC
CER_SCORE_VERSION = eb.CER_SCORE_VERSION
NORMALIZATION_VERSION = eb.NORMALIZATION_VERSION
normalization_fingerprint = eb.normalization_fingerprint


def find_all_cloned(out_dir: Path, allow_partial: bool = False):
    return list_clone_items(out_dir, label="cer-scan", allow_partial=allow_partial)


def write_eval_json(json_path: Path, record: dict) -> None:
    write_json(json_path.with_suffix(".eval.json"), record)


def wav_signature(wav_path: Path) -> dict:
    return eb.file_signature(wav_path)


def _load_clone_metadata(json_path: Path) -> dict | None:
    try:
        record = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    text = record.get("gen_text")
    if not isinstance(text, str) or not text:
        return None
    return {
        "gen_text": text,
        "language": record.get("language"),
        "lang_type": record.get("lang_type") or record.get("lang_key"),
    }


def _load_clone_text(json_path: Path) -> str | None:
    metadata = _load_clone_metadata(json_path)
    return metadata["gen_text"] if metadata is not None else None


def _load_eval_sidecar(json_path: Path) -> dict | None:
    try:
        value = json.loads(
            json_path.with_suffix(".eval.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def read_reusable_asr_hypothesis(wav_path: Path, json_path: Path) -> str | None:
    """Read ASR text from any current sidecar without trusting its CER fields."""
    record = _load_eval_sidecar(json_path)
    if record is None:
        return None
    clone_text = _load_clone_text(json_path)
    if (
        clone_text is None
        or record.get("wav_path") != str(wav_path)
        or record.get("wav_signature") != wav_signature(wav_path)
        or record.get("gen_text") != clone_text
        or record.get("asr_model") != eb.QWEN3_ASR_LOCAL
        or record.get("asr_model_fingerprint") != eb.asr_model_fingerprint()
        or record.get("asr_decode_fingerprint") != eb.asr_decode_fingerprint()
    ):
        return None
    hypothesis = record.get("asr_hypo")
    return hypothesis if isinstance(hypothesis, str) and hypothesis else None


def read_valid_eval(wav_path: Path, json_path: Path) -> dict | None:
    """Return a current deterministic v4 sidecar, otherwise ``None``."""
    record = _load_eval_sidecar(json_path)
    clone = _load_clone_metadata(json_path)
    if record is None or clone is None:
        return None
    clone_text = clone["gen_text"]
    context = eb.reference_normalization_context(
        clone.get("language"), clone.get("lang_type"), text=clone_text
    )
    reference_input_fingerprint = eb.reference_normalization_input_fingerprint(
        clone_text,
        language=clone.get("language"),
        lang_type=clone.get("lang_type"),
    )
    expected = {
        "eval_schema_version": eb.EVAL_SCHEMA_VERSION,
        "stage": "complete",
        "wav_path": str(wav_path),
        "wav_signature": wav_signature(wav_path),
        "gen_text": clone_text,
        "gen_text_fingerprint": eb.gen_text_fingerprint(clone_text),
        "asr_model": eb.QWEN3_ASR_LOCAL,
        "asr_model_fingerprint": eb.asr_model_fingerprint(),
        "asr_decode_fingerprint": eb.asr_decode_fingerprint(),
        "cer_metric": CER_METRIC,
        "cer_score_version": CER_SCORE_VERSION,
        "normalization_profile": eb.NORMALIZATION_PROFILE,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_fingerprint": normalization_fingerprint(
            profile=eb.NORMALIZATION_PROFILE
        ),
        "reference_normalization_context": context,
        "reference_normalization_input_fingerprint": reference_input_fingerprint,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    for key in ("asr_hypo", "ref_normalized", "hypo_normalized"):
        if not isinstance(record.get(key), str):
            return None
    if record.get("hypothesis_normalization_input_fingerprint") != eb.gen_text_fingerprint(
        record["asr_hypo"]
    ):
        return None
    if record["ref_normalized"] != eb.normalize_reference(
        clone_text,
        profile=eb.NORMALIZATION_PROFILE,
        language=clone.get("language"),
        lang_type=clone.get("lang_type"),
    ) or record["hypo_normalized"] != eb.normalize_hypothesis(
        record["asr_hypo"], profile=eb.NORMALIZATION_PROFILE
    ):
        return None
    for key in (
        "cer",
        "substitutions",
        "insertions",
        "deletions",
        "reference_chars",
    ):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return None
    if (
        not record["ref_normalized"]
        or record["reference_chars"] != len(record["ref_normalized"])
    ):
        return None
    try:
        expected_cer = eb.calc_cer(record["ref_normalized"], record["hypo_normalized"])
    except (TypeError, ValueError):
        return None
    stored = (
        record["cer"],
        record["substitutions"],
        record["insertions"],
        record["deletions"],
        record["reference_chars"],
    )
    if not math.isclose(float(stored[0]), expected_cer[0], rel_tol=0.0, abs_tol=1e-12):
        return None
    if tuple(stored[1:]) != tuple(expected_cer[1:]):
        return None
    return record


def build_eval_record(item: dict, evaluated_at: str) -> dict:
    return {
        "eval_schema_version": eb.EVAL_SCHEMA_VERSION,
        "stage": "complete",
        "cer_metric": CER_METRIC,
        "cer_score_version": CER_SCORE_VERSION,
        "normalization_profile": eb.NORMALIZATION_PROFILE,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_fingerprint": normalization_fingerprint(
            profile=eb.NORMALIZATION_PROFILE
        ),
        "reference_normalization_context": item["reference_normalization_context"],
        "reference_normalization_input_fingerprint": item[
            "reference_normalization_input_fingerprint"
        ],
        "hypothesis_normalization_input_fingerprint": item[
            "hypothesis_normalization_input_fingerprint"
        ],
        "wav_path": item["wav"],
        "wav_signature": wav_signature(Path(item["wav"])),
        "ref_audio": item.get("ref_audio"),
        "gen_text": item["ref_start"],
        "gen_text_fingerprint": eb.gen_text_fingerprint(item["ref_start"]),
        "asr_model": eb.QWEN3_ASR_LOCAL,
        "asr_model_fingerprint": eb.asr_model_fingerprint(),
        "asr_decode_fingerprint": eb.asr_decode_fingerprint(),
        "asr_hypo": item["hypo_start"],
        "ref_normalized": item["ref_normalized"],
        "hypo_normalized": item["hypo_normalized"],
        "cer": item["cer"],
        "substitutions": item["substitutions"],
        "insertions": item["insertions"],
        "deletions": item["deletions"],
        "reference_chars": item["chars"],
        "evaluated_at": evaluated_at,
    }


def rebuild_cer_details(
    pairs: list[tuple[Path, Path]],
    details_path: Path,
) -> int:
    """Atomically rebuild the canonical v4 filter input from sidecars."""
    details_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = details_path.with_name(f".{details_path.name}.{os.getpid()}.tmp")
    written = 0
    try:
        with open(tmp, "w", encoding="utf-8") as output:
            for wav_path, json_path in pairs:
                record = read_valid_eval(wav_path, json_path)
                if record is None:
                    raise RuntimeError(
                        f"Cannot rebuild {details_path}: invalid v4 sidecar for {wav_path}"
                    )
                row = {
                    "eval_schema_version": eb.EVAL_SCHEMA_VERSION,
                    "wav": str(wav_path),
                    "cloned_audio": str(wav_path),
                    "cloned_audio_signature": wav_signature(wav_path),
                    "cer": record["cer"],
                    "cer_metric": CER_METRIC,
                    "cer_score_version": CER_SCORE_VERSION,
                    "normalization_profile": eb.NORMALIZATION_PROFILE,
                    "normalization_version": NORMALIZATION_VERSION,
                    "normalization_fingerprint": record["normalization_fingerprint"],
                    "reference_normalization_context": record[
                        "reference_normalization_context"
                    ],
                    "reference_normalization_input_fingerprint": record[
                        "reference_normalization_input_fingerprint"
                    ],
                    "hypothesis_normalization_input_fingerprint": record[
                        "hypothesis_normalization_input_fingerprint"
                    ],
                    "asr_model_fingerprint": record["asr_model_fingerprint"],
                    "asr_decode_fingerprint": record["asr_decode_fingerprint"],
                    "stage": "complete",
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(tmp, details_path)
    finally:
        tmp.unlink(missing_ok=True)
    return written


def load_canonical_detailed(pairs: list[tuple[Path, Path]]) -> list[dict]:
    detailed = []
    for wav_path, json_path in pairs:
        record = read_valid_eval(wav_path, json_path)
        if record is None:
            raise RuntimeError(f"Missing/invalid canonical CER sidecar for {wav_path}")
        detailed.append(
            {
                "wav": str(wav_path),
                "json": str(json_path),
                "name": wav_path.name,
                "ref_start": record["gen_text"],
                "hypo_start": record["asr_hypo"],
                "ref_normalized": record["ref_normalized"],
                "hypo_normalized": record["hypo_normalized"],
                "reference_normalization_context": record[
                    "reference_normalization_context"
                ],
                "reference_normalization_input_fingerprint": record[
                    "reference_normalization_input_fingerprint"
                ],
                "hypothesis_normalization_input_fingerprint": record[
                    "hypothesis_normalization_input_fingerprint"
                ],
                "cer": record["cer"],
                "substitutions": record["substitutions"],
                "insertions": record["insertions"],
                "deletions": record["deletions"],
                "chars": record["reference_chars"],
            }
        )
    return detailed


def write_canonical_reports(
    args,
    paths: dict,
    all_pairs: list[tuple[Path, Path]],
    evaluation_scope: dict,
    batch_size: int,
    evaluated_at: str,
    processed_this_run: int,
) -> None:
    detailed = load_canonical_detailed(all_pairs)
    summary = eb.summarize_cer(detailed)
    report = {
        "eval_schema_version": eb.EVAL_SCHEMA_VERSION,
        "stage": "complete",
        "out_dir": str(args.out_dir),
        "items_processed_this_run": processed_this_run,
        "items_total_canonical": len(detailed),
        **evaluation_scope,
        "batch_size": batch_size,
        "cer_metric": CER_METRIC,
        "cer_score_version": CER_SCORE_VERSION,
        "normalization_profile": eb.NORMALIZATION_PROFILE,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_fingerprint": normalization_fingerprint(
            profile=eb.NORMALIZATION_PROFILE
        ),
        "asr_model": eb.QWEN3_ASR_LOCAL,
        "asr_model_fingerprint": eb.asr_model_fingerprint(),
        "asr_decode_fingerprint": eb.asr_decode_fingerprint(),
        "cer": summary,
        "asr_cache": str(paths["asr_cache"]),
        "details_jsonl": str(paths["details_jsonl"]),
        "evaluated_at": evaluated_at,
    }
    write_json(paths["summary"], report)
    write_json(paths["summary_progress"], report)
    eb.write_details(
        paths["details"],
        f"Full deterministic CER ({len(detailed)} audios)",
        summary,
        detailed,
        evaluated_at,
    )
    print(
        f"Files: {len(detailed)} | processed_this_run={processed_this_run}",
        flush=True,
    )


def _seed_asr_from_sidecars(
    pairs: list[tuple[Path, Path]],
    results: dict,
    signatures: dict,
) -> int:
    added = 0
    for wav_path, json_path in pairs:
        key = str(wav_path)
        if results.get(key):
            continue
        hypothesis = read_reusable_asr_hypothesis(wav_path, json_path)
        if hypothesis:
            results[key] = hypothesis
            signatures[key] = wav_signature(wav_path)
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/root/group-shared/voiceprint/data/speech/voice_activity_detection/"
            "batch_cloned_voices_ommivoice_kids_finetuned"
        ),
    )
    parser.add_argument("--skip-asr", action="store_true", help="Require reusable ASR text")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--asr-batch-size", type=int, help="Alias for --batch-size")
    parser.add_argument("--cache-flush-every", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--wav-list",
        type=Path,
        help=(
            "Evaluate exactly the absolute WAV paths in this allowlist. Each path "
            "must belong to the current clone inventory under --out-dir."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help=(
            "Directory for scoped CER aggregate reports. Required with --wav-list "
            "so a subset run cannot overwrite full-inventory canonical reports."
        ),
    )
    parser.add_argument("--refresh-asr-cache", action="store_true")
    parser.add_argument(
        "--refresh-cer",
        action="store_true",
        help="Re-normalize and recalculate CER while reusing current ASR text",
    )
    args = parser.parse_args()
    batch_size = (
        args.asr_batch_size
        if args.asr_batch_size is not None
        else args.batch_size
    )
    if batch_size <= 0:
        parser.error("--batch-size/--asr-batch-size must be positive")
    if args.cache_flush_every <= 0:
        parser.error("--cache-flush-every must be positive")
    if args.skip_asr and args.refresh_asr_cache:
        parser.error("--skip-asr and --refresh-asr-cache are mutually exclusive")
    if args.skip_existing and (args.refresh_asr_cache or args.refresh_cer):
        parser.error("refresh options require the full inventory; remove --skip-existing")
    if (args.wav_list is None) != (args.report_dir is None):
        parser.error("--wav-list and --report-dir must be provided together")

    inventory_pairs = list(find_all_cloned(args.out_dir, allow_partial=args.allow_partial))
    if not inventory_pairs:
        raise RuntimeError("No cloned audio found")
    all_pairs = inventory_pairs
    evaluation_scope = {
        "evaluation_scope": "full_inventory",
        "wav_list": None,
        "wav_list_count": len(all_pairs),
        "wav_list_fingerprint": None,
    }
    if args.wav_list is not None:
        selected_paths, list_fingerprint = load_inventory_wav_list(
            args.wav_list,
            args.out_dir,
            (wav for wav, _ in inventory_pairs),
        )
        pair_by_wav = {
            str(wav.resolve(strict=True)): (wav, json_path)
            for wav, json_path in inventory_pairs
        }
        all_pairs = [pair_by_wav[str(wav)] for wav in selected_paths]
        selection_manifest = validate_sim_selection_manifest(
            args.wav_list,
            args.out_dir,
            selected_paths,
            list_fingerprint,
            len(inventory_pairs),
        )
        evaluation_scope = {
            "evaluation_scope": "wav_list",
            "wav_list": str(args.wav_list.resolve(strict=True)),
            "wav_list_count": len(all_pairs),
            "wav_list_fingerprint": list_fingerprint,
            "selection_manifest": str(
                args.wav_list.resolve(strict=True).with_suffix(
                    args.wav_list.suffix + ".manifest.json"
                )
            ),
            "selection_threshold": selection_manifest[
                "min_raw_cosine_similarity"
            ],
            "selection_operator": selection_manifest["similarity_operator"],
        }
        print(
            f"WAV-list scope: {len(all_pairs)} current inventory audios "
            f"({list_fingerprint})",
            flush=True,
        )
    paths = eb.eval_paths_full(args.out_dir)
    if args.report_dir is not None:
        report_dir = args.report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        paths.update(
            {
                "summary": report_dir / "eval_summary.json",
                "summary_progress": report_dir / "eval_summary_progress.json",
                "details_jsonl": report_dir / "eval_cer_details.jsonl",
                "details": report_dir / "eval_details.txt",
            }
        )
    evaluated_at = datetime.now().isoformat()

    pairs = list(all_pairs)
    if args.skip_existing:
        pairs = [pair for pair in pairs if read_valid_eval(*pair) is None]
        print(
            f"Skip-existing: {len(all_pairs) - len(pairs)} already done, "
            f"{len(pairs)} remaining",
            flush=True,
        )
    if not pairs:
        count = rebuild_cer_details(all_pairs, paths["details_jsonl"])
        write_canonical_reports(
            args,
            paths,
            all_pairs,
            evaluation_scope,
            batch_size,
            evaluated_at,
            processed_this_run=0,
        )
        print(f"All {count} audios already evaluated; rebuilt canonical outputs.")
        return

    asr_results, asr_signatures = eb.load_valid_asr_cache(
        paths["asr_cache"], paths["asr_cache_meta"], inventory_pairs
    )
    if args.refresh_asr_cache:
        refresh_keys = {str(wav) for wav, _ in all_pairs}
        for key in refresh_keys:
            asr_results.pop(key, None)
            asr_signatures.pop(key, None)
    else:
        reused = _seed_asr_from_sidecars(all_pairs, asr_results, asr_signatures)
        if reused:
            print(f"Reused ASR text from {reused} current sidecars", flush=True)
            eb.write_asr_cache(
                paths["asr_cache"],
                paths["asr_cache_meta"],
                asr_results,
                asr_signatures,
            )

    if args.skip_asr:
        eb.validate_asr_cache(pairs, asr_results, require_complete=True)
    asr = None
    if not args.skip_asr and any(not asr_results.get(str(wav)) for wav, _ in pairs):
        asr = eb.load_asr_model(batch_size)

    accumulator = CerAccumulator()
    processed = 0
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    try:
        for batch_index in tqdm(range(num_batches), desc="ASR + deterministic CER"):
            batch = pairs[batch_index * batch_size : (batch_index + 1) * batch_size]
            missing = [pair for pair in batch if not asr_results.get(str(pair[0]))]
            if missing:
                if asr is None:
                    raise RuntimeError(
                        "Current ASR text is unavailable; rerun without --skip-asr"
                    )
                eb.transcribe_asr_batch(asr, missing, asr_results, asr_signatures)
            eb.validate_asr_cache(batch, asr_results, require_complete=True)
            for wav_path, json_path in batch:
                item = eb.build_eval_item(
                    wav_path, json_path, asr_results[str(wav_path)]
                )
                write_eval_json(json_path, build_eval_record(item, evaluated_at))
                accumulator.add(
                    item["substitutions"],
                    item["insertions"],
                    item["deletions"],
                    item["chars"],
                )
                processed += 1
            if (
                (batch_index + 1) % args.cache_flush_every == 0
                or batch_index + 1 == num_batches
            ):
                eb.write_asr_cache(
                    paths["asr_cache"],
                    paths["asr_cache_meta"],
                    asr_results,
                    asr_signatures,
                )
                write_json(
                    paths["summary_progress"],
                    {
                        "eval_schema_version": eb.EVAL_SCHEMA_VERSION,
                        "stage": "running",
                        "items_done": processed,
                        "items_total": len(pairs),
                        **evaluation_scope,
                        "cer_metric": CER_METRIC,
                        "cer_score_version": CER_SCORE_VERSION,
                        "normalization_profile": eb.NORMALIZATION_PROFILE,
                        "normalization_fingerprint": normalization_fingerprint(
                            profile=eb.NORMALIZATION_PROFILE
                        ),
                        "cer": accumulator.to_dict(),
                        "evaluated_at": datetime.now().isoformat(),
                    },
                )
    finally:
        if asr is not None:
            del asr
            torch.cuda.empty_cache()

    canonical_count = rebuild_cer_details(all_pairs, paths["details_jsonl"])
    write_canonical_reports(
        args,
        paths,
        all_pairs,
        evaluation_scope,
        batch_size,
        evaluated_at,
        processed_this_run=processed,
    )
    print(f"Canonical CER records: {canonical_count}", flush=True)


if __name__ == "__main__":
    main()
