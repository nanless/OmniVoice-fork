#!/usr/bin/env python3
"""Evaluate a fixed sample with deterministic, offline CER normalization.

The complete evaluation path is ASR -> independent reference/hypothesis
normalization -> character error rate.  It has no network endpoint or
reference-guided hypothesis selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

import cer_normalization as _normalization
import eval_contract as _contract

warnings.filterwarnings("ignore")

EVAL_DIR = Path(__file__).resolve().parent
QWEN3_ASR_LOCAL = _contract.QWEN3_ASR_LOCAL
OUT_DIR = Path(
    "/root/group-shared/voiceprint/data/speech/voice_activity_detection/"
    "batch_cloned_voices_ommivoice_kids_finetuned"
)
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SAMPLE_SEED = 42
NORMALIZATION_PROFILE = "safe"
EVAL_SCHEMA_VERSION = getattr(_normalization, "EVAL_SCHEMA_VERSION", 4)
CER_METRIC = getattr(_normalization, "CER_METRIC", "deterministic_char_cer")
CER_SCORE_VERSION = getattr(_normalization, "CER_SCORE_VERSION", 4)
NORMALIZATION_VERSION = getattr(_normalization, "NORMALIZATION_VERSION", 4)
ASR_DECODE_CONFIG = _contract.ASR_DECODE_CONFIG


def normalize_reference(
    text: str,
    profile: str = NORMALIZATION_PROFILE,
    *,
    language: str | None = None,
    lang_type: str | None = None,
) -> str:
    function = getattr(_normalization, "normalize_reference", None)
    return (
        function(
            text, profile=profile, language=language, lang_type=lang_type
        )
        if function
        else _normalization.normalize_text(text, profile)
    )


def normalize_hypothesis(text: str, profile: str = NORMALIZATION_PROFILE) -> str:
    function = getattr(_normalization, "normalize_hypothesis", None)
    return function(text, profile=profile) if function else _normalization.normalize_text(text, profile)


def reference_normalization_context(
    language: str | None = None,
    lang_type: str | None = None,
    *,
    text: str | None = None,
) -> dict[str, str]:
    return _normalization.reference_normalization_context(
        language, lang_type, text=text
    )


def reference_normalization_input_fingerprint(
    text: str, *, language: str | None = None, lang_type: str | None = None
) -> str:
    return _normalization.reference_normalization_input_fingerprint(
        text, language=language, lang_type=lang_type
    )


def normalization_fingerprint(profile: str = NORMALIZATION_PROFILE) -> str:
    function = getattr(_normalization, "normalization_fingerprint", None)
    if function:
        return function(profile=profile)
    payload = {
        "profile": profile,
        "version": NORMALIZATION_VERSION,
        "source_sha256": hashlib.sha256(
            Path(_normalization.__file__).read_bytes()
        ).hexdigest(),
    }
    return _json_fingerprint(payload)


def _json_fingerprint(value) -> str:
    return _contract.json_fingerprint(value)


def gen_text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def asr_decode_fingerprint() -> str:
    return _contract.asr_decode_fingerprint()


def eval_paths(sample_size: int) -> dict[str, Path]:
    return {
        "sample_list": EVAL_DIR / f"eval_sample_{sample_size}.json",
        "asr_cache": OUT_DIR / f"eval_asr_cache_{sample_size}.json",
        "asr_cache_meta": OUT_DIR / f"eval_asr_cache_{sample_size}.meta.json",
        "summary": OUT_DIR / f"eval_summary_{sample_size}.json",
        "details": OUT_DIR / f"eval_details_{sample_size}.txt",
    }


def eval_paths_full(out_dir: Path) -> dict[str, Path]:
    """Canonical paths used by the full-dataset evaluator."""
    return {
        "asr_cache": out_dir / "eval_asr_cache.json",
        "asr_cache_meta": out_dir / "eval_asr_cache.meta.json",
        "summary": out_dir / "eval_summary.json",
        "summary_progress": out_dir / "eval_summary_progress.json",
        "details_jsonl": out_dir / "eval_cer_details.jsonl",
        "details": out_dir / "eval_details.txt",
    }


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def file_signature(path: Path) -> dict:
    return _contract.file_signature(path)


def asr_model_signature() -> dict:
    return _contract.asr_model_signature()


def asr_model_fingerprint() -> str:
    return _contract.asr_model_fingerprint()


def load_valid_asr_cache(
    cache_path: Path,
    meta_path: Path,
    pairs,
) -> tuple[dict, dict]:
    """Load hypotheses bound to the current WAV, ASR checkpoint and decode config."""
    if not cache_path.exists() or not meta_path.exists():
        return {}, {}
    try:
        results = load_json(cache_path)
        metadata = load_json(meta_path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}, {}
    if (
        metadata.get("schema_version") != 2
        or metadata.get("asr_model") != QWEN3_ASR_LOCAL
        or metadata.get("asr_model_signature") != asr_model_signature()
        or metadata.get("asr_decode_fingerprint") != asr_decode_fingerprint()
        or not isinstance(metadata.get("entries"), dict)
    ):
        return {}, {}
    signatures = metadata["entries"]
    valid_results = {}
    valid_signatures = {}
    for wav_path, _ in pairs:
        key = str(wav_path)
        try:
            current = file_signature(wav_path)
        except OSError:
            continue
        if signatures.get(key) == current and isinstance(results.get(key), str) and results[key]:
            valid_results[key] = results[key]
            valid_signatures[key] = current
    return valid_results, valid_signatures


def write_asr_cache(
    cache_path: Path,
    meta_path: Path,
    results: dict,
    signatures: dict,
) -> None:
    write_json_atomic(cache_path, results)
    write_json_atomic(
        meta_path,
        {
            "schema_version": 2,
            "asr_model": QWEN3_ASR_LOCAL,
            "asr_model_signature": asr_model_signature(),
            "asr_model_fingerprint": asr_model_fingerprint(),
            "asr_decode_config": ASR_DECODE_CONFIG,
            "asr_decode_fingerprint": asr_decode_fingerprint(),
            "entries": signatures,
        },
    )


def validate_asr_cache(pairs, asr_results: dict, require_complete: bool = False) -> int:
    wav_keys = [str(wav) for wav, _ in pairs]
    hit = sum(
        1
        for wav in wav_keys
        if isinstance(asr_results.get(wav), str) and asr_results[wav]
    )
    miss = len(wav_keys) - hit
    if miss:
        print(
            f"WARNING: ASR miss/empty for {miss}/{len(wav_keys)} files "
            f"(cache entries: {len(asr_results)})"
        )
        if require_complete or hit == 0:
            raise RuntimeError(
                "ASR cache is incomplete or stale; re-run without --skip-asr"
            )
    return miss


def extract_speech(wav_path: Path, target_sr: int = 16000):
    wav, sample_rate = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != target_sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), orig_freq=sample_rate, new_freq=target_sr
        ).numpy()
    return wav


def transcribe_asr_batch(
    asr,
    batch_pairs: list,
    results: dict,
    signatures: dict | None = None,
) -> None:
    audio_inputs = []
    valid_paths = []
    for wav_path, _ in batch_pairs:
        try:
            audio_inputs.append((extract_speech(wav_path), 16000))
            valid_paths.append(wav_path)
        except Exception as exc:
            print(f"Error loading {wav_path}: {exc}", flush=True)
            results[str(wav_path)] = ""
    if not audio_inputs:
        return
    try:
        hypotheses = asr.transcribe(
            audio=audio_inputs,
            language=ASR_DECODE_CONFIG["language"],
            return_time_stamps=ASR_DECODE_CONFIG["return_time_stamps"],
        )
        if len(hypotheses) != len(valid_paths):
            raise RuntimeError(
                f"ASR returned {len(hypotheses)} results for {len(valid_paths)} inputs"
            )
        for wav_path, hypothesis in zip(valid_paths, hypotheses):
            text = getattr(hypothesis, "text", None)
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"ASR returned an empty hypothesis for {wav_path}")
            results[str(wav_path)] = text
            if signatures is not None:
                signatures[str(wav_path)] = file_signature(wav_path)
    except Exception:
        for wav_path in valid_paths:
            results[str(wav_path)] = ""
        raise


def load_asr_model(batch_size: int = 16, gpu_id: int = 0):
    print("Loading Qwen3-ASR model...", flush=True)
    sys.path.insert(0, "/root/code/github_repos/Qwen3-ASR")
    from qwen_asr import Qwen3ASRModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    asr = Qwen3ASRModel.from_pretrained(
        QWEN3_ASR_LOCAL,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=ASR_DECODE_CONFIG["max_new_tokens"],
    )
    print(f"Model loaded (ASR batch_size={batch_size}, gpu={gpu_id}).\n", flush=True)
    return asr


def run_asr(
    sampled,
    use_cache: bool,
    cache_path: Path,
    cache_meta_path: Path,
    on_batch_done=None,
    batch_size: int = 16,
) -> dict:
    results, signatures = load_valid_asr_cache(cache_path, cache_meta_path, sampled)
    pending = [pair for pair in sampled if not results.get(str(pair[0]))]
    if use_cache and not pending:
        print(f"ASR cache covers all {len(sampled)} items.", flush=True)
        return results
    if not use_cache:
        results, signatures, pending = {}, {}, list(sampled)
    if not pending:
        return results
    asr = load_asr_model(batch_size)
    try:
        for offset in tqdm(range(0, len(pending), batch_size), desc="ASR"):
            batch = pending[offset : offset + batch_size]
            transcribe_asr_batch(asr, batch, results, signatures)
            write_asr_cache(cache_path, cache_meta_path, results, signatures)
            if on_batch_done is not None:
                on_batch_done(batch, results)
    finally:
        del asr
        torch.cuda.empty_cache()
    return results


def calc_cer(ref: str, hyp: str):
    """Calculate CER and fail closed on invalid normalized reference text."""
    from jiwer import process_characters

    if not isinstance(ref, str) or not isinstance(hyp, str):
        raise TypeError("CER inputs must be strings")
    if not ref:
        raise ValueError("normalized reference must not be empty")
    measures = process_characters(ref, hyp)
    return (
        measures.cer,
        measures.substitutions,
        measures.insertions,
        measures.deletions,
        len(ref),
    )


def build_eval_item(wav_path: Path, json_path: Path, hypo_raw: str) -> dict:
    """Build one deterministic v4 item from clone metadata and an ASR hypothesis."""
    metadata = load_json(json_path)
    truth_raw = metadata.get("gen_text")
    if not isinstance(truth_raw, str) or not truth_raw:
        raise ValueError(f"Missing non-empty gen_text in {json_path}")
    if not isinstance(hypo_raw, str) or not hypo_raw:
        raise ValueError(f"Missing non-empty ASR hypothesis for {wav_path}")

    language = metadata.get("language")
    lang_type = metadata.get("lang_type") or metadata.get("lang_key")
    context = reference_normalization_context(
        language, lang_type, text=truth_raw
    )
    ref_normalized = normalize_reference(
        truth_raw,
        profile=NORMALIZATION_PROFILE,
        language=language,
        lang_type=lang_type,
    )
    hypo_normalized = normalize_hypothesis(hypo_raw, profile=NORMALIZATION_PROFILE)
    cer, substitutions, insertions, deletions, chars = calc_cer(
        ref_normalized, hypo_normalized
    )
    return {
        "wav": str(wav_path),
        "json": str(json_path),
        "name": wav_path.name,
        "ref_start": truth_raw,
        "hypo_start": hypo_raw,
        "ref_normalized": ref_normalized,
        "hypo_normalized": hypo_normalized,
        "reference_normalization_context": context,
        "reference_normalization_input_fingerprint": (
            reference_normalization_input_fingerprint(
                truth_raw, language=language, lang_type=lang_type
            )
        ),
        "hypothesis_normalization_input_fingerprint": gen_text_fingerprint(hypo_raw),
        "cer": cer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "chars": chars,
        "ref_audio": metadata.get("ref_audio"),
    }


def load_fixed_sample(
    sample_size: int,
    sample_list_path: Path,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[tuple[Path, Path]]:
    if sample_list_path.exists():
        paths = load_json(sample_list_path).get("wav_paths", [])
    else:
        candidates = [
            str(path)
            for path in sorted(OUT_DIR.rglob("text_*.wav"))
            if path.with_suffix(".json").exists()
        ]
        rng = random.Random(seed)
        paths = sorted(rng.sample(candidates, min(sample_size, len(candidates))))
        write_json_atomic(
            sample_list_path,
            {
                "seed": seed,
                "sample_size": len(paths),
                "created_at": datetime.now().isoformat(),
                "wav_paths": paths,
            },
        )
    pairs = []
    for wav_value in paths:
        wav_path = Path(wav_value)
        json_path = wav_path.with_suffix(".json")
        if wav_path.exists() and json_path.exists():
            pairs.append((wav_path, json_path))
        else:
            print(f"Warning: missing {wav_path}")
    return pairs


def summarize_cer(details: list[dict]) -> dict:
    total_sub = sum(item["substitutions"] for item in details)
    total_ins = sum(item["insertions"] for item in details)
    total_del = sum(item["deletions"] for item in details)
    total_chars = sum(item["chars"] for item in details)
    cers = [item["cer"] for item in details]
    return {
        "weighted_cer": (
            (total_sub + total_ins + total_del) / total_chars * 100
            if total_chars
            else 0.0
        ),
        "avg_cer": float(np.mean(cers) * 100) if cers else 0.0,
        "median_cer": float(np.median(cers) * 100) if cers else 0.0,
        "min_cer": float(min(cers) * 100) if cers else 0.0,
        "max_cer": float(max(cers) * 100) if cers else 0.0,
        "total_chars": total_chars,
        "total_ins": total_ins,
        "total_del": total_del,
        "total_sub": total_sub,
    }


def write_details(
    path: Path,
    title: str,
    summary: dict,
    details: list[dict],
    evaluated_at: str,
) -> None:
    lines = [
        title,
        f"Evaluated at: {evaluated_at}",
        f"Metric: {CER_METRIC} v{CER_SCORE_VERSION}",
        f"Normalization: {NORMALIZATION_PROFILE} v{NORMALIZATION_VERSION}",
        "",
        f"Weighted CER: {summary['weighted_cer']:.2f}%",
        f"Average CER:  {summary['avg_cer']:.2f}%",
        f"Median CER:   {summary['median_cer']:.2f}%",
        f"Char Errors:  {summary['total_ins']} ins, {summary['total_del']} del, "
        f"{summary['total_sub']} sub / {summary['total_chars']} chars",
        "",
    ]
    for rank, item in enumerate(
        sorted(details, key=lambda value: value["cer"], reverse=True), start=1
    ):
        lines.extend(
            [
                "=" * 100,
                f"Rank {rank}/{len(details)} | CER: {item['cer'] * 100:.2f}% | "
                f"Sub: {item['substitutions']} Ins: {item['insertions']} "
                f"Del: {item['deletions']} | Chars: {item['chars']} | {item['name']}",
                f"WAV: {item['wav']}",
                "Reference (raw):",
                item["ref_start"],
                "Hypothesis (raw):",
                item["hypo_start"],
                "Reference (normalized):",
                item["ref_normalized"],
                "Hypothesis (normalized):",
                item["hypo_normalized"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-asr", action="store_true", help="Require cached ASR")
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--refresh-asr-cache", action="store_true")
    parser.add_argument(
        "--refresh-cer",
        action="store_true",
        help="Recompute deterministic normalization/CER from current ASR text",
    )
    args = parser.parse_args()
    if args.asr_batch_size <= 0:
        parser.error("--asr-batch-size must be positive")
    if args.skip_asr and args.refresh_asr_cache:
        parser.error("--skip-asr and --refresh-asr-cache are mutually exclusive")

    sample_size = args.sample_size
    seed = args.seed if args.seed is not None else (
        42 if sample_size == 200 else 43 if sample_size == 500 else sample_size
    )
    paths = eval_paths(sample_size)
    sampled = load_fixed_sample(sample_size, paths["sample_list"], seed=seed)
    if not sampled:
        raise RuntimeError("No valid samples found")

    if args.skip_asr:
        asr_results, _ = load_valid_asr_cache(
            paths["asr_cache"], paths["asr_cache_meta"], sampled
        )
    else:
        asr_results = run_asr(
            sampled,
            use_cache=not args.refresh_asr_cache,
            cache_path=paths["asr_cache"],
            cache_meta_path=paths["asr_cache_meta"],
            batch_size=args.asr_batch_size,
        )
    validate_asr_cache(sampled, asr_results, require_complete=True)

    details = [
        build_eval_item(wav_path, json_path, asr_results[str(wav_path)])
        for wav_path, json_path in sampled
    ]
    summary = summarize_cer(details)
    evaluated_at = datetime.now().isoformat()
    output = {
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "cer_metric": CER_METRIC,
        "cer_score_version": CER_SCORE_VERSION,
        "normalization_profile": NORMALIZATION_PROFILE,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_fingerprint": normalization_fingerprint(
            profile=NORMALIZATION_PROFILE
        ),
        "asr_model": QWEN3_ASR_LOCAL,
        "asr_model_fingerprint": asr_model_fingerprint(),
        "asr_decode_fingerprint": asr_decode_fingerprint(),
        "sample_list": str(paths["sample_list"]),
        "sample_size": len(details),
        "seed": seed,
        "cer": summary,
        "evaluated_at": evaluated_at,
        "details": sorted(details, key=lambda item: item["cer"], reverse=True),
    }
    write_json_atomic(paths["summary"], output)
    write_details(
        paths["details"],
        f"Batch deterministic CER ({len(details)} fixed samples)",
        summary,
        details,
        evaluated_at,
    )
    print(f"Weighted CER: {summary['weighted_cer']:.2f}%")
    print(f"Summary saved to: {paths['summary']}")
    print(f"Details saved to: {paths['details']}")


if __name__ == "__main__":
    main()
