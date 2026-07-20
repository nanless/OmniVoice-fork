#!/usr/bin/env python3
"""Delete bad cloned audios and copy good ones to target speaker directories.

Rules (default raw cosine threshold: 0.80):
  DELETE: CER >= 0.05  AND NOT (CER <= 0.1 AND raw cosine SIM > threshold)
  COPY:   CER <= 0.1   AND raw cosine SIM > threshold
  KEEP:   everything else (stay in source, no action)

Naming:
  {ref_audio_stem}_clone_text_NNN.wav
  {ref_audio_stem}_clone_text_NNN.json
  {ref_audio_stem}_clone_text_NNN.cer.json   (was .eval.json)
  {ref_audio_stem}_clone_text_NNN.sim.json
  {ref_audio_stem}_clone_text_NNN.mos.json

Usage:
    python prune_and_copy.py                     # preview only
    python prune_and_copy.py --execute           # delete/copy after review
    python prune_and_copy.py --execute --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_OUT_DIR = Path(
    "/root/group-shared/voiceprint/data/speech/voice_activity_detection"
    "/batch_cloned_voices_ommivoice_kids_finetuned"
)

DEFAULT_TARGET_ROOT = Path(
    "/root/group-shared/voiceprint/data/speech/speaker_diarization"
    "/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
    "/audio_omnivoice_clone_sim0.8_filtered"
)

DEFAULT_SIM_THRESHOLD = 0.80

SIDECAR_SUFFIXES = [".json", ".eval.json", ".sim.json", ".mos.json"]
SIDECAR_RENAME = {".eval.json": ".cer.json"}

EVAL_SIM_DIR = Path(__file__).resolve().parent / "eval_sim"
sys.path.insert(0, str(EVAL_SIM_DIR))
from metric_contract import (  # noqa: E402
    SimilarityCollectionValidator,
    validate_current_audio_files,
    validate_current_model_files,
)
EVAL_CER_DIR = Path(__file__).resolve().parent / "eval_cer"
sys.path.insert(0, str(EVAL_CER_DIR))
from cer_normalization import (  # noqa: E402
    CER_SCORE_VERSION,
    EVAL_SCHEMA_VERSION,
    NORMALIZATION_VERSION,
    SAFE_PROFILE,
    normalization_fingerprint,
    reference_normalization_input_fingerprint,
)
from eval_contract import (  # noqa: E402
    asr_decode_fingerprint,
    asr_model_fingerprint,
)
from eval_common import iter_clone_records  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_cer_data(
    cer_path: Path, inventory_records: Dict[str, dict]
) -> Dict[str, float]:
    """Load {wav_path: deterministic CER v4} from eval_cer_details.jsonl."""
    print("[load] CER data…", file=sys.stderr, flush=True)
    data = {}
    with open(cer_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{cer_path}:{line_no}: invalid JSON: {exc}") from exc
            wav = r.get("wav", "")
            if (
                r.get("eval_schema_version") != EVAL_SCHEMA_VERSION
                or r.get("cer_metric") != "deterministic_char_cer"
                or r.get("cer_score_version") != CER_SCORE_VERSION
                or r.get("normalization_profile") != SAFE_PROFILE
                or r.get("normalization_version") != NORMALIZATION_VERSION
                or r.get("normalization_fingerprint")
                != normalization_fingerprint(SAFE_PROFILE)
                or r.get("asr_model_fingerprint") != asr_model_fingerprint()
                or r.get("asr_decode_fingerprint") != asr_decode_fingerprint()
                or r.get("stage") != "complete"
            ):
                raise ValueError(f"{cer_path}:{line_no}: non-v4 CER row for {wav!r}")
            cer = r.get("cer")
            if not isinstance(wav, str) or not wav:
                raise ValueError(f"{cer_path}:{line_no}: missing wav path")
            if (
                isinstance(cer, bool)
                or not isinstance(cer, (int, float))
                or not math.isfinite(float(cer))
                or cer < 0
            ):
                raise ValueError(f"{cer_path}:{line_no}: invalid cer for {wav!r}: {cer!r}")
            try:
                stat = Path(wav).stat()
            except OSError as exc:
                raise ValueError(
                    f"{cer_path}:{line_no}: cannot stat current WAV {wav!r}: {exc}"
                ) from exc
            if r.get("cloned_audio_signature") != {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }:
                raise ValueError(
                    f"{cer_path}:{line_no}: stale CER signature for {wav!r}"
                )
            clone = inventory_records.get(wav)
            if clone is not None:
                text = clone.get("gen_text")
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"{cer_path}:{line_no}: current clone metadata has no gen_text"
                    )
                expected_input = reference_normalization_input_fingerprint(
                    text,
                    language=clone.get("language"),
                    lang_type=clone.get("lang_type") or clone.get("lang_key"),
                )
                if r.get("reference_normalization_input_fingerprint") != expected_input:
                    raise ValueError(
                        f"{cer_path}:{line_no}: stale reference normalization input for {wav!r}"
                    )
            value = float(cer)
            if wav in data and not math.isclose(data[wav], value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{cer_path}:{line_no}: conflicting duplicate CER for {wav!r}: "
                    f"{value!r} vs {data[wav]!r}"
                )
            data[wav] = value
    print(f"[load] CER: {len(data):,} records", file=sys.stderr)
    return data


def load_sim_data(sim_dir: Path) -> Dict[str, dict]:
    """Load {cloned_audio: {similarity, ref_audio, dataset, ...}} from all sim jsonl files."""
    print("[load] SIM data…", file=sys.stderr, flush=True)
    validator = SimilarityCollectionValidator()
    files = sorted(sim_dir.glob("eval_sim_details*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No eval_sim_details*.jsonl found in {sim_dir}")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{fp}:{line_no}: invalid JSON: {exc}") from exc
                source = f"{fp}:{line_no}"
                validator.add(r, source)
                validate_current_audio_files(r, source)
    data = dict(validator.records)
    if not data:
        raise ValueError(f"No valid raw-cosine v2 records found in {sim_dir}")
    failed = [wav for wav, record in data.items() if record.get("similarity") is None]
    if failed:
        preview = "\n  ".join(failed[:5])
        raise ValueError(
            f"Refusing destructive classification: {len(failed)} similarity records "
            f"have null scores. Re-run eval_sim successfully.\n  {preview}"
        )
    print(f"[load] SIM: {len(data):,} records from {len(files)} files", file=sys.stderr)
    return data


# ────────────────────────────────────────────────────────────────────
# Mapping logic
# ────────────────────────────────────────────────────────────────────

def parse_ref_audio(ref_path: str, cloned_path: str = "") -> Optional[Tuple[str, str, str]]:
    """Parse ref_audio path → (target_dataset, target_speaker, ref_stem).
    
    Uses cloned_path to distinguish King-ASR-612 vs King-ASR-725 (not in ref_audio).
    """
    if not ref_path:
        return None
    parts = ref_path.split("/")
    norm = ref_path.lower()

    # Canonical merged-dataset reference layout:
    #   .../audio/<dataset>/<speaker>/<wav>
    canonical_datasets = {
        "childmandarin", "chineseenglishchildren", "king-asr-725",
        "kingasr612", "speechocean762",
    }
    for index, part in enumerate(parts[:-2]):
        if part == "audio" and parts[index + 1] in canonical_datasets:
            return (parts[index + 1], parts[index + 2], Path(ref_path).stem)

    # Chinese_English_Children: .../WAV/G0001/G0001_X_SXXXX.wav
    if "chinese_english" in norm or "children_integrated" in norm:
        spk = parts[-2]
        stem = Path(ref_path).stem
        return ("chineseenglishchildren", f"chineseenglishchildren_{spk}", stem)

    # King-ASR-EN-Kid: determine 612 vs 725 from cloned_audio path
    if "king-asr-en-kid" in norm:
        is_612 = "King-ASR-612" in cloned_path
        for p in parts:
            if p.upper().startswith("SPEAKER") and len(p) > 7:
                num = p[7:]
                if num.isdigit():
                    if is_612:
                        return ("kingasr612", f"kingasr612_{num}", Path(ref_path).stem)
                    else:
                        return ("king-asr-725", f"king-asr-725_{p}", Path(ref_path).stem)
        # Fallback
        for p in parts:
            if p.upper().startswith("SPEAKER") and len(p) > 7:
                num = p[7:]
                return ("kingasr612", f"kingasr612_{num}", Path(ref_path).stem) if is_612 else \
                       ("king-asr-725", f"king-asr-725_{p}", Path(ref_path).stem)

    # speechocean762: .../WAVE/SPEAKER3049/030490062.WAV
    if "speechocean762" in norm:
        for p in parts:
            if p.upper().startswith("SPEAKER") and p[7:].isdigit():
                num = p[7:]
                return ("speechocean762", f"speechocean762_{num}", Path(ref_path).stem)

    # BAAI-ChildMandarin: .../dev/014/014_4_M_L_WUHAN_iPhone_036.wav
    if "baai" in norm or "childmandarin" in norm:
        group = parts[-2]
        try:
            gid = int(group)
        except ValueError:
            return None
        return ("childmandarin", f"childmandarin_{gid:03d}", Path(ref_path).stem)

    return None


# ────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────

def classify(
    cer: Optional[float],
    sim: Optional[float],
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> str:
    """Return: 'COPY' | 'DELETE' | 'KEEP'."""
    if cer is None or sim is None:
        return "KEEP"

    sim_pass = sim > sim_threshold
    cer_lt_005 = cer < 0.05
    cer_le_01 = cer <= 0.1

    # COPY: acceptable content and voice similarity above the configured threshold.
    if cer_le_01 and sim_pass:
        return "COPY"

    # DELETE: CER >= 0.05 unless already accepted by the COPY rule above.
    if not cer_lt_005:
        return "DELETE"

    return "KEEP"


# ────────────────────────────────────────────────────────────────────
# Actions
# ────────────────────────────────────────────────────────────────────

def delete_files(wav_path: str, dry: bool = False) -> Tuple[int, int]:
    """Delete a wav and all its sidecar JSONs. Returns (deleted, failed)."""
    wav = Path(wav_path)
    del_files = [wav]
    for suffix in SIDECAR_SUFFIXES:
        sf = wav.with_suffix(suffix)
        if sf.is_file():
            del_files.append(sf)

    ok, fail = 0, 0
    for f in del_files:
        try:
            if not dry:
                os.remove(str(f))
            ok += 1
        except OSError as e:
            print(f"  [DELETE FAIL] {f}: {e}", file=sys.stderr)
            fail += 1
    return ok, fail


_COPY_SIDE_FILE = None  # shared by pool workers via initializer


def _copy_init(target_root: str, dry: bool):
    global _COPY_SIDE_FILE
    _COPY_SIDE_FILE = {"target_root": target_root, "dry": dry}


def _copy_one(args: Tuple[str, dict]) -> Tuple[int, int, Optional[str]]:
    """Copy one cloned wav + all sidecars to target dir. Returns (ok, fail, error_msg)."""
    wav_str, ref_info = args
    target_root = _COPY_SIDE_FILE["target_root"]
    dry = _COPY_SIDE_FILE["dry"]

    wav = Path(wav_str)
    target_ds = ref_info["target_ds"]
    target_spk = ref_info["target_spk"]
    ref_stem = ref_info["ref_stem"]

    # A top-up round may restore a speaker that had no previously accepted clone.
    spk_dir = Path(target_root) / target_ds / target_spk
    if not dry:
        spk_dir.mkdir(parents=True, exist_ok=True)

    # Determine cloned filename parts
    cloned_stem = wav.stem  # e.g., text_003
    task_id = None
    clone_meta_path = wav.with_suffix(".json")
    if clone_meta_path.is_file():
        try:
            task_id = json.loads(clone_meta_path.read_text(encoding="utf-8")).get("task_id")
        except (json.JSONDecodeError, OSError):
            task_id = None
    unique_suffix = str(task_id)[:16] if task_id else cloned_stem
    new_prefix = f"{ref_stem}_clone_{unique_suffix}"

    ok, fail = 0, 0
    # Copy wav
    dst_wav = spk_dir / f"{new_prefix}.wav"
    try:
        if not dry:
            if dst_wav.exists() and dst_wav.read_bytes() != wav.read_bytes():
                return 0, 1, f"refusing to overwrite different audio: {dst_wav}"
            shutil.copy2(str(wav), str(dst_wav))
        ok += 1
    except OSError as e:
        fail += 1
        print(f"  [COPY FAIL] {wav_str}: {e}", file=sys.stderr)

    # Copy sidecar files with rename
    for suffix in SIDECAR_SUFFIXES:
        src = wav.with_suffix(suffix)
        if not src.is_file():
            continue
        new_suffix = SIDECAR_RENAME.get(suffix, suffix)
        dst = spk_dir / f"{new_prefix}{new_suffix}"
        try:
            if not dry:
                shutil.copy2(str(src), str(dst))
            ok += 1
        except OSError as e:
            fail += 1
            print(f"  [COPY FAIL] {src}: {e}", file=sys.stderr)

    return ok, fail, None


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete/copy files; default is a read-only preview",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated explicit preview flag; preview is already the default",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel copy workers")
    parser.add_argument("--skip-delete", action="store_true", help="Skip deletion, only copy")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copy, only delete")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help=f"Raw cosine similarity threshold (exclusive; default: {DEFAULT_SIM_THRESHOLD})",
    )
    parser.add_argument("--log", type=Path, default=None, help="Write operation log")
    args = parser.parse_args()

    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    args.dry_run = not args.execute

    if not math.isfinite(args.min_sim) or not -1.0 <= args.min_sim <= 1.0:
        parser.error("--min-sim must be finite and in [-1, 1]")

    t0 = time.time()

    # Load data
    cer_path = args.out_dir / "eval_cer_details.jsonl"
    if not cer_path.exists():
        print(f"ERROR: {cer_path} not found", file=sys.stderr)
        sys.exit(1)

    inventory_records = {
        str(wav): record
        for wav, _, record in iter_clone_records(
            args.out_dir.resolve(), allow_partial=False
        )
    }
    inventory = set(inventory_records)
    cer_data = load_cer_data(cer_path, inventory_records)
    extra_cer = set(cer_data) - inventory
    missing_cer = inventory - set(cer_data)
    if extra_cer or missing_cer:
        raise RuntimeError(
            "Refusing destructive classification: canonical CER coverage does not "
            f"match clone inventory (missing={len(missing_cer)}, extra={len(extra_cer)})"
        )
    sim_data = load_sim_data(args.out_dir)
    first_sim = next(iter(sim_data.values()))
    validate_current_model_files(first_sim, "similarity collection")

    resolved_root = args.out_dir.resolve()
    outside = []
    missing_audio = []
    for wav in cer_data:
        resolved_wav = Path(wav).resolve()
        try:
            resolved_wav.relative_to(resolved_root)
        except ValueError:
            outside.append(wav)
        if not resolved_wav.is_file():
            missing_audio.append(wav)
    if outside:
        preview = "\n  ".join(outside[:5])
        raise RuntimeError(
            f"Refusing destructive classification: {len(outside)} CER paths are outside "
            f"--out-dir {resolved_root}.\n  {preview}"
        )
    if missing_audio:
        preview = "\n  ".join(missing_audio[:5])
        raise RuntimeError(
            f"Refusing destructive classification: {len(missing_audio)} CER audio files "
            f"are missing.\n  {preview}"
        )
    missing_sim = sorted(set(cer_data) - set(sim_data))
    if missing_sim:
        preview = "\n  ".join(missing_sim[:5])
        raise RuntimeError(
            f"Refusing destructive classification: {len(missing_sim)} CER records "
            f"have no raw-cosine v2 similarity record.\n  {preview}"
        )

    # Classify
    print("\n[classify] Applying rules…", file=sys.stderr, flush=True)
    copy_list: List[Tuple[str, dict]] = []  # (wav, ref_info)
    delete_list: List[str] = []
    keep_count = 0
    no_ref_count = 0
    stats = Counter()

    for wav, cer in cer_data.items():
        sim_rec = sim_data.get(wav, {})
        sim = sim_rec.get("similarity") if sim_rec else None
        action = classify(cer, sim, sim_threshold=args.min_sim)
        stats[action] += 1

        if action == "COPY":
            ref_audio = sim_rec.get("ref_audio", "")
            parsed = parse_ref_audio(ref_audio, cloned_path=wav)
            if parsed is None:
                no_ref_count += 1
                stats["COPY_failed(no_ref)"] += 1
                continue
            target_ds, target_spk, ref_stem = parsed
            copy_list.append((wav, {
                "target_ds": target_ds,
                "target_spk": target_spk,
                "ref_stem": ref_stem,
            }))
        elif action == "DELETE":
            delete_list.append(wav)
        else:
            keep_count += 1

    print(f"[classify] COPY={len(copy_list):,}  DELETE={len(delete_list):,}  KEEP={keep_count:,}", file=sys.stderr)
    if no_ref_count:
        print(f"[classify] COPY failed (no ref_audio): {no_ref_count}", file=sys.stderr)

    # ── Phase 1: Delete ──
    if not args.skip_delete:
        print(f"\n{'='*60}", file=sys.stderr)
        tag = "[DRY-RUN] " if args.dry_run else ""
        print(f"Phase 1: {tag}DELETE {len(delete_list):,} files", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        del_ok, del_fail = 0, 0
        chunk = max(1, len(delete_list) // 20)
        for i, wav in enumerate(delete_list):
            ok, fail = delete_files(wav, dry=args.dry_run)
            del_ok += ok
            del_fail += fail
            if (i + 1) % chunk == 0:
                pct = (i + 1) / len(delete_list) * 100
                print(f"  [{tag}DELETE] {i+1:,}/{len(delete_list):,} ({pct:.0f}%) ok={del_ok:,} fail={del_fail:,}",
                      file=sys.stderr, flush=True)
        print(f"  [{tag}DELETE] DONE: ok={del_ok:,} fail={del_fail:,}", file=sys.stderr)
    else:
        del_ok, del_fail = 0, 0
        print("\n[skip] Deletion skipped (--skip-delete)", file=sys.stderr)

    # ── Phase 2: Copy ──
    if not args.skip_copy:
        print(f"\n{'='*60}", file=sys.stderr)
        tag = "[DRY-RUN] " if args.dry_run else ""
        print(f"Phase 2: {tag}COPY {len(copy_list):,} files", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        copy_ok, copy_fail = 0, 0
        copy_errors: List[str] = []

        if len(copy_list) <= 100 or args.workers <= 1:
            # Sequential
            _copy_init(str(args.target_root), args.dry_run)
            for i, item in enumerate(copy_list):
                ok, fail, err = _copy_one(item)
                copy_ok += ok
                copy_fail += fail
                if err:
                    copy_errors.append(err)
                if (i + 1) % 5000 == 0:
                    pct = (i + 1) / len(copy_list) * 100
                    print(f"  [{tag}COPY] {i+1:,}/{len(copy_list):,} ({pct:.0f}%) ok={copy_ok:,} fail={copy_fail:,}",
                          file=sys.stderr, flush=True)
        else:
            # Parallel
            _copy_init(str(args.target_root), args.dry_run)
            with Pool(processes=args.workers, initializer=_copy_init,
                      initargs=(str(args.target_root), args.dry_run)) as pool:
                for i, (ok, fail, err) in enumerate(pool.imap_unordered(_copy_one, copy_list, chunksize=100)):
                    copy_ok += ok
                    copy_fail += fail
                    if err:
                        copy_errors.append(err)
                    if (i + 1) % 5000 == 0:
                        pct = (i + 1) / len(copy_list) * 100
                        print(f"  [{tag}COPY] {i+1:,}/{len(copy_list):,} ({pct:.0f}%) ok={copy_ok:,} fail={copy_fail:,}",
                              file=sys.stderr, flush=True)

        print(f"  [{tag}COPY] DONE: ok={copy_ok:,} fail={copy_fail:,}", file=sys.stderr)
        if copy_errors:
            print(f"  [{tag}COPY] {len(copy_errors)} target dir missing:", file=sys.stderr)
            for e in copy_errors[:10]:
                print(f"    {e}", file=sys.stderr)
            if len(copy_errors) > 10:
                print(f"    ... and {len(copy_errors) - 10} more", file=sys.stderr)
    else:
        copy_ok, copy_fail = 0, 0
        print("\n[skip] Copy skipped (--skip-copy)", file=sys.stderr)

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY  (elapsed {elapsed:.0f}s)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Deleted:  {del_ok:,} files removed ({del_fail} failed)", file=sys.stderr)
    print(f"  Copied:   {copy_ok:,} files copied ({copy_fail} failed)", file=sys.stderr)
    print(f"  Kept:     {keep_count:,} files unchanged", file=sys.stderr)
    if no_ref_count:
        print(f"  Skipped:  {no_ref_count} COPY candidates (no ref_audio)", file=sys.stderr)
    print(f"  Dry run:  {args.dry_run}", file=sys.stderr)
    print(f"  Raw SIM:  > {args.min_sim}", file=sys.stderr)

    if args.log:
        log_entry = {
            "dry_run": args.dry_run,
            "elapsed_s": elapsed,
            "deleted_ok": del_ok,
            "deleted_fail": del_fail,
            "copied_ok": copy_ok,
            "copied_fail": copy_fail,
            "kept": keep_count,
            "copy_skipped_no_ref": no_ref_count,
            "min_raw_cosine_similarity": args.min_sim,
            "rules": (
                f"DELETE: CER>=0.05 unless CER<=0.1 AND raw_SIM>{args.min_sim}; "
                f"COPY: CER<=0.1 AND raw_SIM>{args.min_sim}"
            ),
        }
        args.log.write_text(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
