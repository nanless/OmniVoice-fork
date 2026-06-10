#!/usr/bin/env python3
"""Delete bad cloned audios and copy good ones to target speaker directories.

Rules:
  DELETE: CER >= 0.05  AND NOT (CER <= 0.1 AND SIM > 0.85)
  COPY:   CER < 0.05  AND SIM > 0.8
  COPY:   CER <= 0.1   AND SIM > 0.85
  KEEP:   everything else (stay in source, no action)

Naming:
  {ref_audio_stem}_clone_text_NNN.wav
  {ref_audio_stem}_clone_text_NNN.json
  {ref_audio_stem}_clone_text_NNN.cer.json   (was .eval.json)
  {ref_audio_stem}_clone_text_NNN.sim.json
  {ref_audio_stem}_clone_text_NNN.mos.json

Usage:
    python prune_and_copy.py                     # execute
    python prune_and_copy.py --dry-run           # preview only
    python prune_and_copy.py --workers 8         # parallel copy
"""

from __future__ import annotations

import argparse
import json
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
    "/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio"
)

SIDECAR_SUFFIXES = [".json", ".eval.json", ".sim.json", ".mos.json"]
SIDECAR_RENAME = {".eval.json": ".cer.json"}


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_cer_data(cer_path: Path) -> Dict[str, float]:
    """Load {wav_path: manual_cer} from eval_cer_details.jsonl."""
    print("[load] CER data…", file=sys.stderr, flush=True)
    data = {}
    with open(cer_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav = r.get("wav", "")
            cer = r.get("manual_cer")
            if wav and cer is not None:
                data[wav] = cer
    print(f"[load] CER: {len(data):,} records", file=sys.stderr)
    return data


def load_sim_data(sim_dir: Path) -> Dict[str, dict]:
    """Load {cloned_audio: {similarity, ref_audio, dataset, ...}} from all sim jsonl files."""
    print("[load] SIM data…", file=sys.stderr, flush=True)
    seen = set()
    data = {}
    files = sorted(sim_dir.glob("eval_sim_details*.jsonl"))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wav = r.get("cloned_audio", "")
                if wav and wav not in seen:
                    seen.add(wav)
                    data[wav] = r
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

def classify(cer: Optional[float], sim: Optional[float]) -> str:
    """Return: 'COPY' | 'DELETE' | 'KEEP'."""
    if cer is None:
        return "KEEP"

    sim_gt_085 = sim is not None and sim > 0.85
    sim_gt_08 = sim is not None and sim > 0.8
    cer_lt_005 = cer < 0.05
    cer_le_01 = cer <= 0.1

    # COPY rules
    if cer_lt_005 and sim_gt_08:
        return "COPY"
    if cer_le_01 and sim_gt_085:
        return "COPY"

    # DELETE: CER >= 0.05, not exempted by SIM>0.85 AND CER<=0.1
    if not cer_lt_005 and not (cer_le_01 and sim_gt_085):
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

    # Check target speaker dir exists
    spk_dir = Path(target_root) / target_ds / target_spk
    if not spk_dir.is_dir():
        return 0, 0, f"target dir missing: {spk_dir}"

    # Determine cloned filename parts
    cloned_name = wav.name  # e.g., text_003.wav
    cloned_stem = wav.stem  # e.g., text_003
    new_prefix = f"{ref_stem}_clone_{cloned_stem}"

    ok, fail = 0, 0
    # Copy wav
    dst_wav = spk_dir / f"{new_prefix}.wav"
    try:
        if not dry:
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
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--workers", type=int, default=4, help="Parallel copy workers")
    parser.add_argument("--skip-delete", action="store_true", help="Skip deletion, only copy")
    parser.add_argument("--skip-copy", action="store_true", help="Skip copy, only delete")
    parser.add_argument("--log", type=Path, default=None, help="Write operation log")
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    cer_path = args.out_dir / "eval_cer_details.jsonl"
    if not cer_path.exists():
        print(f"ERROR: {cer_path} not found", file=sys.stderr)
        sys.exit(1)

    cer_data = load_cer_data(cer_path)
    sim_data = load_sim_data(args.out_dir)

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
        action = classify(cer, sim)
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
            "rules": "DELETE: CER>=0.05 NOT (CER<=0.1 AND SIM>0.85); COPY: CER<0.05 SIM>0.8 OR CER<=0.1 SIM>0.85",
        }
        args.log.write_text(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
