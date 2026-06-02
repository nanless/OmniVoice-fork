#!/usr/bin/env python3
"""Two-round fast MOS evaluation — serial GPU models to avoid OOM.

Round 1a: UTMOS22Strong on GPU 0 + GPU 1
Round 1b: UTMOSv2   on GPU 0 + GPU 1
Round 2:  SCOREQ + TTSDS2 on CPU (multiprocess)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
from scorers import create_scorer

OUT_DIR = Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned")
MODEL_DIR = Path("/root/code/github_repos/OmniVoice-fork/TTS_eval_models")
LOG_PATH = OUT_DIR / "logs" / "eval_fast.log"

UTMOS_BS = 32
UTMOSV2_BS = 8

# ---------------------------------------------------------------------------
# Tee logger: print to stdout AND append to file, always flushed
# ---------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._stdout.flush()
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

sys.stdout = TeeLogger(LOG_PATH)
sys.stderr = sys.stdout  # errors also go to the same tee


def fast_scan(out_dir: Path) -> List[Path]:
    t0 = time.time()
    result = subprocess.run(
        ["find", str(out_dir), "-type", "f", "-name", "text_*.wav"],
        capture_output=True, text=True, check=True
    )
    wav_paths = [Path(p) for p in result.stdout.strip().split("\n") if p]
    print(f"[scan] {len(wav_paths)} wavs in {time.time()-t0:.1f}s", flush=True)
    return wav_paths


def write_mos(wav_path: Path, data: dict):
    json_path = wav_path.with_suffix(".json")
    mos_path = wav_path.with_suffix(".mos.json")

    record = {}
    if mos_path.exists():
        try:
            with open(mos_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            pass

    record.update(data)
    record["cloned_audio"] = str(wav_path)
    record["sidecar_json"] = str(json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("status") == "generated":
            record.update(
                dataset=json_path.relative_to(OUT_DIR).parts[0],
                ref_audio=meta.get("ref_audio"),
                gen_text=meta.get("gen_text"),
                language=meta.get("language"),
                speed=meta.get("speed"),
            )
    except Exception:
        pass

    with open(mos_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_utmos22strong(items: List[Path], device: str):
    print(f"[{device}] Loading UTMOS22Strong ...", flush=True)
    scorer = create_scorer("UTMOS22Strong", device=device, model_dir=MODEL_DIR)
    total = len(items)
    print(f"[{device}] Evaluating {total} files ...", flush=True)

    t0 = time.time()
    for i in range(0, total, UTMOS_BS):
        batch = items[i:i + UTMOS_BS]
        try:
            scores = scorer.score_files(batch)
            for wav, score in zip(batch, scores):
                write_mos(wav, {"utmos22strong": score})
        except Exception as e:
            print(f"[{device}] batch error: {e}", flush=True)

        if (i // UTMOS_BS) % 20 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            print(f"[{device}] {i}/{total} {rate:.1f} files/s ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"[{device}] Done {total} files in {elapsed:.1f}s ({total/elapsed:.1f} files/s)", flush=True)


def run_utmosv2(items: List[Path], device: str):
    print(f"[{device}] Loading UTMOSv2 ...", flush=True)
    scorer = create_scorer("UTMOSv2", device=device)
    total = len(items)
    print(f"[{device}] Evaluating {total} files ...", flush=True)

    t0 = time.time()
    for i in range(0, total, UTMOSV2_BS):
        batch = items[i:i + UTMOSV2_BS]
        try:
            scores = scorer.score_files(batch)
            for wav, score in zip(batch, scores):
                write_mos(wav, {"utmosv2": score})
        except Exception as e:
            print(f"[{device}] batch error: {e}", flush=True)

        if (i // UTMOSV2_BS) % 20 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            print(f"[{device}] {i}/{total} {rate:.1f} files/s ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"[{device}] Done {total} files in {elapsed:.1f}s ({total/elapsed:.1f} files/s)", flush=True)


def run_scoreq_ttsds(items: List[Path]):
    print(f"[cpu] Loading SCOREQ + TTSDS2 ...", flush=True)
    scoreq = create_scorer("SCOREQ", device="cpu")
    ttsds = create_scorer("TTSDS2", device="cpu")
    total = len(items)
    print(f"[cpu] Evaluating {total} files ...", flush=True)

    t0 = time.time()
    for i, wav in enumerate(items):
        try:
            sq = scoreq.score_file(wav)
            ts = ttsds.score_file(wav)
            write_mos(wav, {"scoreq": sq, "ttsds2": ts})
        except Exception as e:
            print(f"[cpu] Error on {wav.name}: {e}", flush=True)

        if i > 0 and i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            print(f"[cpu] {i}/{total} {rate:.1f} files/s ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"[cpu] Done {total} files in {elapsed:.1f}s ({total/elapsed:.1f} files/s)", flush=True)


def run_gpu_round(items0: List[Path], items1: List[Path], fn, label: str):
    print(f"\n=== {label} ===", flush=True)
    p0 = mp.Process(target=fn, args=(items0, "cuda:0"))
    p1 = mp.Process(target=fn, args=(items1, "cuda:1"))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print(f"=== {label} done ===\n", flush=True)


def main():
    wavs = fast_scan(OUT_DIR)
    if not wavs:
        print("No wavs found.")
        return

    import random
    random.seed(42)
    random.shuffle(wavs)

    n = len(wavs)
    mid = n // 2
    gpu0_items = wavs[:mid]
    gpu1_items = wavs[mid:]

    mp.set_start_method("spawn", force=True)

    # Round 1a: UTMOS22Strong (GPU, batch)
    run_gpu_round(gpu0_items, gpu1_items, run_utmos22strong, "Round 1a: UTMOS22Strong")

    # Round 1b: UTMOSv2 (GPU, batch)
    run_gpu_round(gpu0_items, gpu1_items, run_utmosv2, "Round 1b: UTMOSv2")

    # Round 2: SCOREQ + TTSDS2 (CPU, multiprocess)
    print(f"\n=== Round 2: SCOREQ + TTSDS2 on CPU ===", flush=True)
    cpu_workers = min(32, os.cpu_count() or 4)
    shard_size = (n + cpu_workers - 1) // cpu_workers
    cpu_procs = []
    for i in range(cpu_workers):
        start = i * shard_size
        end = min(start + shard_size, n)
        if start >= end:
            continue
        p = mp.Process(target=run_scoreq_ttsds, args=(wavs[start:end],))
        cpu_procs.append(p)
        p.start()

    for p in cpu_procs:
        p.join()

    print(f"\n=== All rounds done ===", flush=True)


if __name__ == "__main__":
    main()
