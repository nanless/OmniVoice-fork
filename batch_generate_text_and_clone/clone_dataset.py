#!/usr/bin/env python3
"""Clone dataset voices using 10 random texts per audio with random speed.
Supports multi-GPU multi-worker via --worker-id / --num-workers.

Usage:
    # Single worker
    python clone_dataset.py --gpu 0 --worker-id 0 --num-workers 1

    # Worker 3 of 8 (GPU 0, worker 3)
    python clone_dataset.py --gpu 0 --worker-id 3 --num-workers 8
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import soundfile as sf
import torchaudio.functional as TAF
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

OUTPUT_SR = 16000  # Target output sampling rate

# ── config ──────────────────────────────────────────────────────────
DATASETS = [
    "/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth_onlyenhanced",
]

TEXTS_PATH = "/root/code/github_repos/OmniVoice-fork/batch_generated_text/llm_children_100k_asr_complete.jsonl"
MODEL_PATH = "/root/code/github_repos/OmniVoice-fork/exp/children_finetune_20260519_1418/checkpoints/checkpoint-62000"
OUT_ROOT = Path("/root/code/github_repos/OmniVoice-fork/batch_cloned_voices")

GEN_CONFIG = OmniVoiceGenerationConfig(
    num_step=32, guidance_scale=2.0, class_temperature=0.1,
    denoise=False, preprocess_prompt=True, postprocess_output=True,
)

TEXTS_PER_AUDIO = 10
SPEED_MIN, SPEED_MAX = 0.85, 1.15
SEED = 42

# ── helpers ─────────────────────────────────────────────────────────
def read_kaldi_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                m[parts[0]] = parts[1]
    return m


def load_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append({
                "text": item.get("text", ""),
                "language": item.get("language", "zh"),
            })
    return texts


def clone_one(model, text, ref_audio, ref_text, out_wav, language="zh"):
    speed = round(random.uniform(SPEED_MIN, SPEED_MAX), 2)
    try:
        audio = model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text or None,
            language=language,
            generation_config=GEN_CONFIG,
            speed=speed,
        )
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        # Resample from model SR (24k) to target 16k using high-quality sinc interpolation
        if model.sampling_rate != OUTPUT_SR:
            audio_tensor = torch.from_numpy(audio[0]).float().unsqueeze(0)
            audio_16k = TAF.resample(audio_tensor, model.sampling_rate, OUTPUT_SR).squeeze(0).numpy()
        else:
            audio_16k = audio[0]
        sf.write(str(out_wav), audio_16k, OUTPUT_SR)
        return True, speed
    except Exception as e:
        print(f"      ERROR: {e}")
        return False, speed


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"Worker {args.worker_id}/{args.num_workers} on GPU {args.gpu}")
    all_texts = load_texts(TEXTS_PATH)
    print(f"Loaded {len(all_texts)} texts.")

    print(f"Loading model …")
    model = OmniVoice.from_pretrained(MODEL_PATH, device_map=device, dtype=torch.float16)
    print("Model loaded.\n")

    total_ok = total_fail = total_skip = 0

    for ds in DATASETS:
        ds_name = Path(ds).name

        kaldi_dir = Path(ds) / "kaldi_files"
        if kaldi_dir.exists():
            wav_map = read_kaldi_map(kaldi_dir / "wav.scp")
            text_map = read_kaldi_map(kaldi_dir / "text")
            all_items = [(utt, wav_map[utt], text_map.get(utt, "")) for utt in wav_map]
        else:
            all_items = [(p.stem, str(p), "") for p in Path(ds).rglob("*")
                         if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac")]

        # Shuffle then assign by modulo for balanced random distribution
        rng = random.Random(f"{SEED}:shuffle:{ds_name}")
        shuffled = list(all_items)
        rng.shuffle(shuffled)
        items = [it for i, it in enumerate(shuffled) if i % args.num_workers == args.worker_id]

        if args.limit:
            items = items[: args.limit]

        print(f"[{ds_name}] Worker gets {len(items)} audios")
        if args.dry_run:
            for utt, wav, txt in items[:3]:
                print(f"  {utt}")
            continue

        for i, (utt_id, ref_audio, ref_text) in enumerate(items, 1):
            rng = random.Random(f"{SEED}:{utt_id}")
            sampled_texts = rng.sample(all_texts, TEXTS_PER_AUDIO)

            rel = Path(ref_audio).relative_to(ds)
            base_dir = OUT_ROOT / ds_name / rel.parent / rel.stem

            for tidx, text_item in enumerate(sampled_texts, 1):
                out_wav = base_dir / f"text_{tidx:03d}.wav"
                out_json = base_dir / f"text_{tidx:03d}.json"

                if out_wav.exists():
                    total_skip += 1
                    continue

                # Map lang_type to OmniVoice language code
                lang_map = {"en_mostly": "en", "frequent_mix": "zh", "cn_mostly": "zh"}
                ov_lang = lang_map.get(text_item["language"], text_item["language"])
                ok, speed = clone_one(
                    model, text_item["text"], ref_audio, ref_text, out_wav, ov_lang
                )

                out_json.write_text(
                    json.dumps({
                        "status": "generated" if ok else "failed",
                        "ref_audio": ref_audio,
                        "ref_text": ref_text,
                        "gen_text": text_item["text"],
                        "cloned_audio": str(out_wav) if ok else None,
                        "speed": speed,
                        "language": text_item["language"],
                        "model": MODEL_PATH,
                        "model_sr": model.sampling_rate,
                        "generated_at": datetime.now().isoformat(),
                    }, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                if ok:
                    total_ok += 1
                else:
                    total_fail += 1

            if i % 10 == 0:
                print(f"  [{i}/{len(items)}] {utt_id} ({total_ok} ok, {total_fail} fail, {total_skip} skip)", flush=True)

    print(f"\nWorker {args.worker_id} done! ok={total_ok} fail={total_fail} skip={total_skip}")


if __name__ == "__main__":
    main()
