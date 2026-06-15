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
OUT_ROOT = Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned")

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
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append({
                "id": item.get("id") or item.get("text_id") or f"text_{line_no:06d}",
                "text": item.get("text", ""),
                "language": item.get("language", "zh"),
                "lang_type": item.get("lang_type"),
                "length_type": item.get("length_type"),
                "scenario": item.get("scenario") or item.get("scenario_key"),
                "subscene": item.get("subscene"),
                "emotion": item.get("emotion"),
                "age_tier": item.get("age_tier"),
                "text_tn": item.get("text_tn"),
                "task_id": item.get("task_id"),
            })
    return texts


def sample_texts_for_audio(all_texts, utt_id, n=TEXTS_PER_AUDIO):
    """Deterministically pick diverse texts for one reference audio."""
    if len(all_texts) <= n:
        return list(all_texts)

    rng = random.Random(f"{SEED}:{utt_id}")
    pool = list(all_texts)
    rng.shuffle(pool)

    selected = []
    seen = {
        "lang_type": set(),
        "length_type": set(),
        "scenario": set(),
        "age_tier": set(),
    }

    while pool and len(selected) < n:
        def diversity_score(idx):
            item = pool[idx]
            score = 0
            for key, values in seen.items():
                value = item.get(key)
                if value and value not in values:
                    score += 1
            return (score, -idx)

        best_idx = max(range(len(pool)), key=diversity_score)
        item = pool.pop(best_idx)
        selected.append(item)
        for key, values in seen.items():
            value = item.get(key)
            if value:
                values.add(value)

    return selected


SUPPORTED_LANGS = {"zh", "en", "ja", "de", "fr", "es", "ko", "ar", "ru", "pt", "it"}


def resolve_language(raw_lang):
    """Map raw language field to a valid OmniVoice language code, defaulting to 'zh'."""
    lang_map = {"en_mostly": "en", "frequent_mix": "zh", "cn_mostly": "zh"}
    ov_lang = lang_map.get(raw_lang, raw_lang)
    if ov_lang not in SUPPORTED_LANGS:
        return "zh"
    return ov_lang


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
    parser.add_argument(
        "--texts-path",
        default=TEXTS_PATH,
        help="JSONL path with generated clone texts (default: children 100k)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"Worker {args.worker_id}/{args.num_workers} on GPU {args.gpu}")
    texts_path = args.texts_path
    all_texts = load_texts(texts_path)
    print(f"Loaded {len(all_texts)} texts from {texts_path}.")

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
            sampled_texts = sample_texts_for_audio(all_texts, utt_id, TEXTS_PER_AUDIO)

            rel = Path(ref_audio).relative_to(ds)
            base_dir = OUT_ROOT / ds_name / rel.parent / rel.stem

            # Check if all texts for this audio are already done (WAV or JSON exists)
            all_done = True
            for tidx in range(1, TEXTS_PER_AUDIO + 1):
                out_wav = base_dir / f"text_{tidx:03d}.wav"
                out_json = base_dir / f"text_{tidx:03d}.json"
                if not out_wav.exists() and not out_json.exists():
                    all_done = False
                    break
            if all_done:
                total_skip += TEXTS_PER_AUDIO
                continue

            for tidx, text_item in enumerate(sampled_texts, 1):
                out_wav = base_dir / f"text_{tidx:03d}.wav"
                out_json = base_dir / f"text_{tidx:03d}.json"

                if out_wav.exists() or out_json.exists():
                    total_skip += 1
                    continue

                # Ensure output directory exists before generation attempt
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                # Map language to valid OmniVoice language code
                ov_lang = resolve_language(text_item["language"])
                ok, speed = clone_one(
                    model, text_item["text"], ref_audio, ref_text, out_wav, ov_lang
                )

                out_json.write_text(
                    json.dumps({
                        "status": "generated" if ok else "failed",
                        "ref_audio": ref_audio,
                        "ref_text": ref_text,
                        "text_id": text_item.get("id"),
                        "gen_text": text_item["text"],
                        "gen_text_tn": text_item.get("text_tn"),
                        "cloned_audio": str(out_wav) if ok else None,
                        "speed": speed,
                        "language": text_item["language"],
                        "lang_type": text_item.get("lang_type"),
                        "length_type": text_item.get("length_type"),
                        "scenario": text_item.get("scenario"),
                        "subscene": text_item.get("subscene"),
                        "emotion": text_item.get("emotion"),
                        "age_tier": text_item.get("age_tier"),
                        "task_id": text_item.get("task_id"),
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
