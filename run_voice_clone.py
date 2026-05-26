#!/usr/bin/env python3
"""Optimized voice cloning script using OmniVoice.

Randomly picks 20 speakers, concatenates their utterances into 3-10s
reference clips if needed, and uses higher-quality generation settings.
"""

import os
import shutil
import random
import torch
import soundfile as sf
import numpy as np
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KALDI_DIR = "/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth/kaldi_files"
MODEL_PATH = "/root/.cache/modelscope/k2-fsa/OmniVoice"
OUTPUT_DIR = "/root/code/github_repos/OmniVoice/voice_clone_results_v2"
NUM_SAMPLES = 20
REF_MIN_DURATION = 3.0
REF_MAX_DURATION = 10.0
TARGET_REF_DURATION = 6.0  # target midpoint
SAMPLE_RATE = 24000  # OmniVoice audio tokenizer sample rate

# Short text for voice cloning target
CLONE_TEXT = "你好，欢迎使用语音克隆。"

# Higher-quality generation config
gen_config = OmniVoiceGenerationConfig(
    num_step=50,
    guidance_scale=2.5,
    class_temperature=0.0,
    denoise=False,
    preprocess_prompt=True,
    postprocess_output=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading OmniVoice model ...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = OmniVoice.from_pretrained(
    MODEL_PATH,
    device_map=device,
    dtype=torch.float32,
)
print(f"Model loaded on {device}")

# ---------------------------------------------------------------------------
# Read data files
# ---------------------------------------------------------------------------
with open(os.path.join(KALDI_DIR, "wav.scp")) as f:
    wav_lines = [l.strip() for l in f if l.strip()]

with open(os.path.join(KALDI_DIR, "text")) as f:
    text_lines = [l.strip() for l in f if l.strip()]

text_map = {}
for line in text_lines:
    parts = line.split(None, 1)
    if len(parts) == 2:
        text_map[parts[0]] = parts[1]

# ---------------------------------------------------------------------------
# Group by speaker, get durations
# ---------------------------------------------------------------------------
spk_to_utts = {}  # spk -> list of (utt_id, wav_path, ref_text, dur)
for line in wav_lines:
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    utt_id, wav_path = parts
    if not os.path.exists(wav_path):
        continue
    try:
        info = sf.info(wav_path)
        dur = info.duration
    except Exception:
        continue
    spk = utt_id.split("_")[0]
    ref_text = text_map.get(utt_id, "")
    spk_to_utts.setdefault(spk, []).append((utt_id, wav_path, ref_text, dur))

print(f"Total speakers: {len(spk_to_utts)}")

# ---------------------------------------------------------------------------
# Randomly pick 20 speakers, concatenate if needed
# ---------------------------------------------------------------------------
random.seed(42)
all_spks = list(spk_to_utts.keys())
random.shuffle(all_spks)

selected = []  # list of (utt_id, wav_path, ref_text, dur, spk, concat_utts)

for spk in all_spks:
    if len(selected) >= NUM_SAMPLES:
        break

    utts = spk_to_utts[spk]
    # shuffle utterances for randomness
    random.shuffle(utts)

    # Try to find a single utterance in the 3-10s range
    single_candidates = [u for u in utts if REF_MIN_DURATION <= u[3] <= REF_MAX_DURATION]
    if single_candidates:
        # pick the one closest to target duration
        best = min(single_candidates, key=lambda x: abs(x[3] - TARGET_REF_DURATION))
        selected.append((best[0], best[1], best[2], best[3], spk, None))
        continue

    # Otherwise, concatenate multiple utterances to reach 3-10s
    concat_utts = []
    total_dur = 0.0
    for u in utts:
        concat_utts.append(u)
        total_dur += u[3]
        if total_dur >= REF_MIN_DURATION:
            break

    if total_dur >= REF_MIN_DURATION and total_dur <= REF_MAX_DURATION:
        # Use first utt_id + "_concat" as combined id
        first_utt = concat_utts[0][0]
        concat_id = f"{first_utt}_concat"
        selected.append((concat_id, None, None, total_dur, spk, concat_utts))
    elif total_dur >= REF_MIN_DURATION:
        # Too long, trim by removing last utt until within range
        while total_dur > REF_MAX_DURATION and len(concat_utts) > 1:
            removed = concat_utts.pop()
            total_dur -= removed[3]
        if total_dur >= REF_MIN_DURATION:
            first_utt = concat_utts[0][0]
            concat_id = f"{first_utt}_concat"
            selected.append((concat_id, None, None, total_dur, spk, concat_utts))

print(f"Selected {len(selected)} speakers for voice cloning")
for utt_id, wav_path, ref_text, dur, spk, concat_utts in selected:
    if concat_utts:
        print(f"  {utt_id} (spk={spk}, dur={dur:.2f}s): concatenated {len(concat_utts)} utts")
    else:
        print(f"  {utt_id} (spk={spk}, dur={dur:.2f}s): {ref_text}")

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(OUTPUT_DIR, "original"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "cloned"), exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "metadata.txt"), "w", encoding="utf-8") as f:
    f.write("utt_id\tspk\tduration\tref_text\tclone_text\n")

# ---------------------------------------------------------------------------
# Helper: concatenate audio files
# ---------------------------------------------------------------------------
def concatenate_audios(utt_list, out_path, target_sr=24000):
    """Concatenate multiple wav files with 0.3s silence between them."""
    segments = []
    ref_texts = []
    for utt_id, wav_path, ref_text, _ in utt_list:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        if sr != target_sr:
            # Simple resampling with scipy if needed
            import scipy.signal as sps
            num_samples = int(len(wav) * target_sr / sr)
            wav = sps.resample(wav, num_samples)
        segments.append(wav)
        ref_texts.append(ref_text)
        # Add 0.3s silence between segments
        segments.append(np.zeros(int(0.3 * target_sr), dtype=wav.dtype))

    # Remove trailing silence
    if segments:
        segments = segments[:-1]

    combined = np.concatenate(segments)
    sf.write(out_path, combined, target_sr)
    combined_ref_text = " ".join([t for t in ref_texts if t])
    return combined_ref_text


# ---------------------------------------------------------------------------
# Perform voice cloning for each sample
# ---------------------------------------------------------------------------
for idx, (utt_id, wav_path, ref_text, dur, spk, concat_utts) in enumerate(selected, 1):
    print(f"\n[{idx}/{len(selected)}] Processing {utt_id} (spk={spk}, dur={dur:.2f}s)")

    if concat_utts:
        # Build concatenated reference audio
        concat_wav_path = os.path.join(OUTPUT_DIR, "original", f"{utt_id}.wav")
        ref_text = concatenate_audios(concat_utts, concat_wav_path)
        wav_path = concat_wav_path
        print(f"  Concatenated {len(concat_utts)} utterances, ref_text: {ref_text}")
    else:
        # Copy original single audio
        orig_ext = os.path.splitext(wav_path)[1]
        orig_out = os.path.join(OUTPUT_DIR, "original", f"{utt_id}{orig_ext}")
        shutil.copy2(wav_path, orig_out)
        wav_path = orig_out

    print(f"  Saved original -> {wav_path}")

    # Voice clone with optimized settings
    try:
        audio = model.generate(
            text=CLONE_TEXT,
            ref_audio=wav_path,
            ref_text=ref_text if ref_text else None,
            language="zh",
            generation_config=gen_config,
        )
        cloned_out = os.path.join(OUTPUT_DIR, "cloned", f"{utt_id}_cloned.wav")
        sf.write(cloned_out, audio[0], model.sampling_rate)
        print(f"  Saved cloned   -> {cloned_out}")
    except Exception as e:
        print(f"  ERROR during cloning: {e}")
        import traceback
        traceback.print_exc()
        continue

    with open(os.path.join(OUTPUT_DIR, "metadata.txt"), "a", encoding="utf-8") as f:
        f.write(f"{utt_id}\t{spk}\t{dur:.2f}\t{ref_text}\t{CLONE_TEXT}\n")

print(f"\nDone! Results saved to: {OUTPUT_DIR}")
