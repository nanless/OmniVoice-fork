#!/usr/bin/env python3
"""
对所有保存的 checkpoint 批量做 voice clone 推理。
输出目录结构：
  voice_clone_results_v2/ckpt_compare/checkpoint-XXXXX/<utt_id>.wav
"""

import os
import csv
import gc
import torch
import soundfile as sf
from pathlib import Path

CKPT_DIR   = Path("exp/children_finetune_20260519_1418/checkpoints")
TOKENIZER  = Path("/root/.cache/modelscope/k2-fsa/OmniVoice/audio_tokenizer")
METADATA   = Path("voice_clone_results_v2/metadata.txt")
REF_DIR    = Path("voice_clone_results_v2/original")
OUT_BASE   = Path("voice_clone_results_v2/ckpt_compare")

# 按步数排序
checkpoints = sorted(
    [d for d in CKPT_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
    key=lambda d: int(d.name.split("-")[1])
)
print(f"共 {len(checkpoints)} 个 checkpoint：{[d.name for d in checkpoints]}\n")

# 读取 metadata
samples = []
with open(METADATA, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        samples.append(row)
print(f"共 {len(samples)} 条推理样本\n")

from omnivoice import OmniVoice

for ckpt in checkpoints:
    out_dir = OUT_BASE / ckpt.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 确保 audio_tokenizer 符号链接存在
    tok_link = ckpt / "audio_tokenizer"
    if not tok_link.exists():
        tok_link.symlink_to(TOKENIZER)
        print(f"  已创建 symlink: {tok_link}")

    print(f"{'='*60}")
    print(f"加载 {ckpt.name} ...")
    model = OmniVoice.from_pretrained(
        str(ckpt),
        device_map="cuda:0",
        dtype=torch.float16,
    )

    ok = err = skip = 0
    for i, s in enumerate(samples, 1):
        utt_id     = s["utt_id"]
        ref_text   = s["ref_text"]
        clone_text = s["clone_text"]
        ref_audio  = REF_DIR / f"{utt_id}.wav"
        out_path   = out_dir / f"{utt_id}.wav"

        if out_path.exists():
            print(f"  [{i:2d}/{len(samples)}] SKIP  {utt_id} (已存在)")
            skip += 1
            continue
        if not ref_audio.exists():
            print(f"  [{i:2d}/{len(samples)}] SKIP  {utt_id} (ref 不存在)")
            skip += 1
            continue
        try:
            audio = model.generate(
                text=clone_text,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                language="zh",
            )
            sf.write(str(out_path), audio[0], 24000)
            print(f"  [{i:2d}/{len(samples)}] OK    {utt_id}")
            ok += 1
        except Exception as e:
            print(f"  [{i:2d}/{len(samples)}] ERROR {utt_id}: {e}")
            err += 1

    print(f"  {ckpt.name} 完成：ok={ok} skip={skip} err={err}\n")

    # 释放显存，加载下一个
    del model
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n全部完成！结果保存在 {OUT_BASE}")
