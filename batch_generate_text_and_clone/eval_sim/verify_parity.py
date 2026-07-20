#!/usr/bin/env python3
"""Compare eval_sim vs wespeaker v2_organized on the same audio pairs."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, "/root/code/github_repos/wespeaker")

from speaker_encoder import SpeakerEncoder  # noqa: E402
from wespeaker.cli.speaker import load_model_local  # noqa: E402

MODEL_DIR = "/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"
OUT_DIR = Path("/root/code/github_repos/OmniVoice-fork/batch_cloned_voices")


def sample_pairs(n=5):
    pairs = []
    for jp in sorted(OUT_DIR.rglob("text_*.json")):
        if jp.name.endswith((".eval.json", ".sim.json")):
            continue
        meta = json.loads(jp.read_text())
        if meta.get("status") != "generated":
            continue
        ref = meta["ref_audio"]
        clone = meta.get("cloned_audio") or str(jp.with_suffix(".wav"))
        if Path(ref).exists() and Path(clone).exists():
            pairs.append((ref, clone))
        if len(pairs) >= n:
            break
    return pairs


def main():
    device = "cuda:0"
    ws = load_model_local(MODEL_DIR)
    ws.set_device(device)
    es = SpeakerEncoder(EVAL_DIR / "model", device=device)

    print(f"{'pair':>4}  {'emb_ref':>12}  {'emb_clone':>12}  {'sim_ws':>10}  {'sim_es':>10}  {'sim_diff':>10}")
    max_emb_diff = 0.0
    max_sim_diff = 0.0
    for i, (ref, clone) in enumerate(sample_pairs(8)):
        e_ref_ws = ws.extract_embedding(ref)
        e_ref_es = es.extract_embedding(ref)
        e_cl_ws = ws.extract_embedding(clone)
        e_cl_es = es.extract_embedding(clone)

        emb_ref_diff = torch.max(torch.abs(e_ref_ws - e_ref_es)).item()
        emb_cl_diff = torch.max(torch.abs(e_cl_ws - e_cl_es)).item()
        max_emb_diff = max(max_emb_diff, emb_ref_diff, emb_cl_diff)

        # Compare raw cosine directly. wespeaker's convenience method may apply
        # a legacy (cosine + 1) / 2 score transform.
        sim_ws = F.cosine_similarity(e_ref_ws.flatten(), e_cl_ws.flatten(), dim=0).item()
        sim_es = es.cosine_similarity(e_ref_es, e_cl_es)
        sim_diff = abs(sim_ws - sim_es)
        max_sim_diff = max(max_sim_diff, sim_diff)

        print(
            f"{i:>4}  {emb_ref_diff:12.6e}  {emb_cl_diff:12.6e}  "
            f"{sim_ws:10.6f}  {sim_es:10.6f}  {sim_diff:10.6e}"
        )

    print(f"\nmax embedding diff: {max_emb_diff:.6e}")
    print(f"max similarity diff: {max_sim_diff:.6e}")
    ok = max_emb_diff < 1e-4 and max_sim_diff < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
