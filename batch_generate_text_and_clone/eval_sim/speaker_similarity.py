#!/usr/bin/env python3
"""Speaker similarity helpers for eval_sim."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from speaker_encoder import SpeakerEncoder

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = EVAL_DIR / "model"


def load_encoder(model_dir: str | None = None, device: str = "cuda:0") -> SpeakerEncoder:
    return SpeakerEncoder(model_dir=model_dir, device=device)


def embedding_to_numpy(emb) -> np.ndarray:
    if isinstance(emb, torch.Tensor):
        arr = emb.detach().cpu().numpy()
    else:
        arr = np.asarray(emb)
    return arr.flatten().astype(np.float32)


def cosine_similarity(e1, e2) -> float:
    """Return unshifted, unscaled cosine similarity in [-1, 1]."""
    if isinstance(e1, np.ndarray):
        e1 = torch.from_numpy(e1)
    if isinstance(e2, np.ndarray):
        e2 = torch.from_numpy(e2)
    return SpeakerEncoder.cosine_similarity(e1, e2)


def compute_similarity(
    encoder: SpeakerEncoder, path1: str, path2: str
) -> Optional[float]:
    e1 = encoder.extract_embedding(path1)
    e2 = encoder.extract_embedding(path2)
    if e1 is None or e2 is None:
        return None
    return cosine_similarity(e1, e2)
