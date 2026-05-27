#!/usr/bin/env python3
"""Embedding cache and similarity helpers for eval_sim."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Union

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
    if isinstance(e1, np.ndarray):
        e1 = torch.from_numpy(e1)
    if isinstance(e2, np.ndarray):
        e2 = torch.from_numpy(e2)
    return SpeakerEncoder.cosine_similarity(e1, e2)


class EmbeddingCache:
    """Disk-backed cache: audio_path -> embedding ndarray."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, np.ndarray] = {}

    def _key_path(self, audio_path: str) -> Path:
        safe = audio_path.replace("/", "__")
        return self.cache_dir / f"{safe}.pkl"

    def get(self, audio_path: str) -> Optional[np.ndarray]:
        if audio_path in self._mem:
            return self._mem[audio_path]
        p = self._key_path(audio_path)
        if p.exists():
            with open(p, "rb") as f:
                data = pickle.load(f)
            emb = data.get("embedding")
            if emb is not None:
                emb = embedding_to_numpy(emb)
                self._mem[audio_path] = emb
                return emb
        return None

    def put(self, audio_path: str, embedding) -> np.ndarray:
        emb = embedding_to_numpy(embedding)
        self._mem[audio_path] = emb
        with open(self._key_path(audio_path), "wb") as f:
            pickle.dump({"audio_path": audio_path, "embedding": emb}, f)
        return emb

    def get_or_extract(self, encoder: SpeakerEncoder, audio_path: str) -> Optional[np.ndarray]:
        cached = self.get(audio_path)
        if cached is not None:
            return cached
        emb = encoder.extract_embedding(audio_path)
        if emb is None:
            return None
        return self.put(audio_path, emb)

    def similarity(self, encoder: SpeakerEncoder, path1: str, path2: str) -> Optional[float]:
        e1 = self.get_or_extract(encoder, path1)
        e2 = self.get_or_extract(encoder, path2)
        if e1 is None or e2 is None:
            return None
        return cosine_similarity(e1, e2)
