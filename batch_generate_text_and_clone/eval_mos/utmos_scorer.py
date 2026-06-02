#!/usr/bin/env python3
"""Backward-compatible UTMOS scorer wrapper.

This module re-exports UTMosScorer from scorers.py for backward compatibility.
New code should import from scorers.py directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "TTS_EVAL_MODEL_DIR",
        os.environ.get("UTMOS_MODEL_DIR", REPO_ROOT / "TTS_eval_models"),
    )
)


class UTMosScorer:
    """Backward-compatible UTMOS scorer wrapper around UTMOS22StrongScorer."""

    def __init__(
        self,
        model_dir: Union[str, Path, None] = None,
        device: str = "cuda:0",
    ):
        from scorers import UTMOS22StrongScorer

        self._inner = UTMOS22StrongScorer(model_dir=model_dir or DEFAULT_MODEL_DIR, device=device)
        self.model_dir = self._inner.model_dir
        self.model = self._inner.model
        self.sample_rate = self._inner.sample_rate

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        return self._inner.score_file(wav_path)
