#!/usr/bin/env python3
"""UTMOS scorer — loads omnivoice/eval code without importing omnivoice package."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Union

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "omnivoice" / "eval"

DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "TTS_EVAL_MODEL_DIR",
        os.environ.get("UTMOS_MODEL_DIR", REPO_ROOT / "TTS_eval_models"),
    )
)
UTMOS_CKPT_NAME = "mos/utmos22_strong_step7459_v1.pt"
TARGET_SR = 16000


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_utmos_mod = _load_module("omnivoice_eval_utmos", EVAL_ROOT / "models" / "utmos.py")
_utils_mod = _load_module("omnivoice_eval_utils", EVAL_ROOT / "utils.py")

UTMOS22Strong = _utmos_mod.UTMOS22Strong
load_eval_waveform = _utils_mod.load_eval_waveform


def resolve_model_path(model_dir: Union[str, Path] | None = None) -> Path:
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    ckpt = model_dir / UTMOS_CKPT_NAME
    if not ckpt.exists():
        alt = REPO_ROOT / "download" / "tts_eval_models" / UTMOS_CKPT_NAME
        if alt.exists():
            return alt
        raise FileNotFoundError(
            f"UTMOS checkpoint not found at {ckpt}.\n"
            "Download from https://huggingface.co/k2-fsa/TTS_eval_models:\n"
            "  huggingface-cli download --local-dir TTS_eval_models k2-fsa/TTS_eval_models "
            "mos/utmos22_strong_step7459_v1.pt"
        )
    return ckpt


class UTMosScorer:
    """Score audio with UTMOS22Strong (omnivoice/eval/mos/utmos.py)."""

    def __init__(
        self,
        model_dir: Union[str, Path] | None = None,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        ckpt = resolve_model_path(model_dir)
        self.model_dir = str(ckpt.parent.parent)
        self.model = UTMOS22Strong()
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = TARGET_SR

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        speech = load_eval_waveform(
            str(wav_path), self.sample_rate, device=self.device
        )
        score = self.model(speech.unsqueeze(0), self.sample_rate)
        return float(score.item())
