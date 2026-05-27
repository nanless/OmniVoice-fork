#!/usr/bin/env python3
"""Self-contained samresnet100 speaker encoder for eval_sim.

Preprocessing matches wespeaker.cli.speaker.Speaker (v2_organized pipeline):
  torchaudio.load(..., normalize=False) -> float pcm -> Resample -> fbank * (1<<15) -> CMN
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from models.samresnet import SimAMResNet100ASP

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = EVAL_DIR / "model"


def load_checkpoint(model: torch.nn.Module, path: Union[str, Path]) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)


class SpeakerEncoder:
    """Extract 256-d speaker embeddings; scoring matches wespeaker CLI."""

    def __init__(self, model_dir: Union[str, Path] | None = None, device: str = "cuda:0"):
        model_dir = Path(model_dir or os.environ.get("EVAL_SIM_MODEL_DIR", DEFAULT_MODEL_DIR))
        config_path = model_dir / "config.yaml"
        weights_path = model_dir / "avg_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing {weights_path}. Place avg_model.pt under eval_sim/model/."
            )

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        model_args = dict(cfg.get("model_args", {}))
        self.model = SimAMResNet100ASP(**model_args)
        load_checkpoint(self.model, weights_path)
        self.model.eval()

        self.device = torch.device(device)
        self.model.to(self.device)
        self.resample_rate = 16000
        fbank_args = cfg.get("dataset_args", {}).get("fbank_args", {})
        self.num_mel_bins = int(fbank_args.get("num_mel_bins", 80))
        self.frame_length = int(fbank_args.get("frame_length", 25))
        self.frame_shift = int(fbank_args.get("frame_shift", 10))
        self.model_dir = str(model_dir)

    def compute_fbank(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        cmn: bool = True,
    ) -> torch.Tensor:
        """Same as wespeaker Speaker.compute_fbank (waveform already at target sr)."""
        waveform = waveform * (1 << 15)
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            sample_frequency=sample_rate,
            window_type="hamming",
        )
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
        """Mirror wespeaker Speaker.extract_embedding_from_pcm (VAD disabled)."""
        pcm = pcm.to(torch.float)
        if sample_rate != self.resample_rate:
            pcm = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate
            )(pcm)
        feats = self.compute_fbank(pcm, sample_rate=self.resample_rate, cmn=True)
        feats = feats.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        return outputs[0].cpu()

    def extract_embedding(self, audio_path: Union[str, Path]) -> Optional[torch.Tensor]:
        pcm, sample_rate = torchaudio.load(str(audio_path), normalize=False)
        return self.extract_embedding_from_pcm(pcm, sample_rate)

    @staticmethod
    def cosine_similarity(e1, e2) -> float:
        if isinstance(e1, torch.Tensor):
            e1 = e1.detach().cpu()
        if isinstance(e2, torch.Tensor):
            e2 = e2.detach().cpu()
        score = torch.dot(e1, e2) / (torch.norm(e1) * torch.norm(e2))
        return float((score.item() + 1.0) / 2.0)

    def compute_similarity(
        self, audio_path1: Union[str, Path], audio_path2: Union[str, Path]
    ) -> Optional[float]:
        e1 = self.extract_embedding(audio_path1)
        e2 = self.extract_embedding(audio_path2)
        if e1 is None or e2 is None:
            return None
        return self.cosine_similarity(e1, e2)
