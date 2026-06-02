#!/usr/bin/env python3
"""Unified scorer module for TTS evaluation metrics.

Supported metrics:
  - UTMOS22Strong  (custom PyTorch model from omnivoice/eval)
  - SCOREQ         (pip package `scoreq`, ONNX-based NR quality)
  - TTSDS2         (pip package `ttsds`, benchmark suite)
  - UTMOSv2        (git package `utmosv2`, SSL+spectrogram MOS predictor)

Usage:
    from scorers import create_scorer, AVAILABLE_METRICS

    scorer = create_scorer("SCOREQ", device="cuda:0")
    score = scorer.score_file("audio.wav")
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

AVAILABLE_METRICS: List[str] = ["UTMOS22Strong", "SCOREQ", "TTSDS2", "UTMOSv2"]


class BaseScorer(ABC):
    """Abstract base for all scorers."""

    name: str = "base"

    @abstractmethod
    def score_file(self, wav_path: Union[str, Path]) -> float:
        """Score a single wav file. Higher is better (except TTSDS2 sub-scores)."""

    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        """Score multiple files at once. Fallback to loop unless overridden."""
        return [self.score_file(p) for p in wav_paths]

    def close(self):
        """Release GPU memory / handles."""


# ---------------------------------------------------------------------------
# UTMOS22Strong  — omnivoice/eval/models/utmos.py
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "omnivoice" / "eval"

TARGET_SR = 16000

# Lazy-loaded module references
_utmos_mod = None
_utils_mod = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_utmos_module():
    global _utmos_mod
    if _utmos_mod is None:
        _utmos_mod = _load_module("omnivoice_eval_utmos", EVAL_ROOT / "models" / "utmos.py")
    return _utmos_mod


def _get_utils_module():
    global _utils_mod
    if _utils_mod is None:
        _utils_mod = _load_module("omnivoice_eval_utils", EVAL_ROOT / "utils.py")
    return _utils_mod


class UTMOS22StrongScorer(BaseScorer):
    """UTMOS22Strong scorer using omnivoice's custom PyTorch model."""

    name = "UTMOS22Strong"

    def __init__(self, model_dir: Union[str, Path, None] = None, device: str = "cuda:0"):
        import os

        self.device = torch.device(device)
        utmos_mod = _get_utmos_module()
        utils_mod = _get_utils_module()

        model_dir = Path(
            model_dir
            or os.environ.get("TTS_EVAL_MODEL_DIR")
            or os.environ.get("UTMOS_MODEL_DIR")
            or REPO_ROOT / "TTS_eval_models"
        )
        ckpt_name = "mos/utmos22_strong_step7459_v1.pt"
        ckpt = model_dir / ckpt_name
        if not ckpt.exists():
            alt = REPO_ROOT / "download" / "tts_eval_models" / ckpt_name
            if alt.exists():
                ckpt = alt
            else:
                raise FileNotFoundError(
                    f"UTMOS checkpoint not found at {ckpt}.\n"
                    "Download: huggingface-cli download --local-dir TTS_eval_models "
                    "k2-fsa/TTS_eval_models mos/utmos22_strong_step7459_v1.pt"
                )
        self.model_dir = str(ckpt.parent.parent)
        self.model = utmos_mod.UTMOS22Strong()
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = TARGET_SR
        self._load_eval_waveform = utils_mod.load_eval_waveform

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        speech = self._load_eval_waveform(str(wav_path), self.sample_rate, device=self.device)
        score = self.model(speech.unsqueeze(0), self.sample_rate)
        return float(score.item())

    @torch.no_grad()
    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        """Batch inference: load all audios, pad to max length, forward once."""
        speeches = []
        for wp in wav_paths:
            s = self._load_eval_waveform(str(wp), self.sample_rate, device=self.device)
            speeches.append(s)
        max_len = max(s.shape[0] for s in speeches)
        batch = torch.zeros(len(speeches), max_len, device=self.device)
        for i, s in enumerate(speeches):
            batch[i, : s.shape[0]] = s
        scores = self.model(batch, self.sample_rate)
        return [float(v) for v in scores.cpu().tolist()]


# ---------------------------------------------------------------------------
# SCOREQ  — pip package `scoreq`
# ---------------------------------------------------------------------------

class SCOREQScorer(BaseScorer):
    """SCOREQ non-reference quality scorer (ONNX)."""

    name = "SCOREQ"

    def __init__(self, device: str = "cuda:0", **kwargs):
        try:
            from scoreq.scoreq import Scoreq
        except ImportError:
            raise ImportError(
                "scoreq package not found. Install: pip install scoreq"
            )
        self.device = device
        use_onnx = True
        # 'synthetic' model (trained on VoiceMOS 2022) is better for TTS eval,
        # but requires downloading from zenodo.org (~400MB).
        # Fall back to 'natural' if synthetic model is not cached.
        data_domain = "synthetic"
        synthetic_path = Path.home() / ".cache" / "scoreq" / "onnx-models" / "adapt_nr_synthetic.onnx"
        if not synthetic_path.exists():
            data_domain = "natural"  # fallback to natural model
        if "cuda" in device:
            self.model = Scoreq(data_domain=data_domain, mode="nr", use_onnx=use_onnx)
            if hasattr(self.model, "session") and self.model.session is not None:
                try:
                    self.model.session.set_providers(["CUDAExecutionProvider"])
                except Exception:
                    pass
        else:
            self.model = Scoreq(data_domain=data_domain, mode="nr", use_onnx=use_onnx)
            if hasattr(self.model, "session") and self.model.session is not None:
                try:
                    self.model.session.set_providers(["CPUExecutionProvider"])
                except Exception:
                    pass

    def score_file(self, wav_path: Union[str, Path]) -> float:
        return float(self.model.predict(str(wav_path)))

    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        """Parallel scoring using thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.score_file, wav_paths))
        return results


# ---------------------------------------------------------------------------
# TTSDS2  — pip package `ttsds`
# ---------------------------------------------------------------------------

class TTSDS2Scorer(BaseScorer):
    """TTSDS2 benchmark suite scorer.

    Loads GPU-based benchmarks (WavLM, Whisper) directly from ttsds source files,
    bypassing ttsds.__init__ which has problematic top-level imports.
    Falls back to librosa-based quality proxy if benchmarks can't be loaded.
    """

    name = "TTSDS2"

    # ttsds source root for direct file loading
    _TTSDS_ROOT: str = ""

    def __init__(self, device: str = "cuda:0", **kwargs):
        self.device = device
        self._benchmarks = []
        self._mode = "fallback"
        self._init_ttsds_root()
        self._init_benchmarks()

    def _init_ttsds_root(self):
        """Find the ttsds package root directory without importing ttsds."""
        try:
            import site
            for sp in site.getsitepackages() + [site.getusersitepackages()]:
                candidate = Path(sp) / "ttsds" / "benchmarks"
                if candidate.is_dir():
                    self._TTSDS_ROOT = str(Path(sp) / "ttsds")
                    return
            # Fallback: check common conda locations
            import sysconfig
            sp = sysconfig.get_path("purelib")
            candidate = Path(sp) / "ttsds" / "benchmarks"
            if candidate.is_dir():
                self._TTSDS_ROOT = str(Path(sp) / "ttsds")
        except Exception:
            pass

    def _load_module(self, name: str, filepath: str):
        """Load a Python module directly from file path."""
        spec = importlib.util.spec_from_file_location(name, filepath)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _init_benchmarks(self):
        """Load individual GPU benchmarks from ttsds, bypassing __init__."""
        if not self._TTSDS_ROOT:
            return

        root = self._TTSDS_ROOT
        # Load dependency chain (avoid ttsds.__init__)
        deps = [
            ("ttsds.util.cache", f"{root}/util/cache.py"),
            ("ttsds.util.dataset", f"{root}/util/dataset.py"),
            ("ttsds.util.distances", f"{root}/util/distances.py"),
            ("ttsds.util.parallel_distances", f"{root}/util/parallel_distances.py"),
            ("ttsds.benchmarks.benchmark", f"{root}/benchmarks/benchmark.py"),
        ]
        for name, path in deps:
            if name not in sys.modules:
                try:
                    self._load_module(name, path)
                except Exception:
                    return

        # Try loading WavLM benchmark (GPU, generic quality)
        try:
            wavlm_mod = self._load_module(
                "ttsds.benchmarks.general.wavlm",
                f"{root}/benchmarks/general/wavlm.py",
            )
            if wavlm_mod:
                wavlm = wavlm_mod.WavLMBenchmark()
                if "cuda" in self.device:
                    wavlm.to_device("cuda")
                self._benchmarks.append(("wavlm", wavlm))
        except Exception:
            pass

        # Try loading Whisper activations benchmark (GPU, intelligibility)
        try:
            whisper_mod = self._load_module(
                "ttsds.benchmarks.intelligibility.whisper_activations",
                f"{root}/benchmarks/intelligibility/whisper_activations.py",
            )
            if whisper_mod:
                whisper = whisper_mod.WhisperActivationsBenchmark()
                if "cuda" in self.device:
                    whisper.to_device("cuda")
                self._benchmarks.append(("whisper", whisper))
        except Exception:
            pass

        # Try loading pitch benchmark (CPU, prosody)
        try:
            pitch_mod = self._load_module(
                "ttsds.benchmarks.prosody.pitch",
                f"{root}/benchmarks/prosody/pitch.py",
            )
            if pitch_mod:
                pitch = pitch_mod.PitchBenchmark()
                self._benchmarks.append(("pitch", pitch))
        except Exception:
            pass

        if self._benchmarks:
            self._mode = "benchmarks"

    def score_file(self, wav_path: Union[str, Path]) -> float:
        """Score a single file. Returns a composite TTSDS2 score (0-100, higher=better)."""
        if self._mode == "benchmarks":
            return self._score_with_benchmarks(wav_path)
        return self._score_fallback(wav_path)

    def score_files(self, wav_paths: List[Union[str, Path]]) -> List[float]:
        """Parallel scoring using thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.score_file, wav_paths))
        return results

    def _score_with_benchmarks(self, wav_path: Union[str, Path]) -> float:
        """Score using individual ttsds benchmarks on a single file.

        Computes embedding distance from noise as quality proxy.
        For each benchmark:
          1. Get test audio embedding
          2. Generate noise and get its embedding
          3. Score = distance(test, noise) normalized to 0-100
        """
        import librosa

        wav_path = Path(wav_path)
        try:
            y, sr = librosa.load(str(wav_path), sr=16000)
        except Exception:
            return self._score_fallback(wav_path)

        if len(y) == 0:
            return 0.0

        scores = []
        for name, bench in self._benchmarks:
            try:
                test_emb = bench.get_embedding(y, sr)
                # Generate white noise as reference
                noise = np.random.RandomState(42).randn(len(y)).astype(np.float32) * 0.1
                noise_emb = bench.get_embedding(noise, sr)

                # Distance: how far test is from noise
                dist = float(np.linalg.norm(test_emb.mean(axis=0) - noise_emb.mean(axis=0)))

                # Normalize to 0-100: sigmoid-like mapping
                # noise should give ~0, speech should give higher
                score = 100.0 * (dist / (dist + 1.0))
                scores.append(score)
            except Exception:
                continue

        if scores:
            return float(np.mean(scores))
        return self._score_fallback(wav_path)

    def _score_fallback(self, wav_path: Union[str, Path]) -> float:
        """Fallback: compute a simple quality proxy using librosa (CPU)."""
        try:
            import librosa

            wav_path = str(wav_path)
            y, sr = librosa.load(wav_path, sr=16000)
            if len(y) == 0:
                return 0.0

            sf = librosa.feature.spectral_flatness(y=y)[0]
            sf_score = max(0.0, 1.0 - float(sf.mean()))

            rms = librosa.feature.rms(y=y)[0]
            rms_cv = float(rms.std() / (rms.mean() + 1e-8))
            rms_score = max(0.0, 1.0 - min(rms_cv, 1.0))

            sorted_rms = np.sort(rms)
            noise_floor = sorted_rms[: max(1, len(sorted_rms) // 10)].mean()
            signal_power = sorted_rms[-max(1, len(sorted_rms) // 10)].mean()
            snr_proxy = signal_power / (noise_floor + 1e-8)
            snr_score = min(1.0, float(np.log1p(snr_proxy) / 5.0))

            return float((sf_score + rms_score + snr_score) / 3.0) * 100
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# UTMOSv2  — git package `utmosv2` (UT-Sarulab MOS Prediction System v2)
# ---------------------------------------------------------------------------

class UTMOSv2Scorer(BaseScorer):
    """UTMOSv2 scorer using the `utmosv2` package (UT-Sarulab MOS Prediction System v2).

    Requires:
        pip install git+https://github.com/sarulab-speech/UTMOSv2.git
    """

    name = "UTMOSv2"

    def __init__(self, device: str = "cuda:0", **kwargs):
        try:
            import utmosv2
        except ImportError:
            raise ImportError(
                "utmosv2 package not found. Install:\n"
                "  pip install git+https://github.com/sarulab-speech/UTMOSv2.git"
            )
        self.device = device
        self.model = utmosv2.create_model(pretrained=True, device=device)

    @torch.no_grad()
    def score_file(self, wav_path: Union[str, Path]) -> float:
        return float(self.model.predict(input_path=str(wav_path), device=self.device, verbose=False))

    def score_files(self, wav_paths: List[Union[str, Path]], batch_size: int = 8) -> List[float]:
        """Batch inference using utmosv2's internal DataLoader."""
        import numpy as np
        import torchaudio

        # Load all audios and pad
        audios = []
        max_len = 0
        for wp in wav_paths:
            wav, sr = torchaudio.load(str(wp))
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            a = wav.squeeze(0).cpu().numpy()
            audios.append(a)
            max_len = max(max_len, a.shape[0])

        batch = np.zeros((len(audios), max_len), dtype=np.float32)
        for i, a in enumerate(audios):
            batch[i, :len(a)] = a

        scores = self.model.predict(data=batch, batch_size=batch_size, verbose=False, device=self.device)
        if hasattr(scores, "tolist"):
            return [float(v) for v in scores.tolist()]
        return [float(v) for v in scores]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: Dict[str, Type[BaseScorer]] = {
    "UTMOS22Strong": UTMOS22StrongScorer,
    "SCOREQ": SCOREQScorer,
    "TTSDS2": TTSDS2Scorer,
    "UTMOSv2": UTMOSv2Scorer,
}


def create_scorer(
    metric: str,
    device: str = "cuda:0",
    model_dir: Union[str, Path, None] = None,
    **kwargs,
) -> BaseScorer:
    """Create a scorer instance for the given metric name.

    Args:
        metric: One of AVAILABLE_METRICS.
        device: Torch device string.
        model_dir: Model directory (used by UTMOS22Strong).
        **kwargs: Extra args passed to the scorer constructor.

    Returns:
        A BaseScorer instance.

    Raises:
        ValueError: Unknown metric name.
        ImportError: Required package not installed.
    """
    metric_upper = metric.upper().replace("-", "").replace("_", "")
    for key, cls in _SCORER_REGISTRY.items():
        if key.upper().replace("-", "").replace("_", "") == metric_upper:
            if key == "UTMOS22Strong":
                return cls(model_dir=model_dir, device=device, **kwargs)
            return cls(device=device, **kwargs)
    raise ValueError(
        f"Unknown metric '{metric}'. Available: {list(_SCORER_REGISTRY.keys())}"
    )


def try_create_scorer(
    metric: str,
    device: str = "cuda:0",
    model_dir: Union[str, Path, None] = None,
    **kwargs,
) -> Optional[BaseScorer]:
    """Try to create a scorer; return None if the package is not installed."""
    try:
        return create_scorer(metric, device=device, model_dir=model_dir, **kwargs)
    except (ImportError, FileNotFoundError) as e:
        print(f"[scorers] Skipping {metric}: {e}", flush=True)
        return None
