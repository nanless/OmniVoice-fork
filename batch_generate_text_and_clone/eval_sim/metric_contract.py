"""Versioned contract for speaker-similarity scores."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


SIMILARITY_METRIC = "raw_cosine"
SIMILARITY_SCORE_VERSION = 2
SIMILARITY_RANGE = (-1.0, 1.0)


def similarity_metadata() -> dict:
    """Return metadata that makes the score semantics explicit in JSON outputs."""
    return {
        "similarity_metric": SIMILARITY_METRIC,
        "score_version": SIMILARITY_SCORE_VERSION,
        "similarity_range": list(SIMILARITY_RANGE),
    }


def validate_raw_cosine_record(record: Mapping[str, Any], source: str | Path = "record") -> None:
    """Raise when a record is legacy, mixed-version, or outside cosine bounds."""
    metric = record.get("similarity_metric")
    version = record.get("score_version")
    if metric != SIMILARITY_METRIC or version != SIMILARITY_SCORE_VERSION:
        raise ValueError(
            f"{source}: unsupported similarity schema "
            f"(similarity_metric={metric!r}, score_version={version!r}); "
            "expected raw_cosine score_version=2. Re-run eval_sim. "
            "For legacy normalized scores only, raw_cosine = 2 * similarity - 1."
        )

    if record.get("similarity_range") != list(SIMILARITY_RANGE):
        raise ValueError(
            f"{source}: similarity_range must be {list(SIMILARITY_RANGE)!r}, "
            f"got {record.get('similarity_range')!r}"
        )

    for field in ("model_signature", "cloned_audio_signature", "ref_audio_signature"):
        if not isinstance(record.get(field), dict) or not record[field]:
            raise ValueError(f"{source}: missing or invalid {field}")
    for field in ("cloned_audio_signature", "ref_audio_signature"):
        signature = record[field]
        if set(signature) != {"size", "mtime_ns"} or not all(
            isinstance(signature[key], int) and signature[key] >= 0
            for key in ("size", "mtime_ns")
        ):
            raise ValueError(f"{source}: invalid {field}: {signature!r}")
    model_signature = record["model_signature"]
    if set(model_signature) != {"config.yaml", "avg_model.pt"}:
        raise ValueError(f"{source}: invalid model_signature files")
    for name, signature in model_signature.items():
        if (
            not isinstance(signature, dict)
            or set(signature) != {"size", "mtime_ns", "sha256"}
            or not isinstance(signature["size"], int)
            or signature["size"] < 0
            or not isinstance(signature["mtime_ns"], int)
            or signature["mtime_ns"] < 0
            or not isinstance(signature["sha256"], str)
            or len(signature["sha256"]) != 64
        ):
            raise ValueError(f"{source}: invalid model_signature for {name}: {signature!r}")

    value = record.get("similarity")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: similarity must be a number or null, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not SIMILARITY_RANGE[0] <= numeric <= SIMILARITY_RANGE[1]:
        raise ValueError(
            f"{source}: raw cosine similarity must be finite and in [-1, 1], got {value!r}"
        )


def is_complete_raw_cosine_record(record: Mapping[str, Any]) -> bool:
    """Return whether a record is a successful, reusable v2 raw-cosine result."""
    try:
        validate_raw_cosine_record(record)
    except ValueError:
        return False
    return record.get("similarity") is not None


class SimilarityCollectionValidator:
    """Fail closed on mixed models or conflicting duplicate audio records."""

    def __init__(self) -> None:
        self.model_dir: str | None = None
        self.model_signature: Mapping[str, Any] | None = None
        self.records: dict[str, Mapping[str, Any]] = {}

    def add(self, record: Mapping[str, Any], source: str | Path) -> bool:
        """Validate and add a record; return False for an identical duplicate."""
        validate_raw_cosine_record(record, source)
        model_dir = record.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError(f"{source}: missing model_dir in similarity record")
        if self.model_dir is None:
            self.model_dir = model_dir
        elif model_dir != self.model_dir:
            raise ValueError(
                f"{source}: mixed similarity models are not allowed: "
                f"{model_dir!r} vs {self.model_dir!r}"
            )
        model_signature = record["model_signature"]
        if self.model_signature is None:
            self.model_signature = model_signature
        elif model_signature != self.model_signature:
            raise ValueError(f"{source}: mixed similarity model signatures are not allowed")

        wav = record.get("cloned_audio")
        if not isinstance(wav, str) or not wav:
            raise ValueError(f"{source}: missing cloned_audio in similarity record")
        previous = self.records.get(wav)
        if previous is None:
            self.records[wav] = record
            return True
        for field in (
            "similarity",
            "ref_audio",
            "model_dir",
            "model_signature",
            "dataset",
            "language",
            "speed",
        ):
            old_value = previous.get(field)
            new_value = record.get(field)
            same = old_value == new_value
            if field == "similarity" and old_value is not None and new_value is not None:
                same = math.isclose(
                    float(old_value), float(new_value), rel_tol=0.0, abs_tol=1e-6
                )
            if not same:
                raise ValueError(
                    f"{source}: conflicting duplicate for {wav!r}; field {field!r} "
                    f"is {new_value!r}, previously {old_value!r}"
                )
        return False


def validate_current_audio_files(record: Mapping[str, Any], source: str | Path) -> None:
    """Require current cloned/reference files to match the evaluated file signatures."""
    for path_field, signature_field in (
        ("cloned_audio", "cloned_audio_signature"),
        ("ref_audio", "ref_audio_signature"),
    ):
        path = Path(str(record.get(path_field, "")))
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(f"{source}: cannot stat current {path_field} {path}: {exc}") from exc
        current = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if record.get(signature_field) != current:
            raise ValueError(
                f"{source}: current {path_field} no longer matches evaluated signature: {path}"
            )


def validate_current_model_files(record: Mapping[str, Any], source: str | Path) -> None:
    """Require current model files to match the collection's evaluated SHA256 signature."""
    model_dir = Path(str(record.get("model_dir", "")))
    current = {}
    for name in ("config.yaml", "avg_model.pt"):
        path = model_dir / name
        try:
            stat = path.stat()
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"{source}: cannot fingerprint current model file {path}: {exc}") from exc
        current[name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    if record.get("model_signature") != current:
        raise ValueError(f"{source}: current model files no longer match evaluated signature")
