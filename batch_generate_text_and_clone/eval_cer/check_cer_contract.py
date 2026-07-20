#!/usr/bin/env python3
"""Standalone checks for CER v4 context and ASR provenance binding."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import eval_batch_200 as eb
import eval_cloned


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    # Isolate this contract check from large model files; the behavior being
    # tested is exact provenance equality, not model hashing itself.
    eb.QWEN3_ASR_LOCAL = "/contract-check/qwen3-asr"
    eb.asr_model_fingerprint = lambda: "model-fingerprint"
    eb.asr_decode_fingerprint = lambda: "auto-language-decode-fingerprint"

    with tempfile.TemporaryDirectory(prefix="cer-v4-contract-") as directory:
        root = Path(directory)
        wav = root / "sample.wav"
        clone_json = root / "sample.json"
        wav.write_bytes(b"read-only-contract-audio-placeholder")
        clone = {
            "gen_text": "Well, [confirmation-en] I miss you.",
            "language": "en",
            "lang_type": "pure_en",
        }
        write_json(clone_json, clone)

        item = eb.build_eval_item(wav, clone_json, "Well, uh-huh I miss you.")
        assert item["ref_normalized"] == item["hypo_normalized"]
        record = eval_cloned.build_eval_record(item, datetime.now().isoformat())
        eval_cloned.write_eval_json(clone_json, record)
        assert eval_cloned.read_valid_eval(wav, clone_json) is not None
        assert eval_cloned.read_reusable_asr_hypothesis(wav, clone_json) is not None

        clone["language"] = "zh"
        write_json(clone_json, clone)
        assert eval_cloned.read_valid_eval(wav, clone_json) is None

        clone["language"] = "en"
        write_json(clone_json, clone)
        record["asr_decode_fingerprint"] = "forced-chinese-old-decode"
        eval_cloned.write_eval_json(clone_json, record)
        assert eval_cloned.read_reusable_asr_hypothesis(wav, clone_json) is None

    print("CER v4 contract self-check: context invalidation and ASR provenance passed")


if __name__ == "__main__":
    main()
