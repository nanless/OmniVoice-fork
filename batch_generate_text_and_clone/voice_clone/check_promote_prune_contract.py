#!/usr/bin/env python3
"""Model-free contract check for schema-v3 round promotion and pruning."""

from __future__ import annotations

import json
import tempfile
import wave
from argparse import Namespace
from pathlib import Path

from promote_and_prune_round import (
    build_plan,
    load_json,
    promote,
    prune,
    signature,
)
from speaker_topup_common import inventory_strict_accepted


def write_wav(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def list_manifest(
    root: Path,
    path: Path,
    rows: list[Path],
    *,
    kind: str,
    inventory_count: int,
    candidate_fingerprint: str | None = None,
) -> dict:
    from eval_common import wav_list_fingerprint

    fingerprint = wav_list_fingerprint(rows)
    manifest = {
        "schema_version": 2,
        "stage": "complete",
        "selection_kind": kind,
        "out_dir": str(root),
        "output": str(path),
        "output_signature": signature(path),
        "inventory_count": inventory_count,
        "candidate_count": inventory_count if candidate_fingerprint is None else 2,
        "matched_count": len(rows),
        "wav_list_fingerprint": fingerprint,
        "candidate_list_fingerprint": candidate_fingerprint,
        "cer_record_count": 2 if candidate_fingerprint is not None else 0,
        "sim_record_count": inventory_count,
        "similarity_metric": "raw_cosine",
        "similarity_operator": ">",
        "min_raw_cosine_similarity": 0.8,
        "cer_operator": "<" if kind == "cer_sim_threshold" else None,
        "max_cer": 0.1 if kind == "cer_sim_threshold" else None,
    }
    write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = (base / "round_raw").resolve()
        accepted_root = (base / "accepted").resolve()
        ref = base / "ref.wav"
        write_wav(ref, 160)
        wavs = []
        for index in range(3):
            wav = root / "dataset" / "source" / f"text_plan_{index}.wav"
            write_wav(wav, 160 + index)
            meta = {
                "schema_version": 3,
                "status": "generated",
                "plan_id": "plan",
                "round_id": "round_001",
                "task_id": f"task{index}",
                "speaker_key": "childmandarin/childmandarin_001",
                "gen_text": "测试",
                "text_id": f"text{index}",
                "model": "model",
                "model_signature": {"model": {"size": 1, "mtime_ns": 1}},
                "generation_config": {"temperature": 1},
                "ref_audio": str(ref),
                "ref_audio_signature": signature(ref),
                "cloned_audio": str(wav),
                "cloned_audio_signature": signature(wav),
            }
            write_json(wav.with_suffix(".json"), meta)
            write_json(wav.with_suffix(".sim.json"), {"similarity": 0.9})
            wavs.append(wav)
        sim_list = root / "filtered" / "sim_gt0.8.txt"
        sim_list.parent.mkdir()
        sim_rows = wavs[:2]
        sim_list.write_text("".join(f"{path}\n" for path in sim_rows), encoding="utf-8")
        sim_manifest = list_manifest(
            root, sim_list, sim_rows, kind="sim_threshold", inventory_count=3
        )
        final_list = root / "filtered" / "cer_lt0.1_sim_gt0.8.txt"
        final_rows = wavs[:1]
        final_list.write_text("".join(f"{path}\n" for path in final_rows), encoding="utf-8")
        list_manifest(
            root,
            final_list,
            final_rows,
            kind="cer_sim_threshold",
            inventory_count=3,
            candidate_fingerprint=sim_manifest["wav_list_fingerprint"],
        )
        args = Namespace(
            round_root=root,
            accepted_root=accepted_root,
            sim_pass_list=sim_list,
            accepted_list=final_list,
            sim_threshold=0.8,
            cer_threshold=0.1,
        )
        meta, records = build_plan(args)
        assert meta["accepted_count"] == 1
        assert meta["rejected_count"] == 2
        commit = root / "promotion.accepted.commit.json"
        accepted_commit = accepted_root / "commits" / "round_001.accepted.json"
        result = root / "promotion_prune.result.json"
        promote(meta, records, commit, accepted_commit)
        assert accepted_commit.is_file()
        assert len(list(accepted_root.rglob("*.wav"))) == 1
        target_speakers = {"childmandarin/childmandarin_001": base / "speaker"}
        durations, counts, _ = inventory_strict_accepted(
            [accepted_root], target_speakers, scan_workers=1
        )
        assert counts["childmandarin/childmandarin_001"] == 1
        assert durations["childmandarin/childmandarin_001"] > 0
        prune(meta, records, result)
        assert wavs[0].is_file()
        assert not wavs[1].exists() and not wavs[2].exists()
        assert load_json(wavs[1].with_suffix(".json"))["status"] == "rejected"
        assert load_json(result)["deleted_wavs"] == 2

    print("promotion/prune contract checks passed")


if __name__ == "__main__":
    main()
