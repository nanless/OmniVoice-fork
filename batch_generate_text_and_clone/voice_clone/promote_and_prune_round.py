#!/usr/bin/env python3
"""Publish one completed funnel round, then delete rejected audio safely."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent

import sys

sys.path.insert(0, str(PIPELINE_DIR))
from eval_common import (  # noqa: E402
    iter_clone_records,
    load_inventory_wav_list,
    wav_list_fingerprint,
    write_json,
)
from speaker_topup_common import write_jsonl_atomic  # noqa: E402


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def validate_threshold(value: float, name: str, lower: float, upper: float) -> None:
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


def validate_list_manifest(
    list_path: Path,
    paths: list[Path],
    fingerprint: str,
    *,
    selection_kind: str,
    sim_threshold: float,
    cer_threshold: float | None,
) -> dict[str, Any]:
    list_path = list_path.resolve(strict=True)
    manifest_path = list_path.with_suffix(list_path.suffix + ".manifest.json")
    manifest = load_json(manifest_path)
    expected = {
        "schema_version": 2,
        "stage": "complete",
        "selection_kind": selection_kind,
        "output": str(list_path),
        "output_signature": signature(list_path),
        "matched_count": len(paths),
        "wav_list_fingerprint": fingerprint,
        "similarity_metric": "raw_cosine",
        "similarity_operator": ">",
        "min_raw_cosine_similarity": sim_threshold,
    }
    if cer_threshold is not None:
        expected.update({"cer_operator": "<", "max_cer": cer_threshold})
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(
            f"Selection manifest does not match {list_path}: {', '.join(mismatched)}"
        )
    return manifest


def atomic_copy(source: Path, destination: Path) -> str:
    source_hash = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise FileExistsError(f"Refusing to overwrite different file: {destination}")
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        if sha256_file(temporary) != source_hash:
            raise IOError(f"Copied file hash mismatch: {source} -> {temporary}")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def build_plan(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    round_root = args.round_root.resolve(strict=True)
    inventory_rows = list(
        iter_clone_records(round_root, allow_partial=True)
    )
    inventory = {str(wav.resolve(strict=True)): (wav, meta_path, record)
                 for wav, meta_path, record in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise ValueError("Current round inventory contains duplicate WAV identities")

    sim_paths, sim_fingerprint = load_inventory_wav_list(
        args.sim_pass_list, round_root, inventory
    )
    accepted_paths, accepted_fingerprint = load_inventory_wav_list(
        args.accepted_list, round_root, inventory
    )
    sim_manifest = validate_list_manifest(
        args.sim_pass_list,
        sim_paths,
        sim_fingerprint,
        selection_kind="sim_threshold",
        sim_threshold=args.sim_threshold,
        cer_threshold=None,
    )
    final_manifest = validate_list_manifest(
        args.accepted_list,
        accepted_paths,
        accepted_fingerprint,
        selection_kind="cer_sim_threshold",
        sim_threshold=args.sim_threshold,
        cer_threshold=args.cer_threshold,
    )
    sim_set = {str(path) for path in sim_paths}
    accepted_set = {str(path) for path in accepted_paths}
    if not accepted_set <= sim_set:
        raise ValueError("Final accepted list is not a subset of the SIM-pass list")
    if sim_manifest.get("inventory_count") != len(inventory):
        raise ValueError("SIM manifest does not cover the current generated inventory")
    if final_manifest.get("candidate_list_fingerprint") != sim_fingerprint:
        raise ValueError("Final manifest is not bound to the current SIM-pass list")
    if final_manifest.get("candidate_count") != len(sim_set):
        raise ValueError("Final manifest candidate count does not match SIM-pass list")
    if final_manifest.get("cer_record_count") != len(sim_set):
        raise ValueError("CER did not cover every SIM-pass candidate")

    accepted_root = args.accepted_root.resolve()
    records: list[dict[str, Any]] = []
    round_ids: set[str] = set()
    plan_ids: set[str] = set()
    task_ids: set[str] = set()
    for wav_key in sorted(inventory):
        wav, meta_path, clone = inventory[wav_key]
        if clone.get("schema_version") != 3:
            raise ValueError(f"Promotion requires schema-v3 clone metadata: {meta_path}")
        speaker_key = clone.get("speaker_key")
        task_id = clone.get("task_id")
        round_id = clone.get("round_id")
        plan_id = clone.get("plan_id")
        if not all(isinstance(value, str) and value for value in (
            speaker_key, task_id, round_id, plan_id
        )):
            raise ValueError(f"Missing plan identity in {meta_path}")
        speaker_parts = Path(speaker_key).parts
        if (
            len(speaker_parts) != 2
            or Path(speaker_key).is_absolute()
            or ".." in speaker_parts
        ):
            raise ValueError(f"Unsafe speaker_key in {meta_path}: {speaker_key!r}")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id in round inventory: {task_id}")
        task_ids.add(task_id)
        round_ids.add(round_id)
        plan_ids.add(plan_id)
        action = "accept" if wav_key in accepted_set else "reject"
        reject_stage = None
        if action == "reject":
            reject_stage = "cer" if wav_key in sim_set else "sim"
        destination = None
        if action == "accept":
            destination = accepted_root / speaker_key / round_id / f"clone_{task_id}.wav"
        artifacts = {}
        for suffix in (".eval.json", ".sim.json", ".mos.json"):
            path = wav.with_suffix(suffix)
            if path.is_file():
                artifacts[suffix] = {
                    "path": str(path), "signature": signature(path)
                }
        records.append({
            "record_type": "round_publish_item",
            "action": action,
            "reject_stage": reject_stage,
            "source_wav": wav_key,
            "source_wav_signature": signature(wav),
            "source_metadata": str(meta_path.resolve(strict=True)),
            "source_metadata_signature": signature(meta_path),
            "speaker_key": speaker_key,
            "task_id": task_id,
            "round_id": round_id,
            "plan_id": plan_id,
            "accepted_wav": str(destination) if destination is not None else None,
            "metric_artifacts": artifacts,
        })
    if len(round_ids) != 1 or len(plan_ids) != 1:
        raise ValueError(
            f"Round inventory mixes identities: round_ids={round_ids}, plan_ids={plan_ids}"
        )
    meta = {
        "record_type": "round_publish_plan",
        "schema_version": 1,
        "stage": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_root": str(round_root),
        "accepted_root": str(accepted_root),
        "round_id": next(iter(round_ids)),
        "plan_id": next(iter(plan_ids)),
        "sim_threshold": args.sim_threshold,
        "sim_operator": ">",
        "cer_threshold": args.cer_threshold,
        "cer_operator": "<",
        "sim_pass_list": str(args.sim_pass_list.resolve(strict=True)),
        "sim_pass_fingerprint": sim_fingerprint,
        "accepted_list": str(args.accepted_list.resolve(strict=True)),
        "accepted_list_fingerprint": accepted_fingerprint,
        "inventory_count": len(inventory),
        "sim_pass_count": len(sim_set),
        "accepted_count": len(accepted_set),
        "rejected_count": len(inventory) - len(accepted_set),
    }
    return meta, records


def load_plan(path: Path) -> tuple[dict, list[dict]]:
    meta = None
    records = []
    with open(path, encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            value = json.loads(line)
            if line_no == 1:
                meta = value
            else:
                records.append(value)
    if not isinstance(meta, dict) or meta.get("record_type") != "round_publish_plan":
        raise ValueError(f"Invalid publish plan: {path}")
    if len(records) != meta.get("inventory_count"):
        raise ValueError(f"Publish plan item count mismatch: {path}")
    return meta, records


def accepted_commit_payload(meta: dict, completed: int) -> dict[str, Any]:
    return {
        **meta,
        "stage": "accepted_committed",
        "accepted_committed": completed,
        "committed_at": datetime.now(timezone.utc).isoformat(),
    }


def promote(
    meta: dict,
    records: list[dict],
    commit_path: Path,
    accepted_commit_path: Path,
) -> None:
    completed = 0
    for row in records:
        if row["action"] != "accept":
            continue
        source = Path(row["source_wav"])
        if not source.is_file() or signature(source) != row["source_wav_signature"]:
            raise ValueError(f"Accepted source changed before promotion: {source}")
        destination = Path(row["accepted_wav"])
        content_hash = atomic_copy(source, destination)
        acceptance = {
            "schema_version": 1,
            "status": "accepted",
            "speaker_key": row["speaker_key"],
            "round_id": row["round_id"],
            "plan_id": row["plan_id"],
            "task_id": row["task_id"],
            "source_wav": str(source),
            "source_signature": row["source_wav_signature"],
            "accepted_wav": str(destination),
            "accepted_signature": signature(destination),
            "content_sha256": content_hash,
            "selection_fingerprint": meta["accepted_list_fingerprint"],
            "sim_operator": ">",
            "sim_threshold": meta["sim_threshold"],
            "cer_operator": "<",
            "cer_threshold": meta["cer_threshold"],
            "accepted_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(destination.with_suffix(".accepted.json"), acceptance)
        completed += 1
        if completed % 500 == 0:
            print(f"[promote] {completed}/{meta['accepted_count']}", flush=True)
    validate_committed_accepts(meta, records)
    commit = accepted_commit_payload(meta, completed)
    # The accepted-root commit is written last. Target accounting ignores all
    # copied WAVs until this marker exists and validates the complete round.
    write_json(commit_path, commit)
    write_json(accepted_commit_path, commit)


def validate_committed_accepts(meta: dict, records: list[dict]) -> None:
    checked = 0
    for row in records:
        if row["action"] != "accept":
            continue
        destination = Path(row["accepted_wav"])
        acceptance_path = destination.with_suffix(".accepted.json")
        if not destination.is_file() or not acceptance_path.is_file():
            raise FileNotFoundError(f"Committed accepted artifact is missing: {destination}")
        acceptance = load_json(acceptance_path)
        if (
            acceptance.get("schema_version") != 1
            or acceptance.get("status") != "accepted"
            or acceptance.get("task_id") != row["task_id"]
            or acceptance.get("speaker_key") != row["speaker_key"]
            or acceptance.get("round_id") != row["round_id"]
            or acceptance.get("plan_id") != row["plan_id"]
            or acceptance.get("accepted_wav") != str(destination)
            or acceptance.get("selection_fingerprint")
            != meta["accepted_list_fingerprint"]
            or acceptance.get("accepted_signature") != signature(destination)
            or acceptance.get("content_sha256") != sha256_file(destination)
        ):
            raise ValueError(f"Committed accepted artifact is stale: {destination}")
        checked += 1
    if checked != meta["accepted_count"]:
        raise ValueError("Committed accepted artifact count mismatch")


def validate_commit(path: Path, meta: dict) -> None:
    commit = load_json(path)
    expected = {
        "schema_version": 1,
        "stage": "accepted_committed",
        "round_id": meta["round_id"],
        "plan_id": meta["plan_id"],
        "round_root": meta["round_root"],
        "accepted_root": meta["accepted_root"],
        "accepted_list_fingerprint": meta["accepted_list_fingerprint"],
        "accepted_count": meta["accepted_count"],
        "accepted_committed": meta["accepted_count"],
    }
    mismatched = [key for key, value in expected.items() if commit.get(key) != value]
    if mismatched:
        raise ValueError(f"Invalid accepted commit {path}: {mismatched}")


def validate_reject_metadata(meta: dict, row: dict) -> dict[str, Any]:
    meta_path = Path(row["source_metadata"])
    clone = load_json(meta_path)
    if clone.get("schema_version") != 3 or clone.get("task_id") != row["task_id"]:
        raise ValueError(f"Clone metadata identity changed before rejection: {meta_path}")
    if clone.get("status") == "rejected":
        if (
            clone.get("deleted_audio_signature") != row["source_wav_signature"]
            or clone.get("selection_fingerprint")
            != meta["accepted_list_fingerprint"]
        ):
            raise ValueError(f"Stale rejected clone metadata: {meta_path}")
        return clone
    expected_signature = row.get("source_metadata_signature")
    if expected_signature is not None and signature(meta_path) != expected_signature:
        raise ValueError(f"Clone metadata changed before rejection: {meta_path}")
    if clone.get("status") != "generated":
        raise ValueError(f"Clone metadata is not generated/rejected: {meta_path}")
    return clone


def prune(meta: dict, records: list[dict], result_path: Path) -> None:
    # Validate every retained audit record before the first destructive unlink.
    reject_metadata = {
        row["task_id"]: validate_reject_metadata(meta, row)
        for row in records if row["action"] == "reject"
    }
    deleted_wavs = deleted_metrics = rejected_metadata = already_done = 0
    deleted_bytes = 0
    for row in records:
        if row["action"] != "reject":
            continue
        wav = Path(row["source_wav"])
        meta_path = Path(row["source_metadata"])
        if wav.exists() and signature(wav) != row["source_wav_signature"]:
            raise ValueError(f"Reject WAV changed before deletion: {wav}")
        for artifact in row["metric_artifacts"].values():
            path = Path(artifact["path"])
            if path.exists() and signature(path) != artifact["signature"]:
                raise ValueError(f"Metric sidecar changed before deletion: {path}")
        if wav.exists():
            deleted_bytes += wav.stat().st_size
            wav.unlink()
            deleted_wavs += 1
        else:
            already_done += 1
        for artifact in row["metric_artifacts"].values():
            path = Path(artifact["path"])
            if path.exists():
                deleted_bytes += path.stat().st_size
                path.unlink()
                deleted_metrics += 1
        clone = reject_metadata[row["task_id"]]
        if clone.get("status") != "rejected":
            if clone.get("schema_version") != 3 or clone.get("task_id") != row["task_id"]:
                raise ValueError(f"Clone metadata changed before rejection: {meta_path}")
            write_json(meta_path, {
                **clone,
                "status": "rejected",
                "cloned_audio": None,
                "cloned_audio_signature": None,
                "rejected_stage": row["reject_stage"],
                "deleted_audio_signature": row["source_wav_signature"],
                "selection_fingerprint": meta["accepted_list_fingerprint"],
                "sim_operator": ">",
                "sim_threshold": meta["sim_threshold"],
                "cer_operator": "<",
                "cer_threshold": meta["cer_threshold"],
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            rejected_metadata += 1
        if (deleted_wavs + already_done) % 5000 == 0:
            print(
                f"[prune] {deleted_wavs + already_done}/{meta['rejected_count']}",
                flush=True,
            )
    write_json(result_path, {
        **meta,
        "stage": "complete",
        "deleted_wavs": deleted_wavs,
        "already_deleted_wavs": already_done,
        "deleted_metric_sidecars": deleted_metrics,
        "rejected_metadata_updated": rejected_metadata,
        "deleted_bytes": deleted_bytes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--sim-pass-list", type=Path, required=True)
    parser.add_argument("--accepted-list", type=Path, required=True)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--cer-threshold", type=float, default=0.1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_threshold(args.sim_threshold, "sim-threshold", -1.0, 1.0)
    validate_threshold(args.cer_threshold, "cer-threshold", 0.0, float("inf"))
    return args


def run(args: argparse.Namespace) -> int:
    round_root = args.round_root.resolve(strict=True)
    plan_path = round_root / "promotion_prune.plan.jsonl"
    commit_path = round_root / "promotion.accepted.commit.json"
    result_path = round_root / "promotion_prune.result.json"
    if plan_path.exists():
        meta, records = load_plan(plan_path)
    else:
        meta, records = build_plan(args)
        write_jsonl_atomic(plan_path, [meta, *records])
    accepted_commit_path = (
        args.accepted_root.resolve() / "commits" /
        f"{meta['round_id']}.accepted.json"
    )
    expected_plan = {
        "round_root": str(round_root),
        "accepted_root": str(args.accepted_root.resolve()),
        "sim_pass_list": str(args.sim_pass_list.resolve(strict=True)),
        "accepted_list": str(args.accepted_list.resolve(strict=True)),
        "sim_threshold": args.sim_threshold,
        "cer_threshold": args.cer_threshold,
    }
    mismatched = [key for key, value in expected_plan.items() if meta.get(key) != value]
    if mismatched:
        raise ValueError(f"Existing publish plan configuration mismatch: {mismatched}")
    if result_path.exists():
        result = load_json(result_path)
        if result.get("stage") == "complete":
            result_mismatched = [
                key for key, value in {**expected_plan, "stage": "complete"}.items()
                if result.get(key) != value
            ]
            if result_mismatched:
                raise ValueError(
                    f"Existing publish result configuration mismatch: {result_mismatched}"
                )
            validate_commit(commit_path, meta)
            validate_commit(accepted_commit_path, meta)
            validate_committed_accepts(meta, records)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    print(
        f"round={meta['round_id']} inventory={meta['inventory_count']} "
        f"sim_pass={meta['sim_pass_count']} accepted={meta['accepted_count']} "
        f"rejected={meta['rejected_count']}"
    )
    print(f"plan={plan_path}")
    if not args.execute:
        print("Preview only; rerun with --execute to promote then prune.")
        return 0
    if not commit_path.exists():
        if accepted_commit_path.exists():
            raise ValueError(
                f"Accepted-root commit exists without round commit: {accepted_commit_path}"
            )
        promote(meta, records, commit_path, accepted_commit_path)
    else:
        validate_commit(commit_path, meta)
        validate_committed_accepts(meta, records)
        if accepted_commit_path.exists():
            validate_commit(accepted_commit_path, meta)
        else:
            # Resume a crash between the two commit writes.
            write_json(
                accepted_commit_path,
                accepted_commit_payload(meta, meta["accepted_count"]),
            )
    validate_commit(accepted_commit_path, meta)
    prune(meta, records, result_path)
    print(f"Complete: {result_path}")
    return 0


def main() -> int:
    args = parse_args()
    args.accepted_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.accepted_root.resolve() / ".promotion.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another publisher holds {lock_path}") from exc
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
