#!/usr/bin/env python3
"""Run resumable clone/SIM/CER/promotion rounds until every speaker has 30 minutes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from speaker_topup_common import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "batch_generate_text_and_clone"
VOICE_DIR = PIPELINE_DIR / "voice_clone"


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    accepted_codes: set[int] = {0},
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(command)}", flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode not in accepted_codes:
        raise RuntimeError(
            f"Command exited {result.returncode}; inspect {log_path}: {' '.join(command)}"
        )
    return result.returncode


def update_state(path: Path, state: dict, **updates) -> dict:
    state = {**state, **updates, "updated_at": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(path, state)
    return state


def wait_for_gpus(args) -> None:
    if not args.wait_for_gpus:
        return
    requested = {int(value.strip()) for value in args.gpus.split(",") if value.strip()}
    consecutive = 0
    while consecutive < args.gpu_ready_consecutive:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        readings = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
            index, utilization, memory = map(int, parts)
            readings[index] = (utilization, memory)
        if not requested <= readings.keys():
            raise ValueError(f"Requested GPUs are unavailable: {sorted(requested)}")
        busy = {
            index: readings[index]
            for index in sorted(requested)
            if (
                readings[index][0] > args.gpu_ready_max_util
                or readings[index][1] > args.gpu_ready_max_memory_mib
            )
        }
        if busy:
            consecutive = 0
            print(f"[gpu-wait] busy={busy}", flush=True)
        else:
            consecutive += 1
            print(
                f"[gpu-wait] ready sample "
                f"{consecutive}/{args.gpu_ready_consecutive}",
                flush=True,
            )
        if consecutive < args.gpu_ready_consecutive:
            time.sleep(args.gpu_poll_seconds)


def target_check(args, output: Path, log: Path) -> tuple[bool, dict]:
    # A runtime failure commonly exits 1, which is also the checker's
    # documented "deficit remains" status. Removing the old report ensures a
    # failed run can never be mistaken for a fresh deficit result.
    output.unlink(missing_ok=True)
    command = [
        args.python,
        str(VOICE_DIR / "check_speaker_target.py"),
        "--original-root", str(args.original_root),
        "--accepted-root", str(args.legacy_accepted_root),
        "--strict-accepted-root", str(args.topup_accepted_root),
        "--target-seconds", str(args.target_seconds),
        "--summary-json", str(output),
        "--scan-workers", str(args.scan_workers),
    ]
    code = run_logged(command, log, accepted_codes={0, 1})
    if not output.is_file():
        raise RuntimeError(f"Target checker did not produce a fresh report: {output}")
    summary = json.loads(output.read_text(encoding="utf-8"))
    if summary.get("counting_policy") != "original_plus_accepted_only":
        raise ValueError("Target checker did not use accepted-only counting policy")
    expected = {
        "original_root": str(args.original_root),
        "accepted_roots": [str(args.legacy_accepted_root)],
        "strict_accepted_roots": [str(args.topup_accepted_root)],
        "speaker_count": 1646,
    }
    mismatched = [key for key, value in expected.items() if summary.get(key) != value]
    if mismatched:
        raise ValueError(f"Target checker report configuration mismatch: {mismatched}")
    target = summary.get("target_duration", {})
    if target.get("seconds") != float(args.target_seconds):
        raise ValueError("Target checker report has a different target duration")
    return code == 0, summary


def round_paths(args, number: int) -> dict[str, Path]:
    stem = f"round_{number:03d}"
    raw = args.topup_root / f"{stem}_raw"
    return {
        "plan": args.topup_root / f"{stem}.plan.jsonl",
        "summary": args.topup_root / f"{stem}.plan.summary.json",
        "raw": raw,
        "log_dir": args.topup_root / "loop_logs" / stem,
        "sim_list": raw / "filtered" / f"sim_gt{args.sim_threshold}.txt",
        "final_list": raw / "filtered" / (
            f"cer_lt{args.cer_threshold}_sim_gt{args.sim_threshold}.txt"
        ),
        "publish_result": raw / "promotion_prune.result.json",
        "target_after": args.topup_root / f"{stem}.target_after.json",
    }


def prior_artifacts(args, number: int) -> tuple[list[Path], list[Path]]:
    plans = []
    histories = []
    for prior in range(1, number):
        paths = round_paths(args, prior)
        if paths["plan"].is_file():
            plans.append(paths["plan"])
        sim = paths["raw"] / "eval_sim_details.jsonl"
        if sim.is_file():
            histories.append(sim)
    return plans, histories


def plan_round(args, number: int, paths: dict[str, Path]) -> None:
    summary_csv = paths["summary"].with_suffix(".csv")
    outputs = (paths["plan"], paths["summary"], summary_csv)
    if all(path.is_file() for path in outputs):
        return
    partial_outputs = [path for path in outputs if path.exists()]
    if partial_outputs and paths["raw"].exists() and any(
        paths["raw"].rglob("text_plan_*")
    ):
        raise RuntimeError(
            "Incomplete plan outputs already have raw task artifacts; refusing "
            f"to rebuild: {partial_outputs}"
        )
    plans, histories = prior_artifacts(args, number)
    command = [
        args.python,
        str(VOICE_DIR / "plan_speaker_topup.py"),
        "--original-root", str(args.original_root),
        "--accepted-root", str(args.legacy_accepted_root),
        "--strict-accepted-root", str(args.topup_accepted_root),
        "--target-seconds", str(args.target_seconds),
        "--generation-multiplier", str(args.generation_multiplier),
        "--history-sim-threshold", str(args.sim_threshold),
        "--history-reference-limit", str(args.history_reference_limit),
        "--round-id", f"{args.campaign_id}_r{number:03d}",
        "--plan-jsonl", str(paths["plan"]),
        "--summary-json", str(paths["summary"]),
        "--summary-csv", str(summary_csv),
        "--scan-workers", str(args.scan_workers),
    ]
    for plan in plans:
        command.extend(("--prior-plan-jsonl", str(plan)))
    for history in histories:
        command.extend(("--history-sim-details", str(history)))
    if partial_outputs:
        command.append("--force")
    run_logged(command, paths["log_dir"] / "plan.log")


def clone_round(args, number: int, paths: dict[str, Path]) -> int:
    marker = paths["raw"] / ".clone_loop_finished.json"
    if marker.is_file():
        return int(json.loads(marker.read_text())["exit_code"])
    environment = os.environ.copy()
    environment.update({
        "PYTHON": args.python,
        "GPUS": args.gpus,
        "WORKERS_PER_GPU": str(args.workers_per_gpu),
        "CLONE_PLAN_JSONL": str(paths["plan"]),
        "CLONED_VOICES_ROOT": str(paths["raw"]),
        "RUN_ID": f"{args.campaign_id}_r{number:03d}",
    })
    command = ["bash", str(VOICE_DIR / "run_clone_8workers.sh")]
    code = run_logged(
        command,
        paths["log_dir"] / "clone_launcher.log",
        env=environment,
        accepted_codes={0, 1},
    )
    generated = sum(1 for _ in paths["raw"].rglob("text_plan_*.wav"))
    if generated == 0:
        raise RuntimeError("Clone stage produced no WAVs")
    write_json_atomic(marker, {
        "stage": "complete" if code == 0 else "usable_partial",
        "exit_code": code,
        "generated_wav_count": generated,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return code


def evaluate_round(args, number: int, paths: dict[str, Path]) -> None:
    if paths["final_list"].is_file():
        manifest = paths["final_list"].with_suffix(
            paths["final_list"].suffix + ".manifest.json"
        )
        if manifest.is_file() and json.loads(manifest.read_text()).get("stage") == "complete":
            return
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    environment = os.environ.copy()
    environment.update({
        "IN_TMUX": "1",
        "CLONED_VOICES_ROOT": str(paths["raw"]),
        "SKIP_SIM": "0",
        "SKIP_CER": "0",
        "SKIP_MOS": "1",
        "SKIP_EXISTING": "1",
        "ALLOW_PARTIAL": "1",
        "CER_GPU": gpu_ids[0],
        "SIM_GPUS": args.gpus,
        "EVAL_WORKERS": str(len(gpu_ids)),
        "ASR_BATCH_SIZE": str(args.asr_batch_size),
        "SIM_THRESHOLD": str(args.sim_threshold),
        "CER_THRESHOLD": str(args.cer_threshold),
        "EVAL_RUN_ID": f"{args.campaign_id}_r{number:03d}",
        "PYTHON": args.python,
        "PYTHON_CER": args.python_cer,
    })
    run_logged(
        ["bash", str(PIPELINE_DIR / "run_eval_all.sh")],
        paths["log_dir"] / "eval_launcher.log",
        env=environment,
    )


def publish_round(args, paths: dict[str, Path]) -> None:
    if paths["publish_result"].is_file():
        result = json.loads(paths["publish_result"].read_text())
        if result.get("stage") == "complete":
            return
    command = [
        args.python,
        str(VOICE_DIR / "promote_and_prune_round.py"),
        "--round-root", str(paths["raw"]),
        "--accepted-root", str(args.topup_accepted_root),
        "--sim-pass-list", str(paths["sim_list"]),
        "--accepted-list", str(paths["final_list"]),
        "--sim-threshold", str(args.sim_threshold),
        "--cer-threshold", str(args.cer_threshold),
        "--execute",
    ]
    run_logged(command, paths["log_dir"] / "publish.log")


def parse_args() -> argparse.Namespace:
    default_base = Path(
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
        "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_"
        "withclone_addlibrilight_1130"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topup-root", type=Path, default=default_base / "omnivoice_topup")
    parser.add_argument("--original-root", type=Path, default=default_base / "audio")
    parser.add_argument(
        "--legacy-accepted-root", type=Path,
        default=default_base / "audio_omnivoice_clone_sim0.8_filtered",
    )
    parser.add_argument(
        "--topup-accepted-root", type=Path,
        default=default_base / "audio_omnivoice_topup_sim0.8_cer0.1_accepted",
    )
    parser.add_argument("--campaign-id", default="topup_20260721")
    parser.add_argument("--start-round", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--target-seconds", type=int, default=1800)
    parser.add_argument("--generation-multiplier", default="4")
    parser.add_argument("--history-reference-limit", type=int, default=8)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--cer-threshold", type=float, default=0.1)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--wait-for-gpus", action="store_true")
    parser.add_argument("--gpu-ready-max-util", type=int, default=10)
    parser.add_argument("--gpu-ready-max-memory-mib", type=int, default=1024)
    parser.add_argument("--gpu-ready-consecutive", type=int, default=3)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--python", default="/root/miniforge3/envs/omnivoice/bin/python")
    parser.add_argument("--python-cer", default="/root/miniforge3/envs/qwen3-asr/bin/python")
    args = parser.parse_args()
    if args.start_round < 1 or args.max_rounds < args.start_round:
        parser.error("round bounds are invalid")
    if (
        args.target_seconds <= 0
        or args.workers_per_gpu <= 0
        or args.history_reference_limit <= 0
        or args.gpu_ready_max_util < 0
        or args.gpu_ready_max_memory_mib < 0
        or args.gpu_ready_consecutive <= 0
        or args.gpu_poll_seconds <= 0
    ):
        parser.error("target/workers/GPU wait settings are invalid")
    return args


def main() -> int:
    args = parse_args()
    args.topup_root = args.topup_root.resolve()
    args.original_root = args.original_root.resolve(strict=True)
    args.legacy_accepted_root = args.legacy_accepted_root.resolve(strict=True)
    args.topup_accepted_root.mkdir(parents=True, exist_ok=True)
    args.topup_accepted_root = args.topup_accepted_root.resolve(strict=True)
    args.topup_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.topup_root / ".topup_loop.lock"
    state_path = args.topup_root / "loop_state.json"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another top-up loop holds {lock_path}") from exc
        configuration = {
            "schema_version": 1,
            "campaign_id": args.campaign_id,
            "counting_policy": "original_plus_accepted_only",
            "topup_root": str(args.topup_root),
            "original_root": str(args.original_root),
            "legacy_accepted_root": str(args.legacy_accepted_root),
            "topup_accepted_root": str(args.topup_accepted_root),
            "target_seconds": args.target_seconds,
            "sim_operator": ">",
            "sim_threshold": args.sim_threshold,
            "cer_operator": "<",
            "cer_threshold": args.cer_threshold,
            "generation_multiplier": args.generation_multiplier,
            "gpus": args.gpus,
            "workers_per_gpu": args.workers_per_gpu,
        }
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            immutable_keys = (
                "schema_version", "campaign_id", "counting_policy", "topup_root",
                "original_root", "legacy_accepted_root", "topup_accepted_root",
                "target_seconds", "sim_operator", "sim_threshold", "cer_operator",
                "cer_threshold", "generation_multiplier",
            )
            drift = [
                key for key in immutable_keys
                if state.get(key) != configuration.get(key)
            ]
            if drift:
                raise ValueError(f"Existing loop state configuration drift: {drift}")
            state = {**state, **configuration}
        else:
            state = configuration
        initial_path = args.topup_root / "target_before_loop.json"
        complete, before = target_check(
            args, initial_path, args.topup_root / "loop_logs" / "target_before.log"
        )
        state = update_state(
            state_path,
            state,
            stage="COMPLETE" if complete else "READY",
            current_deficit_seconds=before["total_deficit_duration"]["seconds"],
        )
        if complete:
            return 0
        previous_deficit = before["total_deficit_duration"]["seconds"]
        for number in range(args.start_round, args.max_rounds + 1):
            paths = round_paths(args, number)
            if paths["publish_result"].is_file():
                published = json.loads(paths["publish_result"].read_text())
                if published.get("stage") == "complete":
                    complete, after = target_check(
                        args,
                        paths["target_after"],
                        paths["log_dir"] / "target_after.log",
                    )
                    previous_deficit = after["total_deficit_duration"]["seconds"]
                    state = update_state(
                        state_path,
                        state,
                        stage="COMPLETE" if complete else "ROUND_ALREADY_COMPLETE",
                        round=number,
                        current_deficit_seconds=previous_deficit,
                    )
                    if complete:
                        return 0
                    continue
            state = update_state(state_path, state, stage="PLANNING", round=number)
            plan_round(args, number, paths)
            if not (paths["raw"] / ".clone_loop_finished.json").is_file():
                state = update_state(state_path, state, stage="WAITING_FOR_GPUS")
                wait_for_gpus(args)
            state = update_state(state_path, state, stage="CLONING")
            clone_code = clone_round(args, number, paths)
            state = update_state(
                state_path, state, stage="EVALUATING", clone_exit_code=clone_code
            )
            evaluate_round(args, number, paths)
            state = update_state(state_path, state, stage="PROMOTING")
            publish_round(args, paths)
            state = update_state(state_path, state, stage="VERIFYING")
            complete, after = target_check(
                args, paths["target_after"], paths["log_dir"] / "target_after.log"
            )
            current_deficit = after["total_deficit_duration"]["seconds"]
            plan_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            round_baseline_deficit = plan_summary.get("total_deficit_seconds")
            if not isinstance(round_baseline_deficit, (int, float)):
                raise ValueError(f"Plan summary has no baseline deficit: {paths['summary']}")
            if current_deficit >= round_baseline_deficit and not complete:
                update_state(
                    state_path,
                    state,
                    stage="BLOCKED_NO_PROGRESS",
                    current_deficit_seconds=current_deficit,
                )
                raise RuntimeError(
                    "Round produced no accepted-duration progress relative to its plan baseline"
                )
            state = update_state(
                state_path,
                state,
                stage="COMPLETE" if complete else "ROUND_COMPLETE",
                current_deficit_seconds=current_deficit,
            )
            if complete:
                return 0
            previous_deficit = current_deficit
        update_state(state_path, state, stage="MAX_ROUNDS_REACHED")
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
