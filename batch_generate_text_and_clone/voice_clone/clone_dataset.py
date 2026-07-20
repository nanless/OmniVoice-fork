#!/usr/bin/env python3
"""Clone dataset voices using 10 random texts per audio with random speed.
Supports multi-GPU multi-worker via --worker-id / --num-workers.

Usage:
    # Single worker
    python clone_dataset.py --gpu 0 --worker-id 0 --num-workers 1

    # Worker 3 of 8 (GPU 0, worker 3)
    python clone_dataset.py --gpu 0 --worker-id 3 --num-workers 8
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager, nullcontext
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import soundfile as sf

from speaker_topup_common import fraction_from_record

OUTPUT_SR = 16000  # Target output sampling rate

# ── config ──────────────────────────────────────────────────────────
DATASETS = [
    "/root/group-shared/voiceprint/data/speech/speaker_verification/BAAI-ChildMandarin41.25H_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_by_groundtruth_onlyenhanced",
    "/root/group-shared/voiceprint/data/speech/speaker_verification/speechocean762_integrated_by_groundtruth_onlyenhanced",
]

TEXTS_PATH = "/root/code/github_repos/OmniVoice-fork/batch_generated_text/llm_children_100k_asr_complete.jsonl"
MODEL_PATH = "/root/code/github_repos/OmniVoice-fork/exp/children_finetune_20260519_1418/checkpoints/checkpoint-62000"
OUT_ROOT = Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned")

GEN_CONFIG_KWARGS = {
    "num_step": 32,
    "guidance_scale": 2.0,
    "class_temperature": 0.1,
    "denoise": False,
    "preprocess_prompt": True,
    "postprocess_output": True,
}

TEXTS_PER_AUDIO = 10
SPEED_MIN, SPEED_MAX = 0.85, 1.15
SEED = 42

# ── helpers ─────────────────────────────────────────────────────────
def read_kaldi_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                m[parts[0]] = parts[1]
    return m


def load_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append({
                "id": item.get("id") or item.get("text_id") or f"text_{line_no:06d}",
                "text": item.get("text", ""),
                "language": item.get("language", "zh"),
                "lang_type": item.get("lang_type"),
                "length_type": item.get("length_type"),
                "scenario": item.get("scenario") or item.get("scenario_key"),
                "subscene": item.get("subscene"),
                "emotion": item.get("emotion"),
                "age_tier": item.get("age_tier"),
                "text_tn": item.get("text_tn"),
                "task_id": item.get("task_id"),
            })
    return texts


def sample_texts_for_audio(all_texts, utt_id, n=TEXTS_PER_AUDIO):
    """Deterministically pick diverse texts for one reference audio."""
    if len(all_texts) <= n:
        return list(all_texts)

    rng = random.Random(f"{SEED}:{utt_id}")
    pool = list(all_texts)
    rng.shuffle(pool)

    selected = []
    seen = {
        "lang_type": set(),
        "length_type": set(),
        "scenario": set(),
        "age_tier": set(),
    }

    while pool and len(selected) < n:
        def diversity_score(idx):
            item = pool[idx]
            score = 0
            for key, values in seen.items():
                value = item.get(key)
                if value and value not in values:
                    score += 1
            return (score, -idx)

        best_idx = max(range(len(pool)), key=diversity_score)
        item = pool.pop(best_idx)
        selected.append(item)
        for key, values in seen.items():
            value = item.get(key)
            if value:
                values.add(value)

    return selected


SUPPORTED_LANGS = {"zh", "en", "ja", "de", "fr", "es", "ko", "ar", "ru", "pt", "it"}


def resolve_language(raw_lang):
    """Map raw language field to a valid OmniVoice language code, defaulting to 'zh'."""
    lang_map = {"en_mostly": "en", "frequent_mix": "zh", "cn_mostly": "zh"}
    ov_lang = lang_map.get(raw_lang, raw_lang)
    if ov_lang not in SUPPORTED_LANGS:
        return "zh"
    return ov_lang


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_signature(model_path: Path) -> dict[str, dict[str, int]]:
    signature = {}
    for name in ("config.json", "model.safetensors"):
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")
        signature[name] = file_signature(path)
    return signature


def write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def output_state(out_wav: Path, out_json: Path, expected: dict[str, Any]) -> str:
    """Return complete, failed, partial, corrupt, or missing for one output slot."""
    wav_exists = out_wav.is_file()
    json_exists = out_json.is_file()
    if not wav_exists and not json_exists:
        return "missing"
    if not json_exists:
        return "partial"
    try:
        record = json.loads(out_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "corrupt"
    if record.get("status") == "failed":
        if record.get("schema_version") != expected["schema_version"]:
            return "corrupt"
        return (
            "failed"
            if all(record.get(field) == value for field, value in expected.items())
            else "corrupt"
        )
    if record.get("status") != "generated" or not wav_exists:
        return "partial"
    if record.get("schema_version") == expected["schema_version"]:
        for field, value in expected.items():
            if record.get(field) != value:
                return "corrupt"
    elif expected["schema_version"] == 2 and record.get("schema_version") in (None, 1):
        # Upgrade sidecars written by the only pre-v2 writer without regenerating
        # valid audio. Identity fields that existed in v1 must still match.
        for field in ("ref_audio", "text_id", "gen_text", "model"):
            if record.get(field) != expected[field]:
                return "corrupt"
    else:
        return "corrupt"
    if record.get("cloned_audio") != str(out_wav):
        return "corrupt"
    try:
        info = sf.info(str(out_wav))
        if info.samplerate != OUTPUT_SR or info.frames <= 0:
            return "corrupt"
        if (
            record.get("schema_version") == expected["schema_version"]
            and record.get("cloned_audio_signature") != file_signature(out_wav)
        ):
            return "corrupt"
    except (OSError, RuntimeError):
        return "corrupt"
    return "complete" if record.get("schema_version") == expected["schema_version"] else "upgrade"


def wav_duration(path: Path) -> Fraction:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Invalid WAV duration: {path}")
    return Fraction(info.frames, info.samplerate)


def speaker_worker(speaker_key: str, num_workers: int) -> int:
    digest = hashlib.sha256(speaker_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_workers


def safe_output_path(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe plan output path: {relative_value!r}")
    result = root / relative
    try:
        result.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Plan output escapes output root: {relative_value!r}") from exc
    return result


def load_clone_plan(
    path: Path, worker_id: int, num_workers: int
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Validate the complete plan while retaining only this worker's tasks.

    Plans can be hundreds of megabytes.  Every worker still streams and
    validates every record (including global duplicate IDs/output paths), but
    keeping only the assigned speakers avoids multiplying the full decoded
    JSON object graph by the number of GPU workers.
    """
    meta = None
    tasks = []
    task_ids = set()
    output_path_digests = set()
    total_tasks = 0
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if record.get("record_type") == "plan_meta":
                if meta is not None or tasks:
                    raise ValueError(f"{path}:{line_no}: plan_meta must be the first record")
                meta = record
                continue
            if record.get("record_type") != "clone_task":
                raise ValueError(f"{path}:{line_no}: unknown record_type")
            if meta is None:
                raise ValueError(f"{path}:{line_no}: task before plan_meta")
            for field in (
                "plan_id", "round_id", "task_id", "speaker_key", "ref_audio", "ref_text",
                "text_id", "gen_text", "language", "expected_output_relpath",
                "baseline_duration", "target_duration",
            ):
                if field not in record:
                    raise ValueError(f"{path}:{line_no}: missing {field}")
            if record["plan_id"] != meta.get("plan_id"):
                raise ValueError(f"{path}:{line_no}: task plan_id mismatch")
            if record["task_id"] in task_ids:
                raise ValueError(f"{path}:{line_no}: duplicate task_id {record['task_id']}")
            output_path_digest = hashlib.sha256(
                record["expected_output_relpath"].encode("utf-8")
            ).digest()
            if output_path_digest in output_path_digests:
                raise ValueError(
                    f"{path}:{line_no}: duplicate output path {record['expected_output_relpath']}"
                )
            fraction_from_record(record["baseline_duration"])
            fraction_from_record(record["target_duration"])
            task_ids.add(record["task_id"])
            output_path_digests.add(output_path_digest)
            total_tasks += 1
            if speaker_worker(record["speaker_key"], num_workers) == worker_id:
                tasks.append(record)
    if meta is None:
        raise ValueError(f"Plan has no plan_meta record: {path}")
    if meta.get("plan_schema_version") != 1:
        raise ValueError(f"Unsupported plan schema: {meta.get('plan_schema_version')!r}")
    if meta.get("task_count") != total_tasks:
        raise ValueError(
            f"Plan task_count mismatch: meta={meta.get('task_count')} actual={total_tasks}"
        )
    return meta, tasks, total_tasks


@contextmanager
def speaker_lock(out_root: Path, speaker_key: str):
    lock_dir = out_root / ".speaker_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{hashlib.sha256(speaker_key.encode()).hexdigest()}.lock"
    with open(lock_path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Speaker is already being processed: {speaker_key}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def clone_one(
    model, gen_config, text, ref_audio, ref_text, out_wav, language="zh", fixed_speed=None
):
    import torch
    import torchaudio.functional as TAF

    speed = fixed_speed if fixed_speed is not None else round(random.uniform(SPEED_MIN, SPEED_MAX), 2)
    tmp_wav = out_wav.with_name(f".{out_wav.stem}.{os.getpid()}.tmp.wav")
    tmp_wav.unlink(missing_ok=True)
    try:
        audio = model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text or None,
            language=language,
            generation_config=gen_config,
            speed=speed,
        )
        if not audio or len(audio[0]) == 0:
            raise ValueError("Model returned empty audio")
        # Resample from model SR (24k) to target 16k using high-quality sinc interpolation
        if model.sampling_rate != OUTPUT_SR:
            audio_tensor = torch.from_numpy(audio[0]).float().unsqueeze(0)
            audio_16k = TAF.resample(audio_tensor, model.sampling_rate, OUTPUT_SR).squeeze(0).numpy()
        else:
            audio_16k = audio[0]
        sf.write(str(tmp_wav), audio_16k, OUTPUT_SR, format="WAV")
        info = sf.info(str(tmp_wav))
        if info.samplerate != OUTPUT_SR or info.frames <= 0:
            raise ValueError(f"Invalid generated WAV: samplerate={info.samplerate}, frames={info.frames}")
        with open(tmp_wav, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_wav, out_wav)
        fsync_directory(out_wav.parent)
        return True, speed, None
    except Exception as e:
        print(f"      ERROR: {e}")
        return False, speed, f"{type(e).__name__}: {e}"
    finally:
        tmp_wav.unlink(missing_ok=True)


def plan_expected(task, model_path, model_sig):
    return {
        "schema_version": 3,
        "plan_id": task["plan_id"],
        "round_id": task["round_id"],
        "task_id": task["task_id"],
        "speaker_key": task["speaker_key"],
        "utt_id": task["utt_id"],
        "ref_audio": task["ref_audio"],
        "ref_audio_signature": task["ref_audio_signature"],
        "text_id": task["text_id"],
        "gen_text": task["gen_text"],
        "model": str(model_path),
        "model_signature": model_sig,
        "generation_config": GEN_CONFIG_KWARGS,
    }


def run_plan(args, meta, tasks, model, gen_config, model_sig):
    out_root = args.out_dir
    grouped = {}
    for task in tasks:
        if speaker_worker(task["speaker_key"], args.num_workers) == args.worker_id:
            grouped.setdefault(task["speaker_key"], []).append(task)
    print(
        f"Plan {meta['plan_id']}: worker owns {len(grouped)} speakers / "
        f"{sum(len(rows) for rows in grouped.values())} tasks"
    )
    total_ok = total_skip = total_fail = unmet = 0
    for speaker_key in sorted(grouped):
        speaker_tasks = sorted(grouped[speaker_key], key=lambda row: row["task_ordinal"])
        baselines = {fraction_from_record(row["baseline_duration"]) for row in speaker_tasks}
        targets = {fraction_from_record(row["target_duration"]) for row in speaker_tasks}
        if len(baselines) != 1 or len(targets) != 1:
            raise ValueError(f"Inconsistent quota fields for {speaker_key}")
        baseline = baselines.pop()
        target = targets.pop()
        lock_context = nullcontext() if args.dry_run else speaker_lock(out_root, speaker_key)
        with lock_context:
            prepared = []
            credited = baseline
            for task in speaker_tasks:
                ref_path = Path(task["ref_audio"])
                if file_signature(ref_path) != task["ref_audio_signature"]:
                    raise ValueError(f"Reference changed since planning: {ref_path}")
                out_wav = safe_output_path(out_root, task["expected_output_relpath"])
                out_json = out_wav.with_suffix(".json")
                expected = plan_expected(task, args.model_path, model_sig)
                state = output_state(out_wav, out_json, expected)
                prepared.append((task, out_wav, out_json, expected, state))
                if state == "complete":
                    credited += wav_duration(out_wav)
            if args.dry_run:
                pending = sum(state != "complete" for *_, state in prepared)
                print(
                    f"  {speaker_key}: credited={float(credited):.3f}s "
                    f"target={float(target):.3f}s pending_tasks={pending}"
                )
                continue
            for task, out_wav, out_json, expected, state in prepared:
                if credited >= target:
                    break
                if state == "complete":
                    total_skip += 1
                    continue
                previous_attempts = 0
                if out_json.is_file():
                    try:
                        previous = json.loads(out_json.read_text(encoding="utf-8"))
                        if all(previous.get(field) == value for field, value in expected.items()):
                            previous_attempts = int(previous.get("attempt_count", 0))
                    except (json.JSONDecodeError, OSError, TypeError, ValueError):
                        pass
                if previous_attempts >= args.max_attempts:
                    total_fail += 1
                    continue
                attempt_count = previous_attempts + 1
                out_wav.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(out_json, {
                    **expected, "status": "generating", "cloned_audio": None,
                    "attempt_count": attempt_count, "started_at": datetime.now().isoformat(),
                })
                fixed_speed = round(
                    random.Random(f"{SEED}:speed:{task['task_id']}").uniform(SPEED_MIN, SPEED_MAX), 2
                )
                ok, speed, error = clone_one(
                    model, gen_config, task["gen_text"], task["ref_audio"], task["ref_text"],
                    out_wav, resolve_language(task["language"]), fixed_speed,
                )
                duration = wav_duration(out_wav) if ok else None
                write_json_atomic(out_json, {
                    **expected,
                    "status": "generated" if ok else "failed",
                    "ref_text": task["ref_text"],
                    "gen_text_tn": task.get("gen_text_tn"),
                    "cloned_audio": str(out_wav) if ok else None,
                    "cloned_audio_signature": file_signature(out_wav) if ok else None,
                    "duration_seconds": float(duration) if duration is not None else None,
                    "speed": speed,
                    "language": task["language"],
                    "lang_type": task.get("lang_type"),
                    "length_type": task.get("length_type"),
                    "scenario": task.get("scenario"),
                    "subscene": task.get("subscene"),
                    "emotion": task.get("emotion"),
                    "age_tier": task.get("age_tier"),
                    "source_text_task_id": task.get("source_text_task_id"),
                    "attempt_count": attempt_count,
                    "error": error,
                    "model_sr": model.sampling_rate,
                    "generated_at": datetime.now().isoformat() if ok else None,
                    "failed_at": datetime.now().isoformat() if not ok else None,
                })
                if ok:
                    credited += duration
                    total_ok += 1
                else:
                    total_fail += 1
            if credited < target:
                unmet += 1
                print(
                    f"  UNMET {speaker_key}: {float(credited):.3f}/{float(target):.3f}s; "
                    "plan capacity exhausted"
                )
            else:
                print(f"  MET {speaker_key}: {float(credited):.3f}/{float(target):.3f}s")
    if args.dry_run:
        return 0
    print(
        f"Plan worker done: ok={total_ok} skip={total_skip} failed_tasks={total_fail} "
        f"unmet_speakers={unmet}"
    )
    return 1 if unmet else 0


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--texts-per-audio",
        type=int,
        default=TEXTS_PER_AUDIO,
        help=f"Number of deterministic clone texts per reference audio (default: {TEXTS_PER_AUDIO})",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plan-jsonl", type=Path,
        help="Immutable speaker top-up plan; switches from dataset scan to speaker sharding",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(os.environ.get("CLONED_VOICES_ROOT", str(OUT_ROOT))),
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path(os.environ.get("OMNIVOICE_MODEL_PATH", MODEL_PATH)),
    )
    parser.add_argument(
        "--dataset", type=Path, action="append", dest="datasets",
        help="Dataset root; repeat to override the built-in dataset list",
    )
    parser.add_argument(
        "--texts-path",
        default=TEXTS_PATH,
        help="JSONL path with generated clone texts (default: children 100k)",
    )
    args = parser.parse_args()

    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker-id must satisfy 0 <= worker-id < num-workers")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.texts_per_audio <= 0:
        parser.error("--texts-per-audio must be positive")
    if args.plan_jsonl and args.datasets:
        parser.error("--dataset cannot be combined with --plan-jsonl")
    if args.plan_jsonl and args.limit is not None:
        parser.error("--limit cannot be combined with --plan-jsonl")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = "cuda:0"
    out_root = args.out_dir
    model_path = args.model_path

    print(f"Worker {args.worker_id}/{args.num_workers} on GPU {args.gpu}")
    model_sig = model_signature(model_path)
    plan_meta = plan_tasks = None
    if args.plan_jsonl:
        plan_meta, plan_tasks, total_plan_tasks = load_clone_plan(
            args.plan_jsonl, args.worker_id, args.num_workers
        )
        print(
            f"Validated {total_plan_tasks} plan tasks; retained "
            f"{len(plan_tasks)} tasks for worker {args.worker_id}."
        )
    else:
        datasets = args.datasets or [Path(path) for path in DATASETS]
        texts_path = args.texts_path
        all_texts = load_texts(texts_path)
        print(f"Loaded {len(all_texts)} texts from {texts_path}.")

    model = gen_config = None
    if not args.dry_run:
        import torch
        from omnivoice import OmniVoice, OmniVoiceGenerationConfig

        print("Loading model …")
        model = OmniVoice.from_pretrained(str(model_path), device_map=device, dtype=torch.float16)
        gen_config = OmniVoiceGenerationConfig(**GEN_CONFIG_KWARGS)
        print("Model loaded.\n")

    if args.plan_jsonl:
        return run_plan(args, plan_meta, plan_tasks, model, gen_config, model_sig)

    total_ok = total_fail = total_skip = total_upgrade = 0

    state_counts = {
        key: 0 for key in ("complete", "upgrade", "failed", "partial", "corrupt", "missing")
    }

    for ds_path in datasets:
        if not ds_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {ds_path}")
        ds = str(ds_path)
        ds_name = ds_path.name

        kaldi_dir = Path(ds) / "kaldi_files"
        if kaldi_dir.exists():
            wav_map = read_kaldi_map(kaldi_dir / "wav.scp")
            text_map = read_kaldi_map(kaldi_dir / "text")
            all_items = [(utt, wav_map[utt], text_map.get(utt, "")) for utt in wav_map]
        else:
            audio_paths = sorted(
                p for p in Path(ds).rglob("*")
                if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac")
            )
            all_items = [(p.stem, str(p), "") for p in audio_paths]

        # Shuffle then assign by modulo for balanced random distribution
        rng = random.Random(f"{SEED}:shuffle:{ds_name}")
        shuffled = list(all_items)
        rng.shuffle(shuffled)
        items = [it for i, it in enumerate(shuffled) if i % args.num_workers == args.worker_id]

        if args.limit:
            items = items[: args.limit]

        print(f"[{ds_name}] Worker gets {len(items)} audios")
        for i, (utt_id, ref_audio, ref_text) in enumerate(items, 1):
            sampled_texts = sample_texts_for_audio(
                all_texts, utt_id, args.texts_per_audio
            )

            rel = Path(ref_audio).relative_to(ds)
            base_dir = out_root / ds_name / rel.parent / rel.stem

            slot_states = []
            for tidx, text_item in enumerate(sampled_texts, 1):
                out_wav = base_dir / f"text_{tidx:03d}.wav"
                out_json = base_dir / f"text_{tidx:03d}.json"
                expected = {
                    "schema_version": 2,
                    "utt_id": utt_id,
                    "ref_audio": ref_audio,
                    "ref_audio_signature": file_signature(Path(ref_audio)),
                    "text_id": text_item.get("id"),
                    "gen_text": text_item["text"],
                    "model": str(model_path),
                    "model_signature": model_sig,
                    "generation_config": GEN_CONFIG_KWARGS,
                }
                state = output_state(out_wav, out_json, expected)
                slot_states.append((tidx, text_item, out_wav, out_json, expected, state))
                state_counts[state] += 1

            if args.dry_run:
                actions = sum(state not in ("complete", "upgrade") for *_, state in slot_states)
                upgrades = sum(state == "upgrade" for *_, state in slot_states)
                print(
                    f"  {utt_id}: generate={actions} upgrade={upgrades} "
                    f"skip={len(slot_states) - actions - upgrades}"
                )
                continue

            for tidx, text_item, out_wav, out_json, expected, state in slot_states:
                if state == "complete":
                    total_skip += 1
                    continue
                if state == "upgrade":
                    previous = json.loads(out_json.read_text(encoding="utf-8"))
                    upgraded = {
                        **previous,
                        **expected,
                        "status": "generated",
                        "cloned_audio": str(out_wav),
                        "cloned_audio_signature": file_signature(out_wav),
                        "upgraded_at": datetime.now().isoformat(),
                    }
                    write_json_atomic(out_json, upgraded)
                    total_upgrade += 1
                    continue

                # Ensure output directory exists before generation attempt
                out_wav.parent.mkdir(parents=True, exist_ok=True)

                # Map language to valid OmniVoice language code
                ov_lang = resolve_language(text_item["language"])
                previous_attempts = 0
                if out_json.is_file():
                    try:
                        previous = json.loads(out_json.read_text())
                        if (
                            previous.get("schema_version") == 2
                            and all(
                                previous.get(field) == value
                                for field, value in expected.items()
                            )
                        ):
                            previous_attempts = int(previous.get("attempt_count", 0))
                    except (json.JSONDecodeError, OSError, TypeError, ValueError):
                        pass
                attempt_count = previous_attempts + 1
                if previous_attempts >= args.max_attempts:
                    exhausted = {
                        **expected,
                        "status": "failed",
                        "failure_reason": "max_attempts_exhausted",
                        "attempt_count": previous_attempts,
                        "cloned_audio": str(out_wav),
                        "updated_at": datetime.now().isoformat(),
                    }
                    write_json_atomic(out_json, exhausted)
                    print(
                        f"      EXHAUSTED: {out_json} attempts={previous_attempts} "
                        f"max={args.max_attempts}"
                    )
                    total_fail += 1
                    continue
                generating_record = {
                    **expected,
                    "status": "generating",
                    "cloned_audio": None,
                    "attempt_count": attempt_count,
                    "started_at": datetime.now().isoformat(),
                }
                write_json_atomic(out_json, generating_record)

                ok, speed, error = clone_one(
                    model, gen_config, text_item["text"], ref_audio, ref_text, out_wav, ov_lang
                )

                record = {
                        **expected,
                        "status": "generated" if ok else "failed",
                        "utt_id": utt_id,
                        "ref_text": ref_text,
                        "gen_text_tn": text_item.get("text_tn"),
                        "cloned_audio": str(out_wav) if ok else None,
                        "cloned_audio_signature": file_signature(out_wav) if ok else None,
                        "speed": speed,
                        "language": text_item["language"],
                        "lang_type": text_item.get("lang_type"),
                        "length_type": text_item.get("length_type"),
                        "scenario": text_item.get("scenario"),
                        "subscene": text_item.get("subscene"),
                        "emotion": text_item.get("emotion"),
                        "age_tier": text_item.get("age_tier"),
                        "task_id": text_item.get("task_id"),
                        "attempt_count": attempt_count,
                        "error": error,
                        "model_sr": model.sampling_rate,
                        "generated_at": datetime.now().isoformat(),
                        "failed_at": datetime.now().isoformat() if not ok else None,
                    }
                write_json_atomic(out_json, record)

                if ok:
                    total_ok += 1
                else:
                    total_fail += 1

            if i % 10 == 0:
                print(f"  [{i}/{len(items)}] {utt_id} ({total_ok} ok, {total_fail} fail, {total_skip} skip)", flush=True)

    if args.dry_run:
        print(f"\nDry run state counts: {state_counts}")
        return 0

    print(
        f"\nWorker {args.worker_id} done! ok={total_ok} upgraded={total_upgrade} "
        f"fail={total_fail} skip={total_skip}"
    )
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
