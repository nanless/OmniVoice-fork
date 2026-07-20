#!/usr/bin/env python3
"""Fix paths in deterministic CER sidecars and ASR cache after moving files."""

import json
import os
from pathlib import Path

OLD_PREFIX = "/root/code/github_repos/OmniVoice-fork/batch_cloned_voices"
NEW_PREFIX = "/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"

TARGET_DIR = Path(NEW_PREFIX)


def fix_eval_json_files():
    """Fix wav_path in all .eval.json files."""
    count = 0
    fixed = 0
    for root, dirs, files in os.walk(TARGET_DIR):
        for name in files:
            if not name.endswith(".eval.json"):
                continue
            json_path = Path(root) / name
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            count += 1
            changed = False
            for key in ("wav_path",):
                val = data.get(key)
                if val and isinstance(val, str) and val.startswith(OLD_PREFIX):
                    data[key] = val.replace(OLD_PREFIX, NEW_PREFIX, 1)
                    changed = True
            if changed:
                json_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                fixed += 1
            if count % 10000 == 0:
                print(f"  [eval.json] scanned {count}, fixed {fixed}")
    print(f"[eval.json] done: scanned {count}, fixed {fixed}")


def fix_sidecar_json_files():
    """Fix cloned_audio in text_*.json sidecar files and sim.json files."""
    count = 0
    fixed = 0
    SIDEcar_SUFFIXES = (".eval.json", ".mos.json")
    for root, dirs, files in os.walk(TARGET_DIR):
        for name in files:
            if not name.startswith("text_") or not name.endswith(".json"):
                continue
            if any(name.endswith(s) for s in SIDEcar_SUFFIXES):
                continue
            json_path = Path(root) / name
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            count += 1
            changed = False
            for key in ("cloned_audio", "wav_path"):
                val = data.get(key)
                if val and isinstance(val, str) and val.startswith(OLD_PREFIX):
                    data[key] = val.replace(OLD_PREFIX, NEW_PREFIX, 1)
                    changed = True
            if changed:
                json_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                fixed += 1
            if count % 10000 == 0:
                print(f"  [sidecar] scanned {count}, fixed {fixed}")
    print(f"[sidecar] done: scanned {count}, fixed {fixed}")


def fix_cache_keys(cache_path: Path):
    """Fix keys in an ASR cache JSON file."""
    if not cache_path.exists():
        print(f"[cache] {cache_path} not found, skipping")
        return
    print(f"[cache] loading {cache_path} ...")
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    new_data = {}
    fixed = 0
    for k, v in data.items():
        if k.startswith(OLD_PREFIX):
            new_key = k.replace(OLD_PREFIX, NEW_PREFIX, 1)
            new_data[new_key] = v
            fixed += 1
        else:
            new_data[k] = v
    if fixed:
        cache_path.write_text(
            json.dumps(new_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[cache] {cache_path.name}: fixed {fixed}/{len(data)} keys")
    else:
        print(f"[cache] {cache_path.name}: no keys to fix")


def main():
    print("=== Fixing eval.json wav_path ===")
    fix_eval_json_files()

    print("\n=== Fixing sidecar json cloned_audio ===")
    fix_sidecar_json_files()

    print("\n=== Fixing ASR cache keys ===")
    fix_cache_keys(TARGET_DIR / "eval_asr_cache.json")


    print("\nDone!")


if __name__ == "__main__":
    main()
