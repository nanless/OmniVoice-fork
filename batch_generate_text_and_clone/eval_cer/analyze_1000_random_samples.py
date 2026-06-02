#!/usr/bin/env python3
"""Randomly sample 1000 eval records from current run with full paths and full details.

This script:
1. Identifies .eval.json files from the current eval_cloned.py run by checking evaluated_at field
2. Randomly samples 1000 files
3. Outputs complete paths and complete eval.json contents for manual inspection
"""

import json
import multiprocessing as mp
import random
import sys
from datetime import datetime
from pathlib import Path

OUT_DIR = Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned")
RANDOM_SEED = 42
SAMPLE_SIZE = 1000
N_WORKERS = 8


def get_current_run_timestamp():
    """Find the evaluated_at timestamp of the current run."""
    # Check a few files to find the most recent timestamp
    files = list(OUT_DIR.rglob("*.eval.json"))
    timestamps = []
    for f in files[:100]:
        try:
            d = json.loads(f.read_text())
            ts = d.get("evaluated_at", "")
            if ts.startswith("2026-06-02"):
                timestamps.append(ts)
        except:
            pass
    
    if timestamps:
        # Use the most common recent timestamp
        from collections import Counter
        ts_counts = Counter(timestamps)
        current_ts = ts_counts.most_common(1)[0][0]
        print(f"Detected current run timestamp: {current_ts}", file=sys.stderr)
        return current_ts
    return None


def check_file(args):
    """Check if a single file belongs to current run."""
    f, current_ts = args
    try:
        # Quick check: read first 2000 bytes to get evaluated_at
        content = f.read_bytes()[:2000]
        # Find evaluated_at field
        ts_start = content.find(b'"evaluated_at"')
        if ts_start > 0:
            ts_val_start = content.find(b'"', ts_start + 15)
            ts_val_end = content.find(b'"', ts_val_start + 1)
            if ts_val_start > 0 and ts_val_end > ts_val_start:
                file_ts = content[ts_val_start + 1:ts_val_end].decode("utf-8", errors="ignore")
                if file_ts == current_ts:
                    return f
    except:
        pass
    return None


def find_current_run_files(current_ts):
    """Find all .eval.json files from current run."""
    print(f"Scanning for files with evaluated_at={current_ts}...", file=sys.stderr)
    
    all_files = list(OUT_DIR.rglob("*.eval.json"))
    print(f"Total .eval.json files: {len(all_files)}", file=sys.stderr)
    
    # Use multiprocessing to check files
    with mp.Pool(N_WORKERS) as pool:
        args = [(f, current_ts) for f in all_files]
        results = pool.map(check_file, args)
    
    current_run_files = [r for r in results if r is not None]
    print(f"Found {len(current_run_files)} files from current run", file=sys.stderr)
    return current_run_files


def load_and_sample(files, n=1000, seed=42):
    """Randomly sample n files and load their contents."""
    random.seed(seed)
    
    if len(files) < n:
        print(f"Warning: only {len(files)} current run files, requested {n}", file=sys.stderr)
        sampled = files
    else:
        sampled = random.sample(files, n)
    
    records = []
    for f in sampled:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append({"path": str(f), "data": data})
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: failed to load {f}: {e}", file=sys.stderr)
    
    return records


def generate_full_report(records):
    """Generate report with full paths and complete eval.json contents."""
    lines = []
    lines.append("=" * 100)
    lines.append("Complete Eval Results - 1000 Random Samples from Current Run")
    lines.append("=" * 100)
    lines.append(f"Analysis Time: {datetime.now().isoformat()}")
    lines.append(f"Total Samples: {len(records)}")
    lines.append(f"Random Seed: {RANDOM_SEED}")
    lines.append("")
    
    for i, record in enumerate(records, 1):
        lines.append("-" * 100)
        lines.append(f"Sample {i}")
        lines.append("-" * 100)
        lines.append(f"Audio Path: {record['path']}")
        lines.append("")
        lines.append("Complete Eval Record:")
        lines.append(json.dumps(record["data"], ensure_ascii=False, indent=2))
        lines.append("")
    
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def generate_path_list(records):
    """Generate simple path list for quick reference."""
    lines = []
    lines.append("=" * 100)
    lines.append("Audio File Paths - All Samples")
    lines.append("=" * 100)
    for record in records:
        lines.append(record["path"])
    return "\n".join(lines)


def main():
    # Step 1: Get current run timestamp
    current_ts = get_current_run_timestamp()
    if not current_ts:
        print("Error: Could not detect current run timestamp", file=sys.stderr)
        return
    
    # Step 2: Find files from current run
    current_files = find_current_run_files(current_ts)
    
    # Step 3: Random sample and load
    print("Loading sampled records...", file=sys.stderr)
    records = load_and_sample(current_files, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Loaded {len(records)} records", file=sys.stderr)
    
    # Step 4: Generate reports
    print("Generating full report...", file=sys.stderr)
    full_report = generate_full_report(records)
    path_list = generate_path_list(records)
    
    # Save
    base_dir = Path("/root/code/github_repos/OmniVoice-fork/batch_generate_text_and_clone/eval_cer")
    report_path = base_dir / "analysis_1000_random_samples_full.txt"
    path_list_path = base_dir / "analysis_1000_random_samples_paths.txt"
    
    report_path.write_text(full_report, encoding="utf-8")
    path_list_path.write_text(path_list, encoding="utf-8")
    
    print(f"\nFull report saved to: {report_path}", file=sys.stderr)
    print(f"Path list saved to: {path_list_path}", file=sys.stderr)
    print(f"Total current run files: {len(current_files)}", file=sys.stderr)
    
    # Print summary to stdout
    print(f"Analysis complete!")
    print(f"Total current run files: {len(current_files)}")
    print(f"Sampled: {len(records)}")
    print(f"Full report: {report_path}")
    print(f"Path list: {path_list_path}")


if __name__ == "__main__":
    main()
