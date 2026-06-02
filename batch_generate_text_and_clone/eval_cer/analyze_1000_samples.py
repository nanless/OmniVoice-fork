#!/usr/bin/env python3
"""Analyze 1000 samples from the current eval run.

This script:
1. Identifies newly created .eval.json files (mtime after the run started)
2. Selects 1000 samples
3. Analyzes CER distribution, LLM vs Manual improvement, error patterns
"""

import argparse
import json
import os
import random
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def find_new_eval_files(out_dir: Path, run_start_time: datetime, max_samples: int = 1000) -> list[Path]:
    """Find .eval.json files created after run_start_time."""
    print(f"Scanning for .eval.json files newer than {run_start_time}...")
    
    # Get all .eval.json files with their mtime
    all_files = []
    for f in out_dir.rglob("*.eval.json"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime >= run_start_time:
            all_files.append((f, mtime))
    
    print(f"Found {len(all_files)} new .eval.json files")
    
    if len(all_files) > max_samples:
        # Sort by mtime and take the earliest max_samples (to avoid taking from ongoing run)
        all_files.sort(key=lambda x: x[1])
        selected = all_files[:max_samples]
    else:
        selected = all_files
    
    return [f for f, _ in selected]


def load_eval_records(files: list[Path]) -> list[dict]:
    """Load eval records from .eval.json files."""
    records = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: failed to load {f}: {e}")
    return records


def analyze_records(records: list[dict]) -> dict:
    """Analyze eval records and return statistics."""
    
    n = len(records)
    
    # CER stats
    manual_cers = [r["manual_cer"] for r in records if "manual_cer" in r]
    llm_cers = [r["llm_cer"] for r in records if "llm_cer" in r]
    
    # Improvement stats
    improvements = []
    llm_worse = []
    no_change = []
    for r in records:
        if "manual_cer" in r and "llm_cer" in r:
            diff = r["manual_cer"] - r["llm_cer"]
            improvements.append(diff)
            if diff < 0:
                llm_worse.append(r)
            elif diff == 0:
                no_change.append(r)
    
    # CER bins
    def bin_cer(cer):
        if cer == 0:
            return "0%"
        elif cer <= 0.05:
            return "(0%, 5%]"
        elif cer <= 0.10:
            return "(5%, 10%]"
        elif cer <= 0.20:
            return "(10%, 20%]"
        elif cer <= 0.50:
            return "(20%, 50%]"
        else:
            return ">50%"
    
    manual_bins = Counter(bin_cer(c) for c in manual_cers)
    llm_bins = Counter(bin_cer(c) for c in llm_cers)
    
    stats = {
        "total_samples": n,
        "with_llm": len(llm_cers),
        "manual_cer": {
            "mean": statistics.mean(manual_cers) if manual_cers else 0,
            "median": statistics.median(manual_cers) if manual_cers else 0,
            "max": max(manual_cers) if manual_cers else 0,
            "min": min(manual_cers) if manual_cers else 0,
            "q90": np.percentile(manual_cers, 90) if manual_cers else 0,
        },
        "llm_cer": {
            "mean": statistics.mean(llm_cers) if llm_cers else 0,
            "median": statistics.median(llm_cers) if llm_cers else 0,
            "max": max(llm_cers) if llm_cers else 0,
            "min": min(llm_cers) if llm_cers else 0,
            "q90": np.percentile(llm_cers, 90) if llm_cers else 0,
        },
        "improvement": {
            "mean": statistics.mean(improvements) if improvements else 0,
            "median": statistics.median(improvements) if improvements else 0,
            "improved_count": sum(1 for x in improvements if x > 0),
            "worsened_count": sum(1 for x in improvements if x < 0),
            "no_change_count": sum(1 for x in improvements if x == 0),
            "avg_improvement_when_improved": statistics.mean([x for x in improvements if x > 0]) if any(x > 0 for x in improvements) else 0,
        },
        "manual_bins": dict(manual_bins),
        "llm_bins": dict(llm_bins),
    }
    
    return stats


def format_report(stats: dict, sample_records: list[dict]) -> str:
    """Generate a formatted text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Eval CER Analysis Report - 1000 Samples from Current Run")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Analysis Time: {datetime.now().isoformat()}")
    lines.append(f"Total Samples Analyzed: {stats['total_samples']}")
    lines.append(f"Samples with LLM ITN: {stats['with_llm']}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("CER Statistics")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"{'Metric':<20} {'Manual ITN':>15} {'LLM ITN':>15}")
    lines.append(f"{'-'*20} {'-'*15} {'-'*15}")
    for key in ["mean", "median", "min", "max", "q90"]:
        lines.append(f"{key:<20} {stats['manual_cer'][key]*100:>14.2f}% {stats['llm_cer'][key]*100:>14.2f}%")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("CER Distribution")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"{'Bin':<20} {'Manual':>10} {'LLM':>10}")
    lines.append(f"{'-'*20} {'-'*10} {'-'*10}")
    for bin_name in ["0%", "(0%, 5%]", "(5%, 10%]", "(10%, 20%]", "(20%, 50%]", ">50%"]:
        m_count = stats["manual_bins"].get(bin_name, 0)
        l_count = stats["llm_bins"].get(bin_name, 0)
        lines.append(f"{bin_name:<20} {m_count:>10} {l_count:>10}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("LLM ITN Impact")
    lines.append("-" * 80)
    lines.append("")
    imp = stats["improvement"]
    lines.append(f"Improved by LLM:        {imp['improved_count']} ({imp['improved_count']/stats['with_llm']*100:.1f}%)")
    lines.append(f"Worsened by LLM:        {imp['worsened_count']} ({imp['worsened_count']/stats['with_llm']*100:.1f}%)")
    lines.append(f"No change:              {imp['no_change_count']} ({imp['no_change_count']/stats['with_llm']*100:.1f}%)")
    lines.append(f"Mean improvement:       {imp['mean']*100:.3f}%")
    lines.append(f"Median improvement:     {imp['median']*100:.3f}%")
    if imp['improved_count'] > 0:
        lines.append(f"Avg improvement (when improved): {imp['avg_improvement_when_improved']*100:.3f}%")
    lines.append("")
    
    # Show worst cases
    lines.append("-" * 80)
    lines.append("Top 10 Worst Cases (by Manual CER)")
    lines.append("-" * 80)
    lines.append("")
    sorted_by_manual = sorted(sample_records, key=lambda x: x.get("manual_cer", 0), reverse=True)[:10]
    for i, r in enumerate(sorted_by_manual, 1):
        wav_path = r.get("wav_path", "unknown")
        lines.append(f"{i}. {wav_path}")
        lines.append(f"   Manual CER: {r.get('manual_cer', 0)*100:.2f}% | LLM CER: {r.get('llm_cer', 'N/A')}")
        lines.append(f"   Ref:  {r.get('ref_manual', '')[:80]}")
        lines.append(f"   Hyp:  {r.get('hypo_manual', '')[:80]}")
        if "ref_llm" in r:
            lines.append(f"   LLM:  {r.get('ref_llm', '')[:80]}")
        lines.append("")
    
    # Show best improvements
    lines.append("-" * 80)
    lines.append("Top 10 Best Improvements (Manual CER -> LLM CER)")
    lines.append("-" * 80)
    lines.append("")
    with_improvement = [(r, r["manual_cer"] - r["llm_cer"]) for r in sample_records if "llm_cer" in r and "manual_cer" in r]
    with_improvement.sort(key=lambda x: x[1], reverse=True)
    for i, (r, diff) in enumerate(with_improvement[:10], 1):
        wav_path = r.get("wav_path", "unknown")
        lines.append(f"{i}. {wav_path}")
        lines.append(f"   Manual: {r['manual_cer']*100:.2f}% -> LLM: {r['llm_cer']*100:.2f}% (Δ {diff*100:.2f}%)")
        lines.append(f"   Ref:  {r.get('ref_manual', '')[:80]}")
        lines.append(f"   Hyp:  {r.get('hypo_manual', '')[:80]}")
        lines.append("")
    
    # Audio list
    lines.append("-" * 80)
    lines.append(f"Complete Audio File List ({len(sample_records)} files)")
    lines.append("-" * 80)
    lines.append("")
    for r in sample_records:
        wav_path = r.get("wav_path", "unknown")
        lines.append(wav_path)
    lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze 1000 eval samples from current run")
    parser.add_argument("--out-dir", type=Path, 
                        default=Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"))
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to analyze")
    parser.add_argument("--run-start", type=str, default=None, 
                        help="Run start time in ISO format (e.g., 2026-06-02T11:30:00). If not set, uses file mtime heuristic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--output", type=Path, default=Path("analysis_1000_samples.txt"))
    args = parser.parse_args()
    
    # Determine run start time
    if args.run_start:
        run_start = datetime.fromisoformat(args.run_start)
    else:
        # Use eval_summary_progress.json evaluated_at as proxy
        progress_file = args.out_dir / "eval_summary_progress.json"
        if progress_file.exists():
            progress = json.loads(progress_file.read_text())
            run_start = datetime.fromisoformat(progress.get("evaluated_at", datetime.now().isoformat()))
            # Subtract a small buffer to ensure we don't miss early files
            run_start = datetime.fromtimestamp(run_start.timestamp() - 300)
        else:
            run_start = datetime.now()
    
    print(f"Using run start time: {run_start}")
    
    # Find files
    files = find_new_eval_files(args.out_dir, run_start, args.samples)
    
    if len(files) < args.samples:
        print(f"Warning: only found {len(files)} new files, requested {args.samples}")
    
    # Load records
    print(f"Loading {len(files)} records...")
    records = load_eval_records(files)
    print(f"Successfully loaded {len(records)} records")
    
    # Analyze
    print("Analyzing...")
    stats = analyze_records(records)
    
    # Generate report
    report = format_report(stats, records)
    
    # Save
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {args.output}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
