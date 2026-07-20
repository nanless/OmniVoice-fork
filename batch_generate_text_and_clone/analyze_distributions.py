#!/usr/bin/env python3
"""Comprehensive distribution analysis for CER, MOS, and SIM evaluations.

Reads aggregated eval data and computes detailed statistics:
  - Overall: mean, median, std, percentiles (p1, p5, p10, p25, p50, p75, p90, p95, p99)
  - By dataset
  - By language
  - Top-K best / worst cases
  - CER vs MOS vs SIM correlation

Outputs:
  - Text report to stdout
  - JSON summary to --output-json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))

EVAL_SIM_DIR = EVAL_DIR / "eval_sim"
sys.path.insert(0, str(EVAL_SIM_DIR))
from metric_contract import SimilarityCollectionValidator, similarity_metadata  # noqa: E402

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "CLONED_VOICES_ROOT",
        "/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned",
    )
)

SKIP_DIRS = {"logs", "__pycache__", "eval_sim_embedding_cache"}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def extract_dataset(audio_path: str, out_dir: str) -> str:
    """Extract dataset name from audio path relative to out_dir."""
    try:
        rel = os.path.relpath(audio_path, out_dir)
        return rel.split(os.sep)[0]
    except (ValueError, IndexError):
        return "unknown"


def parse_language(raw_lang) -> str:
    if not raw_lang:
        return "unknown"
    lang_map = {"en_mostly": "en", "frequent_mix": "zh", "cn_mostly": "zh"}
    return lang_map.get(str(raw_lang), str(raw_lang))


def compute_stats(values: List[float]) -> dict:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    a = np.array(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p1": float(np.percentile(a, 1)),
        "p5": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def histogram_bins(values: List[float], num_bins: int = 20) -> dict:
    """Compute histogram bins."""
    if not values:
        return {"bins": [], "counts": [], "edges": []}
    counts, edges = np.histogram(values, bins=num_bins)
    return {
        "bins": [float((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))],
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
    }


# ────────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────────

def load_cer_data(out_dir: Path) -> List[dict]:
    """Load CER data from eval_cer_details.jsonl."""
    path = out_dir / "eval_cer_details.jsonl"
    if not path.exists():
        print(f"WARN: {path} not found", file=sys.stderr)
        return []
    t0 = time.time()
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav = r.get("wav", "")
            records.append({
                "wav": wav,
                "dataset": extract_dataset(wav, str(out_dir)),
                "cer": r.get("cer"),
            })
    print(f"[CER] Loaded {len(records)} records in {time.time() - t0:.1f}s", file=sys.stderr)
    return records


def load_sim_data(out_dir: Path) -> List[dict]:
    """Load SIM data from eval_sim_details.jsonl and worker part files, deduplicating by cloned_audio."""
    t0 = time.time()
    validator = SimilarityCollectionValidator()
    records = []

    sim_files = sorted(out_dir.glob("eval_sim_details*.jsonl"))
    for sfp in sim_files:
        with open(sfp, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{sfp}:{line_no}: invalid JSON: {exc}") from exc
                if not validator.add(r, f"{sfp}:{line_no}"):
                    continue
                wav = r["cloned_audio"]
                records.append({
                    "wav": wav,
                    "dataset": r.get("dataset", extract_dataset(wav, str(out_dir))),
                    "similarity": r.get("similarity"),
                    "language": parse_language(r.get("language")),
                    "speed": r.get("speed"),
                })

    print(f"[SIM] Loaded {len(records)} records (deduped from {len(sim_files)} files) in {time.time() - t0:.1f}s", file=sys.stderr)
    return records


def _scan_mos_dir(root: str) -> List[dict]:
    """Scan one directory tree for .mos.json sidecar files."""
    results = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".mos.json"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                with open(fp, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            wav = r.get("cloned_audio", "")
            results.append({
                "wav": wav,
                "dataset": r.get("dataset", ""),
                "language": parse_language(r.get("language")),
                "speed": r.get("speed"),
                "utmos": r.get("utmos22strong"),
            })
    return results


def load_mos_data(out_dir: Path, workers: int = 8) -> List[dict]:
    """Load MOS data by scanning .mos.json files with multiprocessing."""
    t0 = time.time()
    subdirs = [str(p) for p in out_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    total = 0
    records = []
    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_mos_dir, sd): sd for sd in subdirs}
            for future in as_completed(futures):
                batch = future.result()
                records.extend(batch)
                total += len(batch)
                if total % 50000 < len(batch):
                    print(f"[MOS] {total} records … {time.time() - t0:.1f}s", file=sys.stderr)
    else:
        for sd in subdirs:
            batch = _scan_mos_dir(sd)
            records.extend(batch)
            total += len(batch)
            print(f"[MOS] {total} records … {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"[MOS] Loaded {len(records)} records in {time.time() - t0:.1f}s", file=sys.stderr)
    return records


# ────────────────────────────────────────────────────────────────────────────
# Correlation helper
# ────────────────────────────────────────────────────────────────────────────

def build_joined_table(cer_data: list, sim_data: list, mos_data: list) -> list:
    """Join CER/SIM/MOS records by wav path into a unified table."""
    t0 = time.time()
    cer_map = {r["wav"]: r for r in cer_data}
    sim_map = {r["wav"]: r for r in sim_data}
    mos_map = {r["wav"]: r for r in mos_data}

    all_wavs = set(cer_map.keys()) | set(sim_map.keys()) | set(mos_map.keys())
    table = []
    for wav in all_wavs:
        row = {"wav": wav, "dataset": "", "language": "unknown"}
        r_cer = cer_map.get(wav, {})
        r_sim = sim_map.get(wav, {})
        r_mos = mos_map.get(wav, {})

        row["dataset"] = r_cer.get("dataset") or r_sim.get("dataset") or r_mos.get("dataset") or "unknown"
        row["language"] = r_sim.get("language") or r_mos.get("language") or "unknown"
        row["cer"] = r_cer.get("cer")
        row["similarity"] = r_sim.get("similarity")
        row["utmos"] = r_mos.get("utmos")
        row["speed"] = r_mos.get("speed") or r_sim.get("speed")
        table.append(row)

    print(f"[JOIN] {len(table)} unified records in {time.time() - t0:.1f}s", file=sys.stderr)
    return table


# ────────────────────────────────────────────────────────────────────────────
# Report generation
# ────────────────────────────────────────────────────────────────────────────

def build_distribution_report(
    values: List[float],
    name: str,
    by_dataset: Dict[str, List[float]],
    by_language: Dict[str, List[float]],
    worse_is_better: bool = True,
    top_k: int = 10,
) -> dict:
    """Build full distribution report dict."""
    overall = compute_stats(values)
    overall["histogram"] = histogram_bins(values)

    ds_stats = {}
    for ds, vlist in sorted(by_dataset.items()):
        ds_stats[ds] = compute_stats(vlist)
        ds_stats[ds]["histogram"] = histogram_bins(vlist)

    lang_stats = {}
    for lang, vlist in sorted(by_language.items()):
        if len(vlist) >= 100:
            lang_stats[lang] = compute_stats(vlist)

    # Identify best/worst by wav path
    report = {
        "metric": name,
        "overall": overall,
        "by_dataset": ds_stats,
        "by_language": lang_stats,
    }
    return report


def format_stats_section(label: str, stats: dict) -> str:
    """Format a stats dict as a readable text block."""
    if not stats.get("count"):
        return f"  {label}: (no data)\n"
    lines = [f"  {label}: n={stats['count']}"]
    for k in ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
        v = stats.get(k)
        if v is not None:
            lines.append(f"    {k}: {v:.4f}")
    return "\n".join(lines) + "\n"


def print_text_report(
    cer_records: list,
    sim_records: list,
    mos_records: list,
    table: list,
    out_dir: str,
) -> str:
    """Generate a formatted text report."""
    buf = io.StringIO()

    buf.write("=" * 72 + "\n")
    buf.write("  OmniVoice Evaluation Distribution Report\n")
    buf.write(f"  Out Dir: {out_dir}\n")
    buf.write("=" * 72 + "\n\n")

    # CER
    buf.write("━" * 72 + "\n")
    buf.write(" 1. Character Error Rate (CER)\n")
    buf.write("━" * 72 + "\n\n")

    cer_vals = [r["cer"] for r in cer_records if r["cer"] is not None]

    buf.write(f"  Total CER records: {len(cer_records)}\n")
    buf.write(f"  With deterministic CER v4: {len(cer_vals)}\n\n")

    buf.write("  ── Deterministic safe-TN CER v4 ──\n")
    buf.write(format_stats_section("Overall", compute_stats(cer_vals)))

    cer_by_ds = defaultdict(list)
    cer_by_lang = defaultdict(list)
    for r in cer_records:
        ds = r.get("dataset", "unknown")
        lang = r.get("language", "unknown")
        if r["cer"] is not None:
            cer_by_ds[ds].append(r["cer"])
            cer_by_lang[lang].append(r["cer"])
    if cer_by_ds:
        buf.write("\n  ── CER by Dataset ──\n")
        for ds in sorted(cer_by_ds.keys()):
            buf.write(format_stats_section(ds, compute_stats(cer_by_ds[ds])))

    # SIM
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 2. Raw Cosine Speaker Similarity [-1, 1]\n")
    buf.write("━" * 72 + "\n\n")

    sim_vals = [r["similarity"] for r in sim_records if r["similarity"] is not None]
    buf.write(f"  Total SIM records: {len(sim_records)}\n")
    buf.write(f"  With raw cosine:   {len(sim_vals)}\n\n")
    buf.write(format_stats_section("Overall", compute_stats(sim_vals)))

    sim_by_ds = defaultdict(list)
    sim_by_lang = defaultdict(list)
    for r in sim_records:
        ds = r.get("dataset", "unknown")
        lang = r.get("language", "unknown")
        if r["similarity"] is not None:
            sim_by_ds[ds].append(r["similarity"])
            sim_by_lang[lang].append(r["similarity"])
    if sim_by_ds:
        buf.write("\n  ── Raw Cosine Similarity by Dataset ──\n")
        for ds in sorted(sim_by_ds.keys()):
            buf.write(format_stats_section(ds, compute_stats(sim_by_ds[ds])))

    # MOS
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 3. UTMOS22Strong\n")
    buf.write("━" * 72 + "\n\n")

    mos_vals = [r["utmos"] for r in mos_records if r["utmos"] is not None]
    buf.write(f"  Total MOS records: {len(mos_records)}\n")
    buf.write(f"  With UTMOS:        {len(mos_vals)}\n\n")
    buf.write(format_stats_section("Overall", compute_stats(mos_vals)))

    mos_by_ds = defaultdict(list)
    mos_by_lang = defaultdict(list)
    for r in mos_records:
        ds = r.get("dataset", "unknown")
        lang = r.get("language", "unknown")
        if r["utmos"] is not None:
            mos_by_ds[ds].append(r["utmos"])
            mos_by_lang[lang].append(r["utmos"])
    if mos_by_ds:
        buf.write("\n  ── UTMOS by Dataset ──\n")
        for ds in sorted(mos_by_ds.keys()):
            buf.write(format_stats_section(ds, compute_stats(mos_by_ds[ds])))

    # Correlations
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 4. Cross-Metric Correlations (Pearson's r)\n")
    buf.write("━" * 72 + "\n\n")

    table_vals = [
        r for r in table
        if r["cer"] is not None and r["similarity"] is not None and r["utmos"] is not None
    ]
    buf.write(f"  Paired records: {len(table_vals)}\n\n")

    if len(table_vals) >= 3:
        cer_a = np.array([r["cer"] for r in table_vals])
        sim_a = np.array([r["similarity"] for r in table_vals])
        mos_a = np.array([r["utmos"] for r in table_vals])

        r_cer_sim = np.corrcoef(cer_a, sim_a)[0, 1]
        r_cer_mos = np.corrcoef(cer_a, mos_a)[0, 1]
        r_sim_mos = np.corrcoef(sim_a, mos_a)[0, 1]

        buf.write(f"  CER vs Similarity:  r = {r_cer_sim:.4f}\n")
        buf.write(f"  CER vs UTMOS:       r = {r_cer_mos:.4f}\n")
        buf.write(f"  Similarity vs UTMOS: r = {r_sim_mos:.4f}\n")

    # Top/Bottom lists in table form
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 5. Summary Table\n")
    buf.write("━" * 72 + "\n\n")

    all_ds = set()
    for r in cer_records:
        all_ds.add(r.get("dataset", "unknown"))
    for r in sim_records:
        all_ds.add(r.get("dataset", "unknown"))
    for r in mos_records:
        all_ds.add(r.get("dataset", "unknown"))

    # Build per-dataset summary
    def _ds_stats(ds):
        c = compute_stats([r["cer"] for r in cer_records if r.get("dataset") == ds and r["cer"] is not None])
        s = compute_stats([r["similarity"] for r in sim_records if r.get("dataset") == ds and r["similarity"] is not None])
        m = compute_stats([r["utmos"] for r in mos_records if r.get("dataset") == ds and r["utmos"] is not None])
        return c, s, m

    # Shorten dataset names and build rows first
    def _short_name(name: str) -> str:
        mapping = {
            "BAAI-ChildMandarin41.25H_integrated_by_groundtruth_onlyenhanced": "BAAI-ChildMandarin",
            "Chinese_English_Scripted_Speech_Corpus_Children_integrated_by_groundtruth_onlyenhanced": "Chinese_English_Children",
            "King-ASR-EN-Kid_integrated_by_groundtruth_onlyenhanced": "King-ASR-EN-Kid",
            "speechocean762_integrated_by_groundtruth_onlyenhanced": "speechocean762",
        }
        return mapping.get(name, name[:45])

    rows = []
    for ds in sorted(all_ds, key=lambda d: (d == "unknown", d)):
        c, s, m = _ds_stats(ds)
        sn = _short_name(ds)
        rows.append((sn, c, s, m))
    max_name = max(len(r[0]) for r in rows) if rows else 10
    header = f"  {'Dataset':<{max_name}s} {'CER_n':>7s} {'CER_mean':>9s} {'SIM_n':>7s} {'RAW_SIM':>9s} {'MOS_n':>7s} {'MOS_mean':>9s}"
    buf.write(header + "\n")
    buf.write(f"  {'─'*max_name} {'─'*7} {'─'*9} {'─'*7} {'─'*9} {'─'*7} {'─'*9}\n")
    for sn, c, s, m in rows:
        def _val(stats, key):
            v = stats.get(key) if stats else None
            return f"{v:>9.4f}" if v is not None else f"{'-':>9s}"
        buf.write(
            f"  {sn:<{max_name}s} "
            f"{c.get('count',0):>7d} {_val(c,'mean')}  "
            f"{s.get('count',0):>7d} {_val(s,'mean')}  "
            f"{m.get('count',0):>7d} {_val(m,'mean')}\n"
        )

    buf.write("\n" + "=" * 72 + "\n")
    buf.write("  Report Complete\n")
    buf.write("=" * 72 + "\n")

    return buf.getvalue()


def build_json_report(cer_records, sim_records, mos_records, table) -> dict:
    """Build JSON-serializable report dict."""

    def _build_metric(values, records, metric_key, by_ds=True, by_lang=True):
        overall = compute_stats(values)
        overall["histogram"] = histogram_bins(values)
        report = {"overall": overall}
        if by_ds:
            dd = defaultdict(list)
            for r in records:
                if r.get(metric_key) is not None:
                    dd[r.get("dataset", "unknown")].append(r[metric_key])
            report["by_dataset"] = {k: compute_stats(v) for k, v in sorted(dd.items())}
        if by_lang:
            dl = defaultdict(list)
            for r in records:
                if r.get(metric_key) is not None:
                    dl[r.get("language", "unknown")].append(r[metric_key])
            report["by_language"] = {k: compute_stats(v) for k, v in sorted(dl.items()) if len(v) >= 100}
        return report

    report = {
        "out_dir": str(DEFAULT_OUT_DIR),
        "cer": _build_metric(
            [r["cer"] for r in cer_records if r["cer"] is not None],
            cer_records, "cer",
        ),
    }

    report["sim"] = {
        **similarity_metadata(),
        **_build_metric(
            [r["similarity"] for r in sim_records if r["similarity"] is not None],
            sim_records, "similarity",
        ),
    }
    report["mos"] = _build_metric(
        [r["utmos"] for r in mos_records if r["utmos"] is not None],
        mos_records, "utmos",
    )

    # Correlations
    table_vals = [
        r for r in table
        if r["cer"] is not None and r["similarity"] is not None and r["utmos"] is not None
    ]
    if len(table_vals) >= 3:
        cer_a = np.array([r["cer"] for r in table_vals])
        sim_a = np.array([r["similarity"] for r in table_vals])
        mos_a = np.array([r["utmos"] for r in table_vals])
        report["correlations"] = {
            "cer_vs_similarity": float(np.corrcoef(cer_a, sim_a)[0, 1]),
            "cer_vs_utmos": float(np.corrcoef(cer_a, mos_a)[0, 1]),
            "similarity_vs_utmos": float(np.corrcoef(sim_a, mos_a)[0, 1]),
            "paired_count": len(table_vals),
        }

    report["counts"] = {
        "total_cer": len(cer_records),
        "total_sim": len(sim_records),
        "total_mos": len(mos_records),
        "total_joined": len(table),
    }

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None, help="Write JSON report")
    parser.add_argument("--output-txt", type=Path, default=None, help="Write text report")
    parser.add_argument("--skip-mos", action="store_true", help="Skip MOS (slow)")
    parser.add_argument("--skip-corr", action="store_true", help="Skip correlation analysis")
    parser.add_argument("--mos-workers", type=int, default=8, help="MOS scan workers")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample for quick test")
    args = parser.parse_args()

    out = args.out_dir

    # Load CER (fast — single file)
    print("=== Loading CER data ===", file=sys.stderr)
    cer_records = load_cer_data(out)
    if args.sample_size and args.sample_size < len(cer_records):
        import random
        cer_records = random.Random(42).sample(cer_records, args.sample_size)

    # Load SIM (fast — few files)
    print("=== Loading SIM data ===", file=sys.stderr)
    sim_records = load_sim_data(out)
    if args.sample_size and args.sample_size < len(sim_records):
        import random
        sim_records = random.Random(42).sample(sim_records, args.sample_size)

    # Load MOS (slow — scan 838k sidecar files)
    mos_records = []
    if not args.skip_mos:
        print("=== Loading MOS data ===", file=sys.stderr)
        mos_records = load_mos_data(out, workers=args.mos_workers)
        if args.sample_size and args.sample_size < len(mos_records):
            import random
            mos_records = random.Random(42).sample(mos_records, args.sample_size)

    # Join
    table = []
    if not args.skip_corr and mos_records:
        print("=== Building joined table ===", file=sys.stderr)
        table = build_joined_table(cer_records, sim_records, mos_records)

    # Generate reports
    print("=== Generating reports ===", file=sys.stderr)
    text_report = print_text_report(cer_records, sim_records, mos_records, table, str(out))
    print(text_report)

    if args.output_txt:
        args.output_txt.write_text(text_report, encoding="utf-8")
        print(f"[Wrote] {args.output_txt}", file=sys.stderr)

    if args.output_json:
        json_report = build_json_report(cer_records, sim_records, mos_records, table)
        args.output_json.write_text(
            json.dumps(json_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Wrote] {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
