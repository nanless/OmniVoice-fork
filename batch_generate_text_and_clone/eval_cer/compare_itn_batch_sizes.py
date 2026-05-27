#!/usr/bin/env python3
"""Compare LLM ITN outputs and CER across batch sizes on N fixed samples."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import eval_batch_200 as ev  # noqa: E402


def run_itn_all(
    items: list[dict],
    batch_size: int,
    cache_path: Path | None = None,
) -> dict[str, dict]:
    """Run LLM ITN sequentially (one HTTP request at a time, no concurrency)."""
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = ev.load_json(cache_path)
        print(f"  resume: loaded {len(cache)} cached entries from {cache_path.name}")

    pending = [it for it in items if it["wav"] not in cache]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    if not batches:
        return cache

    print(
        f"  sequential ITN: {len(pending)} pending, "
        f"batch_size={batch_size}, {len(batches)} requests (concurrency=1)",
        flush=True,
    )
    for batch in tqdm(batches, desc=f"ITN bs={batch_size}"):
        cache.update(ev.llm_itn_batch_fetch(batch))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return cache


def cer_for_cache(items: list[dict], cache: dict[str, dict]) -> tuple[list[dict], dict]:
    rows = []
    for item in items:
        llm = cache.get(item["wav"], {})
        ref_llm, hyp_llm = ev.llm_itn_postprocess(
            llm.get("ref_final", ""),
            llm.get("hypo_final", ""),
            item["ref_manual"],
            item["hypo_manual"],
        )
        cer, sub, ins, dele, chars = ev.calc_cer(ref_llm, hyp_llm)
        rows.append(
            {
                "wav": item["wav"],
                "name": item["name"],
                "ref_llm": ref_llm,
                "hypo_llm": hyp_llm,
                "llm_cer": cer,
            }
        )
    summary = ev.summarize_cer(
        [{**r, "ref_final": r["ref_llm"], "hypo_final": r["hypo_llm"]} for r in rows],
        "ref_final",
        "hypo_final",
    )
    return rows, summary


def compare_results(
    items: list[dict],
    results: dict[int, dict[str, dict]],
    batch_sizes: list[int],
) -> dict:
    per_sample = []
    text_diff_counts = {bs: 0 for bs in batch_sizes}
    ref_ref = batch_sizes[0]

    for item in items:
        wav = item["wav"]
        entry = {"wav": wav, "name": item["name"], "by_bs": {}}
        for bs in batch_sizes:
            llm = results[bs].get(wav, {})
            ref_f, hyp_f = ev.llm_itn_postprocess(
                llm.get("ref_final", ""),
                llm.get("hypo_final", ""),
                item["ref_manual"],
                item["hypo_manual"],
            )
            cer, _, _, _, _ = ev.calc_cer(ref_f, hyp_f)
            entry["by_bs"][bs] = {
                "ref_llm": ref_f,
                "hypo_llm": hyp_f,
                "llm_cer": cer,
            }
        ref_entry = entry["by_bs"][ref_ref]
        for bs in batch_sizes:
            cur = entry["by_bs"][bs]
            text_same = (
                cur["ref_llm"] == ref_entry["ref_llm"]
                and cur["hypo_llm"] == ref_entry["hypo_llm"]
            )
            cer_same = abs(cur["llm_cer"] - ref_entry["llm_cer"]) < 1e-9
            cur["text_same_vs_bs4"] = text_same
            cur["cer_same_vs_bs4"] = cer_same
            if not text_same:
                text_diff_counts[bs] += 1
        per_sample.append(entry)

    cer_summaries = {}
    for bs in batch_sizes:
        _, summary = cer_for_cache(items, results[bs])
        cer_summaries[bs] = summary

    return {
        "text_diff_vs_bs4": text_diff_counts,
        "cer_summaries": {
            bs: {
                "weighted_cer": s["weighted_cer"],
                "avg_cer": s["avg_cer"],
                "median_cer": s["median_cer"],
            }
            for bs, s in cer_summaries.items()
        },
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=320)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="4,8,16,32",
        help="Comma-separated ITN batch sizes",
    )
    parser.add_argument(
        "--asr-cache",
        type=Path,
        default=ev.OUT_DIR / "eval_asr_cache.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EVAL_DIR / "tmp_itn_batch_compare",
    )
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    paths = ev.eval_paths(args.sample_size)
    sampled = ev.load_fixed_sample(args.sample_size, paths["sample_list"], seed=args.seed)
    if len(sampled) < args.sample_size:
        print(f"WARNING: only {len(sampled)} valid samples (requested {args.sample_size})")

    asr_results = ev.load_json(args.asr_cache)
    items = []
    missing = 0
    for wav_path, json_path in sampled:
        hypo_raw = asr_results.get(str(wav_path), "")
        if not hypo_raw:
            missing += 1
            continue
        items.append(ev.build_eval_item(wav_path, json_path, hypo_raw))
    print(f"Built {len(items)} eval items (ASR missing: {missing})")
    if not items:
        raise SystemExit("No items with ASR results")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Mode: sequential only (concurrency=1, one LLM request at a time)\n", flush=True)

    results: dict[int, dict[str, dict]] = {}
    for bs in batch_sizes:
        print(f"\n=== Running ITN batch_size={bs} (sequential) ===", flush=True)
        cache_path = args.out_dir / f"llm_itn_cache_bs{bs}.json"
        results[bs] = run_itn_all(items, bs, cache_path=cache_path)
        print(f"Saved {len(results[bs])} entries -> {cache_path}", flush=True)

    report = compare_results(items, results, batch_sizes)
    report["meta"] = {
        "sample_size": len(items),
        "batch_sizes": batch_sizes,
        "asr_cache": str(args.asr_cache),
        "created_at": datetime.now().isoformat(),
    }

    report_path = args.out_dir / "compare_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("CER summary by batch_size")
    print("=" * 70)
    for bs in batch_sizes:
        s = report["cer_summaries"][bs]
        print(
            f"  bs={bs:2d}: weighted={s['weighted_cer']:.4f}%  "
            f"avg={s['avg_cer']:.4f}%  median={s['median_cer']:.4f}%"
        )

    print("\nText differences vs batch_size=4 (ref+hyp both identical):")
    for bs in batch_sizes:
        diff = report["text_diff_vs_bs4"][bs]
        print(f"  bs={bs:2d}: {diff}/{len(items)} samples differ")

    # List samples where any batch size differs
    mismatches = []
    for entry in report["per_sample"]:
        texts = {bs: (entry["by_bs"][bs]["ref_llm"], entry["by_bs"][bs]["hypo_llm"]) for bs in batch_sizes}
        if len(set(texts.values())) > 1:
            cers = {bs: entry["by_bs"][bs]["llm_cer"] for bs in batch_sizes}
            mismatches.append(
                {
                    "name": entry["name"],
                    "cers": {str(k): round(v * 100, 2) for k, v in cers.items()},
                    "by_bs": {
                        str(bs): {
                            "ref": entry["by_bs"][bs]["ref_llm"][:80],
                            "hypo": entry["by_bs"][bs]["hypo_llm"][:80],
                        }
                        for bs in batch_sizes
                    },
                }
            )

    mismatch_path = args.out_dir / "text_mismatches.json"
    mismatch_path.write_text(
        json.dumps(mismatches, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSamples with text differences across batch sizes: {len(mismatches)}")
    print(f"Full report: {report_path}")
    print(f"Mismatches:  {mismatch_path}")


if __name__ == "__main__":
    main()
