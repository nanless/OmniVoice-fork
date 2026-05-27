#!/usr/bin/env python3
"""Evaluate all cloned audios: full-audio ASR + Manual/LLM ITN + char-level CER.

Same pipeline as eval_batch_200.py, applied to every text_*.wav under --out-dir.

Usage:
    conda activate omnivoice
    cd batch_generate_text_and_clone/eval_cer

    python eval_cloned.py
    python eval_cloned.py --skip-asr --refresh-llm-cache
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import eval_batch_200 as eb  # noqa: E402


def find_all_cloned(out_dir: Path):
    """Yield (wav_path, json_path) for every cloned audio."""
    for json_path in sorted(out_dir.rglob("text_*.json")):
        wav_path = json_path.with_suffix(".wav")
        if wav_path.exists():
            yield wav_path, json_path


def write_eval_json(json_path: Path, record: dict):
    eval_path = json_path.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/code/github_repos/OmniVoice-fork/batch_cloned_voices"),
        help="Root directory of cloned wav/json pairs",
    )
    parser.add_argument("--skip-asr", action="store_true", help="Use cached ASR results")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM ITN")
    parser.add_argument("--llm-batch-size", type=int, default=10, help="Text pairs per LLM request")
    parser.add_argument("--llm-concurrency", type=int, default=5, help="Parallel LLM requests")
    parser.add_argument("--refresh-llm-cache", action="store_true", help="Re-run LLM ITN")
    args = parser.parse_args()

    pairs = list(find_all_cloned(args.out_dir))
    if not pairs:
        print("No cloned audio found.")
        return
    print(f"Found {len(pairs)} cloned audios to evaluate")

    paths = eb.eval_paths_full(args.out_dir)

    print("\n[1/4] ASR (full audio, no VAD)...")
    asr_results = eb.run_asr(pairs, use_cache=args.skip_asr, cache_path=paths["asr_cache"])
    eb.validate_asr_cache(pairs, asr_results)

    print("\n[2/4] Manual ITN + CER...")
    detailed = []
    print(f"\n{'='*95}")
    print(f"{'File':<30} {'Manual':>8}")
    print(f"{'='*95}")
    for wav_path, json_path in pairs:
        hypo_raw = asr_results.get(str(wav_path), "")
        item = eb.build_eval_item(wav_path, json_path, hypo_raw)
        detailed.append(item)
        print(f"{wav_path.name:<30} {item['manual_cer']*100:>7.1f}%")

    manual_summary = eb.summarize_cer(
        [{**d, "ref_final": d["ref_manual"], "hypo_final": d["hypo_manual"]} for d in detailed],
        "ref_final",
        "hypo_final",
    )

    llm_summary = None
    model_name = None
    if not args.skip_llm:
        print("\n[3/4] LLM ITN...")
        llm_cache = eb.run_llm_itn(
            detailed,
            cache_path=paths["llm_cache"],
            batch_size=args.llm_batch_size,
            concurrency=args.llm_concurrency,
            use_cache=not args.refresh_llm_cache,
        )
        for item in detailed:
            llm = llm_cache.get(item["wav"], {})
            item["ref_llm"], item["hypo_llm"] = eb.llm_itn_postprocess(
                llm.get("ref_final", ""),
                llm.get("hypo_final", ""),
                item["ref_manual"],
                item["hypo_manual"],
            )
            item["llm_cer"], item["llm_sub"], item["llm_ins"], item["llm_del"], _ = eb.calc_cer(
                item["ref_llm"], item["hypo_llm"]
            )

        llm_summary = eb.summarize_cer(
            [{**d, "ref_final": d["ref_llm"], "hypo_final": d["hypo_llm"]} for d in detailed],
            "ref_final",
            "hypo_final",
        )
        model_name = os.environ.get("ITN_LLM_MODEL", "deepseek-v4-flash")

        print(f"{'='*95}")
        print(f"\nManual Weighted CER: {manual_summary['weighted_cer']:.2f}%")
        print(f"LLM    Weighted CER: {llm_summary['weighted_cer']:.2f}%")
        print(f"Delta: {llm_summary['weighted_cer'] - manual_summary['weighted_cer']:+.2f}%")
        llm_worse = sum(
            1 for d in detailed if d.get("llm_cer", 0) > d["manual_cer"] + 1e-9
        )
        print(f"LLM worse than manual: {llm_worse}/{len(detailed)}")
    else:
        print("\n[3/4] LLM ITN skipped")

    evaluated_at = datetime.now().isoformat()
    n = len(detailed)

    print("\n[4/4] Writing outputs...")
    for item in tqdm(detailed, desc="Write eval.json"):
        json_path = Path(item["json"])
        record = {
            "wav_path": item["wav"],
            "ref_audio": item.get("ref_audio"),
            "gen_text": item["ref_start"],
            "asr_hypo": item["hypo_start"],
            "ref_manual": item["ref_manual"],
            "hypo_manual": item["hypo_manual"],
            "manual_cer": item["manual_cer"],
            "substitutions": item["substitutions"],
            "insertions": item["insertions"],
            "deletions": item["deletions"],
            "chars": item["chars"],
            "evaluated_at": evaluated_at,
        }
        if llm_summary:
            record.update({
                "ref_llm": item["ref_llm"],
                "hypo_llm": item["hypo_llm"],
                "llm_cer": item["llm_cer"],
                "llm_model": model_name,
            })
        write_eval_json(json_path, record)

    summary = {
        "out_dir": str(args.out_dir),
        "sample_size": n,
        "manual": manual_summary,
        "llm": llm_summary,
        "llm_model": model_name,
        "asr_cache": str(paths["asr_cache"]),
        "llm_cache": str(paths["llm_cache"]) if llm_summary else None,
        "evaluated_at": evaluated_at,
        "details": sorted(detailed, key=lambda x: x["manual_cer"], reverse=True),
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    eb.write_details(
        paths["details_manual"],
        f"Full Eval — Manual ITN ({n} audios, sorted by CER high → low)",
        manual_summary, detailed, "ref_manual", "hypo_manual", "manual_cer", evaluated_at,
    )
    if llm_summary:
        eb.write_details(
            paths["details_llm"],
            f"Full Eval — LLM ITN ({model_name}, {n} audios)",
            llm_summary, detailed, "ref_llm", "hypo_llm", "llm_cer", evaluated_at,
        )
        eb.write_comparison(
            paths["comparison"],
            manual_summary, llm_summary, detailed, evaluated_at, n,
        )
    eb.write_details(
        paths["details_legacy"],
        f"Full Eval Details ({n} audios, sorted by CER high → low)",
        manual_summary, detailed, "ref_manual", "hypo_manual", "manual_cer", evaluated_at,
    )

    print(f"\n{'='*60}")
    print(f"Files: {n}")
    print(f"Manual Weighted CER: {manual_summary['weighted_cer']:.2f}%")
    if llm_summary:
        print(f"LLM    Weighted CER: {llm_summary['weighted_cer']:.2f}%")
        print(f"Delta: {llm_summary['weighted_cer'] - manual_summary['weighted_cer']:+.2f}%")
    print(f"{'='*60}")
    print(f"Summary:     {paths['summary']}")
    print(f"Manual txt:  {paths['details_manual']}")
    if llm_summary:
        print(f"LLM txt:     {paths['details_llm']}")
        print(f"Comparison:  {paths['comparison']}")
    print(f"Per-file:    text_*.eval.json")


if __name__ == "__main__":
    main()
