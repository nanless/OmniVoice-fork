#!/usr/bin/env python3
"""Evaluate all cloned audios: async ASR (producer) + LLM ITN (consumer pool).

ASR pushes batch_size=16 items to a queue; ITN workers (default 5) consume in parallel.
"""

import argparse
import json
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_batch_200 as eb  # noqa: E402
from eval_common import CerAccumulator, append_jsonl, list_clone_items, write_json  # noqa: E402


def find_all_cloned(out_dir: Path):
    return list_clone_items(out_dir, label="cer-scan")


def write_eval_json(json_path: Path, record: dict):
    eval_path = json_path.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_eval_record(item: dict, evaluated_at: str, llm: bool = False, model_name=None) -> dict:
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
        "stage": "complete" if llm else "manual",
    }
    if llm:
        record.update({
            "ref_llm": item["ref_llm"],
            "hypo_llm": item["hypo_llm"],
            "llm_cer": item["llm_cer"],
            "llm_model": model_name,
        })
    return record


def apply_llm_to_item(item: dict, llm: dict, model_name: str) -> dict:
    item["ref_llm"], item["hypo_llm"] = eb.llm_itn_postprocess(
        llm.get("ref_final", ""),
        llm.get("hypo_final", ""),
        item["ref_manual"],
        item["hypo_manual"],
    )
    item["llm_cer"], item["llm_sub"], item["llm_ins"], item["llm_del"], _ = eb.calc_cer(
        item["ref_llm"], item["hypo_llm"]
    )
    item["llm_model"] = model_name
    return item


def flush_asr_cache(cache_path: Path, asr_results: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(asr_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def flush_llm_cache(cache_path: Path, llm_cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(llm_cache, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned"),
    )
    parser.add_argument("--skip-asr", action="store_true", help="Use cached ASR results")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM ITN")
    parser.add_argument("--batch-size", type=int, default=16, help="ASR + ITN batch size")
    parser.add_argument("--asr-batch-size", type=int, default=None, help="Alias for --batch-size")
    parser.add_argument("--llm-concurrency", type=int, default=12, help="Parallel ITN workers (3 GPUs x 4 concurrent requests)")
    parser.add_argument("--refresh-llm-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    batch_size = args.asr_batch_size if args.asr_batch_size is not None else args.batch_size

    pairs = list(find_all_cloned(args.out_dir))
    if args.skip_existing:
        before = len(pairs)
        filtered = []
        for w, j in pairs:
            ev = j.with_suffix(".eval.json")
            if not ev.exists():
                filtered.append((w, j))
                continue
            if args.skip_llm:
                continue
            try:
                meta = json.loads(ev.read_text(encoding="utf-8"))
                if meta.get("stage") == "complete" or meta.get("llm_cer") is not None:
                    continue
            except (json.JSONDecodeError, OSError):
                pass
            filtered.append((w, j))
        pairs = filtered
        print(f"Skip-existing: {before - len(pairs)} already done, {len(pairs)} remaining", flush=True)
    if not pairs:
        print("No cloned audio found.")
        return

    print(
        f"Found {len(pairs)} audios | batch={batch_size} itn_workers={args.llm_concurrency}",
        flush=True,
    )

    paths = eb.eval_paths_full(args.out_dir)
    if not args.skip_existing and paths["details_jsonl"].exists():
        paths["details_jsonl"].unlink()

    evaluated_at = datetime.now().isoformat()
    detailed: list[dict] = []
    manual_acc = CerAccumulator()
    llm_acc = CerAccumulator()
    model_name = os.environ.get("ITN_LLM_MODEL", "deepseek-v4-flash")
    use_llm_cache = not args.refresh_llm_cache

    asr_results = eb.load_json(paths["asr_cache"]) if paths["asr_cache"].exists() else {}
    if asr_results:
        print(f"Loaded ASR cache: {len(asr_results)} entries", flush=True)

    llm_cache: dict = {}
    if use_llm_cache and paths["llm_cache"].exists():
        llm_cache = eb.load_json(paths["llm_cache"])
        print(f"Loaded LLM ITN cache: {len(llm_cache)} entries", flush=True)

    write_lock = threading.Lock()
    llm_cache_lock = threading.Lock()
    producer_error: list = []

    def flush_progress(stage: str):
        prog = {
            "out_dir": str(args.out_dir),
            "stage": stage,
            "items_done": len(detailed),
            "items_total": len(pairs),
            "batch_size": batch_size,
            "llm_concurrency": args.llm_concurrency,
            "manual": manual_acc.to_dict(),
            "evaluated_at": datetime.now().isoformat(),
        }
        if llm_acc.count:
            prog["llm"] = llm_acc.to_dict()
        write_json(paths["summary_progress"], prog)

    def log_line(msg: str):
        """Thread-safe log line (visible in cer.log despite tqdm)."""
        tqdm.write(msg, file=sys.stderr)

    def save_item(item: dict, llm: bool = False):
        write_eval_json(Path(item["json"]), build_eval_record(item, evaluated_at, llm=llm, model_name=model_name))
        row = {
            "wav": item["wav"],
            "manual_cer": item["manual_cer"],
            "stage": "complete" if llm else "manual",
        }
        if llm:
            row["llm_cer"] = item["llm_cer"]
        append_jsonl(paths["details_jsonl"], row)

    def build_batch_items(batch_pairs):
        items = []
        for wav_path, json_path in batch_pairs:
            hypo_raw = asr_results.get(str(wav_path), "")
            item = eb.build_eval_item(wav_path, json_path, hypo_raw)
            items.append(item)
            manual_acc.add(item["substitutions"], item["insertions"], item["deletions"], item["chars"])
        return items

    def process_itn_batch(batch_items: list, worker_id: int = 0):
        with llm_cache_lock:
            pending = [
                it for it in batch_items
                if not (use_llm_cache and it["wav"] in llm_cache)
            ]
        n_pending = len(pending)
        log_line(f"  [ITN start w{worker_id}] batch={len(batch_items)} llm_call={n_pending}")
        if pending:
            new_entries = eb.llm_itn_batch_fetch(pending)
            with llm_cache_lock:
                llm_cache.update(new_entries)
                flush_llm_cache(paths["llm_cache"], llm_cache)

        with write_lock:
            for item in batch_items:
                with llm_cache_lock:
                    llm = llm_cache.get(item["wav"], {})
                apply_llm_to_item(item, llm, model_name)
                llm_acc.add(item["llm_sub"], item["llm_ins"], item["llm_del"], item["chars"])
                detailed.append(item)
                save_item(item, llm=True)
            flush_progress("itn")
        log_line(f"  [ITN done w{worker_id}] +{len(batch_items)} total={len(detailed)}")

    num_batches = (len(pairs) + batch_size - 1) // batch_size

    if args.skip_llm:
        asr = None
        if not args.skip_asr and any(not asr_results.get(str(w)) for w, _ in pairs):
            asr = eb.load_asr_model(batch_size)
        print("\n[Pipeline] ASR only (no LLM)...", flush=True)
        for bi in tqdm(range(num_batches), desc="ASR batches"):
            batch = pairs[bi * batch_size:(bi + 1) * batch_size]
            if not args.skip_asr:
                need = [(w, j) for w, j in batch if not asr_results.get(str(w))]
                if need:
                    eb.transcribe_asr_batch(asr, need, asr_results)
                    flush_asr_cache(paths["asr_cache"], asr_results)
            batch_items = build_batch_items(batch)
            with write_lock:
                for item in batch_items:
                    detailed.append(item)
                    save_item(item, llm=False)
                flush_progress("asr")
        if asr is not None:
            del asr
            torch.cuda.empty_cache()
    else:
        batch_queue: queue.Queue = queue.Queue(maxsize=args.llm_concurrency * 6)

        def itn_worker(worker_id: int):
            while True:
                batch_items = batch_queue.get()
                try:
                    if batch_items is None:
                        break
                    process_itn_batch(batch_items, worker_id)
                except Exception as e:
                    producer_error.append(e)
                    log_line(f"[itn-w{worker_id}] error: {e}")
                finally:
                    batch_queue.task_done()

        workers = [
            threading.Thread(target=itn_worker, args=(i,), daemon=True)
            for i in range(args.llm_concurrency)
        ]
        for t in workers:
            t.start()

        asr = None
        if not args.skip_asr and any(not asr_results.get(str(w)) for w, _ in pairs):
            asr = eb.load_asr_model(batch_size)

        print(
            f"\n[Pipeline] ASR producer → queue → {args.llm_concurrency} ITN workers...",
            flush=True,
        )

        def asr_producer():
            try:
                for bi in tqdm(range(num_batches), desc="ASR → queue"):
                    batch = pairs[bi * batch_size:(bi + 1) * batch_size]
                    if not args.skip_asr:
                        need = [(w, j) for w, j in batch if not asr_results.get(str(w))]
                        if need:
                            eb.transcribe_asr_batch(asr, need, asr_results)
                            flush_asr_cache(paths["asr_cache"], asr_results)
                        if bi % 500 == 0 and bi > 0:
                            torch.cuda.empty_cache()
                            log_line(f"  [ASR] cleared GPU cache at batch {bi}")
                    batch_items = build_batch_items(batch)
                    batch_queue.put(batch_items)
                    log_line(f"  [ASR batch {bi + 1}/{num_batches}] queued {len(batch_items)}")
            except Exception as e:
                producer_error.append(e)
                print(f"[ASR producer] error: {e}", flush=True)
            finally:
                for _ in range(args.llm_concurrency):
                    batch_queue.put(None)

        producer = threading.Thread(target=asr_producer, daemon=False)
        producer.start()
        batch_queue.join()
        producer.join()
        for t in workers:
            t.join(timeout=1)

        if asr is not None:
            del asr
            torch.cuda.empty_cache()

        if producer_error:
            raise producer_error[0]

    eb.validate_asr_cache(pairs, asr_results)

    manual_summary = eb.summarize_cer(
        [{**d, "ref_final": d["ref_manual"], "hypo_final": d["hypo_manual"]} for d in detailed],
        "ref_final",
        "hypo_final",
    )
    llm_summary = None
    if not args.skip_llm:
        llm_summary = eb.summarize_cer(
            [{**d, "ref_final": d["ref_llm"], "hypo_final": d["hypo_llm"]} for d in detailed],
            "ref_final",
            "hypo_final",
        )
        flush_progress("complete")
        print(f"Manual Weighted CER: {manual_summary['weighted_cer']:.2f}%", flush=True)
        print(f"LLM    Weighted CER: {llm_summary['weighted_cer']:.2f}%", flush=True)

    n = len(detailed)
    summary = {
        "out_dir": str(args.out_dir),
        "sample_size": n,
        "batch_size": batch_size,
        "llm_concurrency": args.llm_concurrency,
        "manual": manual_summary,
        "llm": llm_summary,
        "llm_model": model_name if llm_summary else None,
        "asr_cache": str(paths["asr_cache"]),
        "llm_cache": str(paths["llm_cache"]) if llm_summary else None,
        "details_jsonl": str(paths["details_jsonl"]),
        "evaluated_at": evaluated_at,
    }
    write_json(paths["summary"], summary)

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

    print(f"\n{'='*60}", flush=True)
    print(f"Files: {n} | batch={batch_size} itn_workers={args.llm_concurrency}", flush=True)
    print(f"Progress: {paths['summary_progress']}", flush=True)


if __name__ == "__main__":
    main()
