#!/usr/bin/env python3
"""Rebuild CER v4 sidecars from provenance-safe v2/v3 ASR text.

The command is a dry run unless ``--write`` is supplied. Legacy normalized
text and every ``llm_*`` field are ignored. Sources without the current ASR
model and decode fingerprints are rejected instead of being relabeled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(EVAL_DIR))

import eval_batch_200 as eb  # noqa: E402
import eval_cloned  # noqa: E402


def read_legacy_source(wav_path: Path, json_path: Path) -> tuple[dict | None, str]:
    eval_path = json_path.with_suffix(".eval.json")
    try:
        record = json.loads(eval_path.read_text(encoding="utf-8"))
        clone = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_sidecar"
    except (json.JSONDecodeError, OSError, TypeError):
        return None, "invalid_json"
    if not isinstance(record, dict) or record.get("eval_schema_version") not in (2, 3):
        return None, "not_legacy_v2_v3"
    gen_text = clone.get("gen_text")
    if not isinstance(gen_text, str) or not gen_text:
        return None, "invalid_gen_text"
    if (
        record.get("wav_path") != str(wav_path)
        or record.get("wav_signature") != eval_cloned.wav_signature(wav_path)
        or record.get("gen_text") != gen_text
    ):
        return None, "source_mismatch"
    if (
        record.get("asr_model") != eb.QWEN3_ASR_LOCAL
        or record.get("asr_model_fingerprint") != eb.asr_model_fingerprint()
    ):
        return None, "stale_asr_model"
    if record.get("asr_decode_fingerprint") != eb.asr_decode_fingerprint():
        return None, "unknown_or_stale_asr_decode"
    hypothesis = record.get("asr_hypo")
    if not isinstance(hypothesis, str) or not hypothesis:
        return None, "missing_asr_hypo"
    return {"gen_text": gen_text, "asr_hypo": hypothesis}, "migratable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace provenance-safe legacy sidecars (default: dry run)",
    )
    args = parser.parse_args()

    pairs = list(
        eval_cloned.find_all_cloned(args.out_dir, allow_partial=args.allow_partial)
    )
    counts: Counter[str] = Counter()
    evaluated_at = datetime.now().isoformat()
    for wav_path, json_path in pairs:
        if eval_cloned.read_valid_eval(wav_path, json_path) is not None:
            counts["already_v4"] += 1
            continue
        source, status = read_legacy_source(wav_path, json_path)
        if source is None:
            counts[status] += 1
            continue
        item = eb.build_eval_item(wav_path, json_path, source["asr_hypo"])
        record = eval_cloned.build_eval_record(item, evaluated_at)
        if args.write:
            eval_cloned.write_eval_json(json_path, record)
            counts["written_v4"] += 1
        else:
            counts["dry_run_migratable"] += 1

    print(json.dumps({"items_total": len(pairs), **counts}, indent=2, sort_keys=True))
    if not args.write:
        print("Dry run only; pass --write to atomically replace safe legacy sidecars.")


if __name__ == "__main__":
    main()
