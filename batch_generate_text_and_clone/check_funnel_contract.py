#!/usr/bin/env python3
"""Fast, model-free regression checks for the SIM -> CER funnel contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval_common import (
    load_inventory_wav_list,
    validate_sim_selection_manifest,
    wav_list_fingerprint,
)
from filter_cloned import filter_and_save


def write_manifest(root: Path, selection: Path, selected: list[Path]) -> None:
    stat = selection.stat()
    manifest = {
        "schema_version": 2,
        "stage": "complete",
        "selection_kind": "sim_threshold",
        "out_dir": str(root.resolve()),
        "output": str(selection.resolve()),
        "output_signature": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "inventory_count": 2,
        "matched_count": len(selected),
        "wav_list_fingerprint": wav_list_fingerprint(selected),
        "similarity_metric": "raw_cosine",
        "similarity_operator": ">",
        "min_raw_cosine_similarity": 0.8,
    }
    selection.with_suffix(selection.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "inventory"
        root.mkdir()
        wav_a = root / "a.wav"
        wav_b = root / "b.wav"
        wav_a.write_bytes(b"a")
        wav_b.write_bytes(b"b")
        selection = root / "sim_gt0.8.txt"
        selection.write_text(f"{wav_a.resolve()}\n", encoding="utf-8")
        selected, fingerprint = load_inventory_wav_list(
            selection, root, [wav_a, wav_b]
        )
        write_manifest(root, selection, selected)
        validate_sim_selection_manifest(
            selection, root, selected, fingerprint, inventory_count=2
        )

        selection.write_text(
            f"{wav_a.resolve()}\n{wav_a.resolve()}\n", encoding="utf-8"
        )
        try:
            load_inventory_wav_list(selection, root, [wav_a, wav_b])
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate allowlist path was accepted")

        selection.write_text(f"{Path(tmp).resolve() / 'outside.wav'}\n", encoding="utf-8")
        try:
            load_inventory_wav_list(selection, root, [wav_a, wav_b])
        except ValueError as exc:
            assert "outside" in str(exc) or "missing" in str(exc)
        else:
            raise AssertionError("out-of-root allowlist path was accepted")

        output = root / "final.txt"
        inventory = {str(wav_a.resolve()), str(wav_b.resolve())}
        cer = {
            str(wav_a.resolve()): {"cer": 0.1},
            str(wav_b.resolve()): {"cer": 0.09999999999999999},
        }
        sim = {
            str(wav_a.resolve()): {"similarity": 0.8000000000000002},
            str(wav_b.resolve()): {"similarity": 0.8},
        }
        matched, _, _ = filter_and_save(
            cer, sim, 0.1, 0.8, output, str(root), inventory, {}
        )
        assert matched == [], "threshold equality must fail for both CER and SIM"

        sim[str(wav_b.resolve())]["similarity"] = 0.8000000000000002
        matched, _, _ = filter_and_save(
            cer, sim, 0.1, 0.8, output, str(root), inventory, {}
        )
        assert matched == [str(wav_b.resolve())]

    print("SIM -> CER funnel contract checks passed")


if __name__ == "__main__":
    main()
