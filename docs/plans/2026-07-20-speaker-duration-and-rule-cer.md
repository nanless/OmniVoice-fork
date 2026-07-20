# Speaker Duration Top-up and Deterministic CER Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a resumable clone/filter loop that gives every target speaker at least 1,800 accepted seconds across original and filtered clone audio, and replace LLM-assisted CER normalization with a deterministic, versioned rule engine.

**Architecture:** A read-only inventory/planner treats `audio/<dataset>/<speaker>` and one or more accepted-clone roots as authoritative duration sources, then writes immutable JSONL generation tasks only for speakers below target. `clone_dataset.py` consumes that plan deterministically. CER v4 normalizes reference and hypothesis independently through one pure rule module, binds results to rule/context/model/decode/audio fingerprints, and exposes one canonical `cer` field to all filters and reports.

**Tech Stack:** Python 3.10+, stdlib, soundfile, PyTorch/OmniVoice, Qwen3-ASR, jiwer, Bash.

---

### Task 1: Speaker-duration inventory and top-up plan

**Files:**
- Create: `batch_generate_text_and_clone/voice_clone/plan_speaker_topup.py`
- Create: `batch_generate_text_and_clone/voice_clone/check_speaker_target.py`

**Steps:**
1. Scan the five original dataset directories and require each WAV to have a valid `.wav.json` containing matching `dataset`, `speaker_id`, `audio_path`, and `transcript`.
2. Scan accepted clone roots by `<dataset>/<speaker>/...`, validate WAV headers, and count only accepted audio.
3. Aggregate exact frames/sample-rate duration under stable `dataset/speaker_id` keys.
4. Compute `max(0, 1800 - original_seconds - accepted_clone_seconds)` per speaker.
5. Rank references deterministically, preferring 3–15 second original recordings with non-empty transcripts.
6. Assign deterministic, diverse generated texts and emit immutable task IDs, expected output paths, reference fields, speaker key, and planning assumptions.
7. Write JSON summary/CSV and fail if any accepted clone cannot map to a target speaker.
8. Smoke-test on a temporary synthetic tree and run a full read-only inventory against the real merged dataset.

### Task 2: Plan-driven, resumable cloning

**Files:**
- Modify: `batch_generate_text_and_clone/voice_clone/clone_dataset.py`
- Modify: `batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh`
- Modify: `batch_generate_text_and_clone/voice_clone/README.md`

**Steps:**
1. Add `--plan-jsonl`; reject dataset-scan-only flags that conflict with a plan.
2. Validate unique task IDs/output paths and stable modulo worker sharding.
3. Write clone sidecar schema v3 with `speaker_key`, `plan_id`, `task_id`, source signatures, generation config, and output signature.
4. Preserve atomic output, retry limits, resume, corruption detection, and dry-run counts.
5. Ensure a task is never silently considered complete when its plan identity changes.
6. Add launcher support for `CLONE_PLAN_JSONL` and planned output root.

### Task 3: Deterministic text normalization

**Files:**
- Create: `batch_generate_text_and_clone/eval_cer/cer_normalization.py`
- Create: `batch_generate_text_and_clone/eval_cer/check_cer_normalization.py`
- Modify: `batch_generate_text_and_clone/text_generation/text_tn.py`

**Steps:**
1. Implement NFC plus explicit width folding, control/zero-width removal, HTML entity handling, case folding, punctuation and CJK/Latin spacing rules.
2. Implement known speech-tag handling without comparing ref and hyp.
3. Implement self-contained Chinese/English cardinal, digit-sequence, decimal, ordinal, percentage, fraction, date, time, currency and unit normalization.
4. Preserve ambiguous lexical uses with explicit exception/context rules; do not use semantic/homophone correction.
5. Define empty-reference behavior and make normalization errors explicit.
6. Compute a stable normalization fingerprint from versioned rules/configuration.
7. Cover examples, adversarial cases and idempotence in a standalone check script.

### Task 4: CER schema v4, with no LLM path

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/eval_batch_200.py`
- Modify: `batch_generate_text_and_clone/eval_cer/eval_cloned.py`
- Create: `batch_generate_text_and_clone/eval_cer/migrate_cer_to_v4.py`
- Modify: `batch_generate_text_and_clone/run_eval_all.sh`

**Steps:**
1. Delete CER LLM prompts, HTTP clients, endpoint configuration, caches, concurrency and ground-truth-guided hypothesis selection.
2. Write only `deterministic_char_cer` schema v4 fields: normalized ref/hyp, `cer`, edit counts, reference context/input and ASR/audio/text/rule fingerprints.
3. Make canonical details/reports reject mixed or stale schemas.
4. Add `--refresh-cer` to reuse valid ASR hypotheses and recompute only rule normalization/CER.
5. Add a CPU-only v2 migration that ignores every LLM field and recalculates from `gen_text` and `asr_hypo`.
6. Remove LLM CER environment variables and flags from the launcher.

### Task 5: Downstream metric contract migration

**Files:**
- Modify: `batch_generate_text_and_clone/filter_cloned.py`
- Modify: `batch_generate_text_and_clone/prune_and_copy.py`
- Modify: `batch_generate_text_and_clone/analyze_distributions.py`
- Modify: `batch_generate_text_and_clone/eval_cer/fix_paths.py`
- Modify: `batch_generate_text_and_clone/eval_cer/README.md`
- Modify: `batch_generate_text_and_clone/README.md`

**Steps:**
1. Consume only canonical `cer` with metric/version/fingerprint validation.
2. Remove `--use-llm-cer`, `llm_cer`, LLM comparison outputs and LLM cache path maintenance.
3. Keep raw cosine SIM semantics and default exclusive threshold `> 0.8`.
4. Document the closed loop: plan → clone → CER/SIM/MOS → filter/copy → target check → next round.

### Task 6: Verification

**Steps:**
1. Run Python AST/compile checks and standalone rule/planner checks; do not run pytest/ruff/mypy.
2. Run `bash -n` on modified launchers.
3. Run `git diff --check` and inspect all existing dirty changes for overlap.
4. Run real full inventory/plan dry-run without loading OmniVoice or changing audio data.
5. Verify no CER production script references an LLM endpoint, prompt, cache or `llm_cer`.
6. Verify the target checker reports exact remaining speaker deficits and exits non-zero until all speakers reach 1,800 accepted seconds.
