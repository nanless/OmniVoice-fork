# Clone and Filter Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the fork-added clone, CER, raw-cosine similarity, and filtering pipeline safely resumable and runnable on the 8-GPU development host.

**Architecture:** Keep OmniVoice core untouched. Harden only `batch_generate_text_and_clone/**`: model outputs are committed atomically, resume decisions are based on validated status rather than file existence, evaluation aggregates are rebuilt from authoritative sidecars, and launchers expose paths and GPU topology through CLI/environment configuration.

**Tech Stack:** Python 3.10, PyTorch 2.8/CUDA 12.8, Bash, JSON/JSONL sidecars, uv lockfile.

---

### Task 1: Clone resume state machine

**Files:**
- Modify: `batch_generate_text_and_clone/voice_clone/clone_dataset.py`

**Step 1:** Add a pure helper that classifies an output pair as `complete`, `retry`, or `missing`.

**Step 2:** Verify malformed JSON, `status=failed`, WAV-only, and JSON-only cases are not complete.

**Step 3:** Skip only when WAV exists and JSON parses with `status=generated` and matching `cloned_audio`.

**Step 4:** Run the helper against temporary fixtures and confirm all state transitions.

### Task 2: Atomic clone outputs and CLI configuration

**Files:**
- Modify: `batch_generate_text_and_clone/voice_clone/clone_dataset.py`

**Step 1:** Generate audio into a same-directory temporary `.wav`.

**Step 2:** Replace the final WAV only after successful generation and soundfile write.

**Step 3:** Write JSON through a same-directory temporary file and atomic replace.

**Step 4:** Add `--out-dir`, `--model-path`, repeatable `--dataset`, and environment-backed defaults.

**Step 5:** Ensure `--dry-run` never loads OmniVoice or initializes CUDA.

**Step 6:** Run `py_compile`, CLI help, dry-run, and failure/retry fixture checks.

### Task 3: Generalized 8-GPU launcher

**Files:**
- Modify: `batch_generate_text_and_clone/voice_clone/run_clone_8workers.sh`

**Step 1:** Parse `GPUS`, defaulting to `0,1,2,3,4,5,6,7`.

**Step 2:** Parse `WORKERS_PER_GPU`, defaulting to one safe model process per GPU.

**Step 3:** Derive stable global worker IDs and total worker count.

**Step 4:** Forward model, text, and output paths explicitly.

**Step 5:** Run `bash -n` and a non-launching configuration validation mode.

### Task 4: Lazy LLM endpoint and atomic CER persistence

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/eval_batch_200.py`
- Modify: `batch_generate_text_and_clone/eval_cer/eval_cloned.py`

**Step 1:** Remove import-time endpoint initialization.

**Step 2:** Initialize endpoints under the existing lock only when an LLM request is made.

**Step 3:** Verify importing `eval_cloned.py` with no endpoint environment succeeds.

**Step 4:** Replace direct cache and eval-sidecar writes with atomic JSON writes.

**Step 5:** Write in-progress JSONL to a `.current` file.

**Step 6:** Rebuild canonical `eval_cer_details.jsonl` atomically from all valid `.eval.json` sidecars at successful completion and when all entries are already cached.

### Task 5: Evaluation orchestration

**Files:**
- Modify: `batch_generate_text_and_clone/run_eval_all.sh`

**Step 1:** Default `CLONED_VOICES_ROOT` to the clone pipeline output directory.

**Step 2:** Default `PARALLEL_SIM=0` so CER and SIM do not contend for one GPU.

**Step 3:** Preserve explicit parallel mode for users who assign disjoint resources.

**Step 4:** Run `bash -n` and inspect the generated tmux command environment.

### Task 6: Reproducible environment

**Files:**
- Do not modify the shared Conda environment.
- Create ignored repository environment: `.venv/`

**Step 1:** Run `/root/.local/bin/uv sync --frozen`.

**Step 2:** Confirm `.venv/bin/python` reports locked torch and transformers versions.

**Step 3:** Confirm `import omnivoice` succeeds without the incompatible shared torchvision.

**Step 4:** Point the clone launcher default Python to `.venv/bin/python`, retaining an explicit `PYTHON` override.

### Task 7: Verification and handoff

**Files:**
- Verify all modified Python and Bash files.

**Step 1:** Run Python syntax compilation; do not run pytest/ruff/mypy because the repository has no configuration.

**Step 2:** Run clone dry-run on a limited partition without model loading.

**Step 3:** Run one generated-audio smoke item only if the target GPUs are idle.

**Step 4:** Run SIM mathematical/cache contract checks and CER import-without-endpoint check.

**Step 5:** Run filter in a temporary output destination; do not execute destructive prune.

**Step 6:** Review `git diff --check`, the complete changed-file list, and include new untracked contract/plan files in the handoff.
