# TN v4 Conservative Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close deterministic TN coverage gaps without introducing fuzzy, phonetic, or reference-guided correction.

**Architecture:** Keep one pure-stdlib normalization module and two independent CER paths: reference normalization may consume authoring metadata such as language and speech tags, while hypothesis normalization never inspects the reference. Canonical entity tokens preserve semantic type; normalization source/version, reference context, and ASR provenance are bound into CER v4 sidecars so stale results fail closed while valid ASR hypotheses remain reusable.

**Tech Stack:** Python 3.10 stdlib, jiwer, standalone assertion checks, CER v4 sidecars.

---

### Task 1: Expand executable rule specification

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/check_cer_normalization.py`

**Steps:**
1. Add expected-equivalence cases for speech tags, safe HTML entities, English cardinal/decimal numbers, Chinese/Arabic ordinals, explicit fractions, leading-zero identifiers, and unambiguous `八点零五` time.
2. Add non-equivalence assertions across `NUM`, `DIGITS`, `ORDINAL`, `FRACTION`, `PERCENT`, `MONEY`, and `QTY`.
3. Add dangerous counterexamples for lexical “second”, malformed entities, version-like strings, dates without an unambiguous year, fillers, homophones, repetitions, and ambiguous Chinese decimals.
4. Require idempotence for every new canonical form.

### Task 2: Implement conservative TN v4

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/cer_normalization.py`
- Modify: `batch_generate_text_and_clone/text_generation/text_tn.py`

**Steps:**
1. Bump safe profile and normalization version; extend protected canonical token types.
2. Add controlled semicolon-terminated HTML entity decoding before width/case cleanup.
3. Centralize deterministic speech-tag replacement and reuse it from `text_tn.py`.
4. Add pure-stdlib English cardinal/decimal parsing and explicit ordinal/fraction grammars with strict token boundaries; keep ambiguous `first`/`second` usages unchanged.
5. Add explicit Arabic/Chinese ordinal and fraction rules.
6. Preserve leading zeros as `DIGITS` instead of collapsing them into `NUM`.
7. Add only unambiguous no-`分` time forms with a leading-zero minute; retain `三点一四` unchanged.
8. Preserve all existing fillers, particles, rhotic suffixes, repetitions, homophones, contractions, and malformed entities.

### Task 3: Bind reference context into CER v4

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/eval_batch_200.py`
- Modify: `batch_generate_text_and_clone/eval_cer/eval_cloned.py`

**Steps:**
1. Pass clone `language` and `lang_type` only to reference normalization.
2. Store normalized reference context in each eval record.
3. Validate that context when reusing CER sidecars, while allowing ASR hypothesis reuse across TN-only changes.
4. Keep migration and batch callers source-compatible through default keyword arguments.

### Task 4: Document and verify

**Files:**
- Modify: `batch_generate_text_and_clone/eval_cer/README.md`
- Modify: `batch_generate_text_and_clone/README.md`

**Steps:**
1. Document exact TN v4 order, typed tokens, reference-only tag handling, preserved ambiguities, and refresh behavior.
2. Run the standalone normalization checks; do not run pytest/ruff/mypy per repository instructions.
3. Run Python compile checks, CLI import/help smoke tests, `git diff --check`, and production-path searches for LLM usage.
4. Have independent reviewers audit rule safety, real-corpus relevance, and CER cache integration.
