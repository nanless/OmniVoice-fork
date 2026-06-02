#!/usr/bin/env python3
"""Batch eval on fixed cloned audio samples using Qwen3-ASR.

No VAD — the full cloned audio is sent directly to ASR.

Steps:
1. Load fixed sample list (eval_sample_{N}.json)
2. ASR with Qwen3-ASR (batch processing, full audio)
3. Stage 1: Manual ITN (baseline; normalize whitespace, keep word spaces)
4. Stage 2: LLM ITN on manual-preprocessed text (continue ITN + cross-align)
5. CER comparison
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import socket
import string
import sys
import time
import threading
import urllib.error
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent
ENV_FILE = EVAL_DIR / ".env"
QWEN3_ASR_LOCAL = "/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local"
TAGS_PY = EVAL_DIR.parent / "text_generation" / "llm_generate_texts.py"
OUT_DIR = Path("/root/group-shared/voiceprint/data/speech/voice_activity_detection/batch_cloned_voices_ommivoice_kids_finetuned")
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SAMPLE_SEED = 42


def eval_paths(sample_size: int) -> dict[str, Path]:
    n = sample_size
    return {
        "sample_list": EVAL_DIR / f"eval_sample_{n}.json",
        "asr_cache": OUT_DIR / f"eval_asr_cache_{n}.json",
        "llm_cache": OUT_DIR / f"eval_llm_itn_cache_{n}.json",
        "summary": OUT_DIR / f"eval_summary_{n}.json",
        "details_manual": OUT_DIR / f"eval_details_{n}_manual.txt",
        "details_llm": OUT_DIR / f"eval_details_{n}_llm.txt",
        "comparison": OUT_DIR / f"eval_comparison_{n}.txt",
        "details_legacy": OUT_DIR / f"eval_details_{n}.txt",
    }


def eval_paths_full(out_dir: Path) -> dict[str, Path]:
    """Output paths for full-dataset eval (eval_cloned.py)."""
    return {
        "asr_cache": out_dir / "eval_asr_cache.json",
        "llm_cache": out_dir / "eval_llm_itn_cache.json",
        "summary": out_dir / "eval_summary.json",
        "summary_progress": out_dir / "eval_summary_progress.json",
        "details_jsonl": out_dir / "eval_cer_details.jsonl",
        "details_manual": out_dir / "eval_details_manual.txt",
        "details_llm": out_dir / "eval_details_llm.txt",
        "comparison": out_dir / "eval_comparison.txt",
        "details_legacy": out_dir / "eval_details.txt",
    }


def validate_asr_cache(pairs, asr_results: dict) -> int:
    """Return miss count; abort if cache has zero overlap with current pairs."""
    wav_keys = [str(w) for w, _ in pairs]
    hit = sum(1 for w in wav_keys if asr_results.get(w))
    miss = len(wav_keys) - hit
    if miss:
        print(
            f"WARNING: ASR miss/empty for {miss}/{len(wav_keys)} files "
            f"(cache entries: {len(asr_results)})"
        )
        if hit == 0:
            raise RuntimeError(
                "ASR cache keys do not match current wav list; "
                "re-run without --skip-asr"
            )
    return miss

PUNCTUATION = set(
    string.punctuation
    + "，。！？；：\"\"''（）【】《》、·～｀＠＃￥％＆＊—＋｜＜＞？／"
    + "\u2018\u2019\u201c\u201d"
    + "\u2026"
)

ITN_SYSTEM_PROMPT = """你是ASR评测ITN专家。对ref/hyp文本对做逆文本归一化+交叉对齐，消除虚假字错。

## 核心规则
1. **等价统一**：读音/语义相同仅写法不同→统一为ref侧形式，hyp跟随
2. **真差异不对齐**：不同名词/动词/数字/语义保留
3. **输出**：JSON数组 [{"id":0,"ref_final":"...","hypo_final":"..."}]，小写无标点保留空格

## 语义词标签处理（重要）
ref中可能有标签，ASR会识别出对应语气词。**ref有标签时，删除hyp中对应语气词**：
- [sigh]→哎/唉 | [question-ah]→啊 | [question-ei]→诶/哎 | [question-yi]→咦
- [surprise-ah]→啊 | [surprise-oh]→哦 | [surprise-wa]→哇 | [surprise-yo]→哟
- [dissatisfaction-hnn]→哼/嗯/hmm | [confirmation-en]→嗯 | [laughter]→哈哈

例：ref `第7题甲乙合作慢点慢点[question-yi]` hyp `第七题甲乙合作慢点慢点咦` → 删除`咦`

## 数字/单位
- 中文数词→阿拉伯：一百二十→120 | 逐字读数：三零二→302 | 英文：forty-five→45
- 百分数：百分之八十/80percent→80 | 分数：三分之七→3分之7
- 单位：km/h→kmh，centimeters→cm，3minutes→3min
- 保留小数点：12.5≠125 | 一样/一般/一起不转1

## 同音对齐
- 人名：芳芳/方方 | 词语：刻舟/勾舟 | 拼音↔汉字：hongjun→红军
- 公式：a²+b²=c²↔a块大c色块 | 英文变体：question1/questionone

## 示例
ref: `80%和120块wait` hyp: `八百分之和一百二十块weight` → `80和120块weight`
ref: `芳芳的blue shirt` hyp: `方方的blue shirt` → `芳芳的blue shirt`
ref: `call eight eight seven seven` hyp: `call 8877` → `call 8877`
ref: `第7题甲乙合作慢点慢点[question-yi]` hyp: `第七题甲乙合作慢点慢点咦` → 删除`咦`
ref: `i fell why[dissatisfaction-hnn]` hyp: `i fell why hmm` → 删除`hmm`
"""


def load_env_file(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


load_env_file(ENV_FILE)

# ── Multi-endpoint round-robin ──────────────────────────────────────
_LLM_ENDPOINTS = []
_LLM_ENDPOINT_LOCK = threading.Lock()
_LLM_ENDPOINT_IDX = 0


def _init_llm_endpoints():
    global _LLM_ENDPOINTS
    urls_str = os.environ.get("ITN_LLM_BASE_URLS", "")
    if urls_str:
        _LLM_ENDPOINTS = [u.strip().rstrip("/") for u in urls_str.split(",") if u.strip()]
    else:
        single = os.environ.get("ITN_LLM_BASE_URL", "")
        if single:
            _LLM_ENDPOINTS = [single.strip().rstrip("/")]
    if not _LLM_ENDPOINTS:
        raise RuntimeError(f"Set ITN_LLM_BASE_URLS or ITN_LLM_BASE_URL in {ENV_FILE}")


def _next_llm_endpoint() -> str:
    global _LLM_ENDPOINT_IDX
    with _LLM_ENDPOINT_LOCK:
        endpoint = _LLM_ENDPOINTS[_LLM_ENDPOINT_IDX % len(_LLM_ENDPOINTS)]
        _LLM_ENDPOINT_IDX += 1
    return endpoint


_init_llm_endpoints()

# ── load tag definitions ────────────────────────────────────────────
def load_tag_names(path):
    code = Path(path).read_text(encoding="utf-8")
    m = re.search(r"TAG_DEFINITIONS\s*=\s*\{(.*?)\n\}", code, re.DOTALL)
    if not m:
        return []
    block = m.group(0)
    return re.findall(r'"(\[.*?\])"\s*:', block)


TAG_NAMES = load_tag_names(TAGS_PY)
TAG_PATTERN = re.compile("(" + "|".join(re.escape(t) for t in TAG_NAMES) + ")")

# 标签到语气词的映射
TAG_TO_PARTICLE = {
    "[sigh]": "哎",
    "[question-ah]": "啊",
    "[laughter]": "哈哈",
    "[question-oh]": "哦",
    "[question-ei]": "诶",
    "[question-yi]": "咦",
    "[surprise-ah]": "啊",
    "[surprise-oh]": "哦",
    "[surprise-wa]": "哇",
    "[surprise-yo]": "哟",
    "[dissatisfaction-hnn]": "嗯",
    "[confirmation-en]": "嗯",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_speech(wav_path, target_sr=16000):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), orig_freq=sr, new_freq=target_sr
        ).numpy()
    return wav


def remove_tags(text):
    return TAG_PATTERN.sub("", text).strip()


def keep_tags(text):
    """保留标签，用于大模型ITN输入"""
    return text


def strip_punctuation(text):
    chars = list(text)
    out = []
    for i, ch in enumerate(chars):
        if (
            ch == '.'
            and i > 0
            and i + 1 < len(chars)
            and chars[i - 1].isdigit()
            and chars[i + 1].isdigit()
        ):
            out.append(ch)
        elif ch not in PUNCTUATION:
            out.append(ch)
    return ''.join(out)


def normalize_spaces(text: str) -> str:
    """Collapse runs of whitespace to single spaces; keep word boundaries."""
    return re.sub(r'\s+', ' ', text).strip()


def finalize_cer_text(text: str) -> str:
    """Final whitespace normalization for CER (keep spaces, do not strip all)."""
    return normalize_spaces(text)


# ── manual number normalization ─────────────────────────────────────
_CN_DIGIT = {
    '零': '0', '〇': '0',
    '一': '1', '二': '2', '两': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
}
_CN_NUM = {
    '零': 0, '〇': 0,
    '一': 1, '壹': 1,
    '二': 2, '两': 2, '贰': 2,
    '三': 3, '叁': 3,
    '四': 4, '肆': 4,
    '五': 5, '伍': 5,
    '六': 6, '陆': 6,
    '七': 7, '柒': 7,
    '八': 8, '捌': 8,
    '九': 9, '玖': 9,
    '十': 10, '拾': 10,
    '百': 100, '佰': 100,
    '千': 1000, '仟': 1000,
    '万': 10000,
}
_CN_SUFFIXES = (
    '单元', '号楼', '毫米', '厘米', '公里', '小时', '分钟',
    '块', '毛', '元', '个', '岁', '分', '秒', '年', '月', '日', '号',
    '层', '页', '名', '倍', '点', '支', '次', '遍', '顿', '米',
)


def _parse_chinese_number(s: str):
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if len(s) == 1 and s in _CN_NUM:
        v = _CN_NUM[s]
        return v if v < 10 else None
    total = 0
    current = 0
    for ch in s:
        if ch not in _CN_NUM:
            return None
        v = _CN_NUM[ch]
        if v >= 10:
            if current == 0:
                current = 1
            total += current * v
            current = 0
        else:
            current = current * 10 + v if current else v
    total += current
    return total


def _cn_to_str(s: str) -> str:
    if s.isdigit():
        return s
    val = _parse_chinese_number(s)
    return str(val) if val is not None else s


from word2number import w2n

_EN_NUM_WORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'billion', 'and', 'point',
}
_EN_DIGIT_WORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
}
_EN_NUM_WORDS_LIST = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'billion',
]
_EN_NUM_PATTERN = '|'.join(_EN_NUM_WORDS_LIST)
_EN_DIGIT_PATTERN = '|'.join(sorted(_EN_DIGIT_WORDS, key=len, reverse=True))
_EN_DIGIT_SEQ_RE = re.compile(
    rf'\b(?:{_EN_DIGIT_PATTERN})(?:[\s,\-]+(?:{_EN_DIGIT_PATTERN}))+\b',
    re.IGNORECASE,
)
_EN_DECIMAL_RE = re.compile(
    rf'\b({_EN_NUM_PATTERN}(?:[\s\-]+{_EN_NUM_PATTERN})*)\s+point\s+'
    rf'((?:\b(?:{_EN_NUM_PATTERN})\b[\s\-]*)+)',
    re.IGNORECASE,
)
_EN_COMPOUND_RE = re.compile(
    rf'\b(?:{_EN_NUM_PATTERN})(?:(?:[\s\-]+(?:and[\s\-]+)?(?:{_EN_NUM_PATTERN}))+)(?![a-zA-Z])',
    re.IGNORECASE,
)
_EN_SINGLE_RE = re.compile(rf'\b({_EN_NUM_PATTERN})\b', re.IGNORECASE)


def _convert_english_number_phrase(phrase: str) -> str:
    phrase_clean = phrase.replace('-', ' ').lower()
    words = phrase_clean.split()
    filtered = []
    for i, w in enumerate(words):
        if w in _EN_NUM_WORDS:
            filtered.append(w)
        elif w == 'and' and 0 < i < len(words) - 1:
            if words[i - 1] in _EN_NUM_WORDS and words[i + 1] in _EN_NUM_WORDS:
                filtered.append(w)
    if not filtered:
        return phrase
    try:
        return str(w2n.word_to_num(' '.join(filtered)))
    except Exception:
        return phrase


def normalize_percent(text: str) -> str:
    text = re.sub(
        r'([零〇一二三四五六七八])百分之',
        lambda m: _cn_to_str(m.group(1)) + '0',
        text,
    )
    text = re.sub(
        r'百分之([零〇一二两三四五六七八九十百千万亿]+)',
        lambda m: _cn_to_str(m.group(1)),
        text,
    )
    text = re.sub(r'百分之(\d+(?:\.\d+)?)', r'\1', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1', text)
    text = re.sub(
        rf'\b({_EN_NUM_PATTERN}(?:[\s\-]+{_EN_NUM_PATTERN})*)\s+percent\b',
        lambda m: _convert_english_number_phrase(m.group(1)),
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace('百分之', '')
    text = re.sub(r'\bpercent\b', '', text, flags=re.IGNORECASE)
    return text


def normalize_fractions(text: str) -> str:
    def _repl(m):
        return f"{_cn_to_str(m.group(1))}分之{_cn_to_str(m.group(2))}"
    return re.sub(
        r'([零〇一二两三四五六七八九十百千万\d]+)分之([零〇一二两三四五六七八九十百千万\d]+)',
        _repl,
        text,
    )


def normalize_chinese_money(text: str) -> str:
    def _repl(m):
        return f"{_cn_to_str(m.group(1))}块{_cn_to_str(m.group(2))}"
    return re.sub(
        r'([零〇一二两三四五六七八九十\d])块([零〇一二两三四五六七八九十\d])',
        _repl,
        text,
    )


def normalize_units(text: str) -> str:
    text = re.sub(
        r'\b(?:kilometers?|kms?)\s*(?:per|/)\s*(?:hour|hr|h)\b',
        'kmh',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'\bkm/h\b', 'kmh', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcentimeters?\b', 'cm', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcentimetres?\b', 'cm', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmillimeters?\b', 'mm', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d)\s+kmh\b', r'\1kmh', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*min(?:ute)?s?\b', r'\1min', text, flags=re.IGNORECASE)
    return text


def normalize_english_digit_sequences(text: str) -> str:
    def _repl(m):
        digits = []
        for w in re.split(r'[\s,\-]+', m.group(0)):
            w = w.lower()
            if w in _EN_DIGIT_WORDS:
                digits.append(str(w2n.word_to_num(w)))
            else:
                return m.group(0)
        return ''.join(digits)
    return _EN_DIGIT_SEQ_RE.sub(_repl, text)


def normalize_english_numbers(text: str) -> str:
    text = normalize_english_digit_sequences(text)

    def _replace_decimal(m):
        try:
            int_val = w2n.word_to_num(m.group(1).replace('-', ' ').lower())
        except Exception:
            return m.group(0)
        frac_digits = []
        for w in re.findall(rf'\b({_EN_NUM_PATTERN})\b', m.group(2), re.IGNORECASE):
            w = w.lower()
            if w in _EN_DIGIT_WORDS:
                frac_digits.append(str(w2n.word_to_num(w)))
            else:
                break
        if not frac_digits:
            return m.group(0)
        return f"{int_val}.{''.join(frac_digits)}"

    text = _EN_DECIMAL_RE.sub(_replace_decimal, text)
    for _ in range(4):
        prev = text
        text = _EN_COMPOUND_RE.sub(
            lambda m: _convert_english_number_phrase(m.group(0)), text
        )
        text = _EN_SINGLE_RE.sub(
            lambda m: _convert_english_number_phrase(m.group(1)), text
        )
        if text == prev:
            break
    return text


_CN_SUFFIX_PATTERN = '|'.join(re.escape(s) for s in _CN_SUFFIXES)
_CN_COMPOUND_RE = re.compile(
    rf'[零〇一二两贰三四五六七八九十拾百千仟万]{{2,}}(?:{_CN_SUFFIX_PATTERN})?'
    rf'|[一二两三四五六七八九](?:{_CN_SUFFIX_PATTERN})'
)
_CN_DIGIT_RUN_RE = re.compile(r'[零〇一二三四五六七八九]{2,}')


def _split_cn_num_suffix(full: str):
    for suffix in sorted(_CN_SUFFIXES, key=len, reverse=True):
        if full.endswith(suffix):
            return full[:-len(suffix)], suffix
    return full, ''


def normalize_chinese_numbers(text: str) -> str:
    def _repl(m):
        full = m.group(0)
        num_str, unit = _split_cn_num_suffix(full)
        val = _parse_chinese_number(num_str)
        if val is not None:
            return str(val) + unit
        return m.group(0)
    return _CN_COMPOUND_RE.sub(_repl, text)


def normalize_chinese_digit_runs(text: str) -> str:
    def _repl(m):
        s = m.group(0)
        if any(c in '十百千万亿' for c in s):
            return s
        return ''.join(_CN_DIGIT.get(c, c) for c in s)
    return _CN_DIGIT_RUN_RE.sub(_repl, text)


def normalize_numbers(text: str) -> str:
    text = normalize_percent(text)
    text = normalize_fractions(text)
    text = normalize_chinese_money(text)
    text = normalize_units(text)
    text = normalize_english_numbers(text)
    text = normalize_chinese_numbers(text)
    text = normalize_chinese_digit_runs(text)
    text = normalize_units(text)
    text = re.sub(r'(\d)\s+kmh\b', r'\1kmh', text, flags=re.IGNORECASE)
    return text


def manual_itn_preprocess(text: str, keep_tag: bool = False) -> str:
    """Stage-1 ITN (LLM input and manual baseline).
    
    Args:
        keep_tag: True保留标签用于大模型ITN，False删除标签用于CER计算
    """
    processed = text if keep_tag else remove_tags(text)
    return finalize_cer_text(strip_punctuation(normalize_numbers(processed)).lower())


def manual_itn(text: str) -> str:
    """Stage-1 ITN for CER baseline."""
    return manual_itn_preprocess(text)


_CJK_CHAR = r'[\u4e00-\u9fff]'


def cleanup_llm_spacing(hyp: str) -> str:
    """Remove spurious spaces between CJK and Latin/digits (哎 we → 哎we)."""
    if not hyp:
        return hyp
    hyp = re.sub(rf'({_CJK_CHAR})\s+([a-zA-Z0-9])', r'\1\2', hyp)
    hyp = re.sub(rf'([a-zA-Z0-9])\s+({_CJK_CHAR})', r'\1\2', hyp)
    return hyp


def llm_itn_postprocess(ref_llm: str, hyp_llm: str, ref_manual: str, hyp_manual: str) -> tuple[str, str]:
    """Pick best hyp under fixed manual ref; never regress below manual CER."""
    ref_f = ref_manual
    if not hyp_llm:
        return ref_f, hyp_manual

    hyp_llm_clean = cleanup_llm_spacing(finalize_cer_text(hyp_llm))
    if calc_cer(ref_f, hyp_llm_clean)[0] <= calc_cer(ref_f, hyp_manual)[0] + 1e-9:
        return ref_f, hyp_llm_clean
    return ref_f, hyp_manual


def calc_cer(ref: str, hyp: str):
    from jiwer import process_characters
    try:
        cm = process_characters(ref, hyp)
        return cm.cer, cm.substitutions, cm.insertions, cm.deletions, len(ref)
    except Exception:
        return 1.0, 0, 0, 0, len(ref)


def load_fixed_sample(
    sample_size: int,
    sample_list_path: Path,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[tuple[Path, Path]]:
    if sample_list_path.exists():
        data = load_json(sample_list_path)
        paths = data.get("wav_paths", [])
        print(f"Loaded fixed sample list: {len(paths)} audios from {sample_list_path}")
    else:
        all_pairs = []
        for wav_path in sorted(OUT_DIR.rglob("text_*.wav")):
            json_path = wav_path.with_suffix(".json")
            if json_path.exists():
                all_pairs.append(str(wav_path))
        random.seed(seed)
        paths = random.sample(all_pairs, min(sample_size, len(all_pairs)))
        paths = sorted(paths)
        sample_list_path.write_text(
            json.dumps({
                "seed": seed,
                "sample_size": len(paths),
                "created_at": datetime.now().isoformat(),
                "wav_paths": paths,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Created fixed sample list: {len(paths)} audios → {sample_list_path}")

    pairs = []
    for wav_str in paths:
        wav_path = Path(wav_str)
        json_path = wav_path.with_suffix(".json")
        if wav_path.exists() and json_path.exists():
            pairs.append((wav_path, json_path))
        else:
            print(f"Warning: missing {wav_path}")
    return pairs


def build_eval_item(wav_path: Path, json_path: Path, hypo_raw: str) -> dict:
    """Build one eval record from wav/json sidecar and ASR hypothesis."""
    meta = load_json(json_path)
    truth_raw = meta.get("gen_text", "")

    ref_manual = manual_itn(truth_raw)
    hyp_manual = manual_itn(hypo_raw)
    # 保留标签版本，用于大模型ITN输入
    ref_manual_prep = manual_itn_preprocess(truth_raw, keep_tag=True)
    hyp_manual_prep = manual_itn_preprocess(hypo_raw, keep_tag=False)
    manual_cer, csub, cins, cdel, cnum = calc_cer(ref_manual, hyp_manual)

    return {
        "wav": str(wav_path),
        "json": str(json_path),
        "name": wav_path.name,
        "ref_start": truth_raw,
        "hypo_start": hypo_raw,
        "ref_no_tag": remove_tags(truth_raw),
        "hypo_no_tag": remove_tags(hypo_raw),
        "ref_raw": truth_raw,
        "hypo_raw": hypo_raw,
        "ref_manual_prep": ref_manual_prep,
        "hypo_manual_prep": hyp_manual_prep,
        "ref_manual": ref_manual,
        "hypo_manual": hyp_manual,
        "manual_cer": manual_cer,
        "substitutions": csub,
        "insertions": cins,
        "deletions": cdel,
        "chars": cnum,
        "ref_audio": meta.get("ref_audio"),
    }


def run_asr(
    sampled,
    use_cache: bool,
    cache_path: Path,
    on_batch_done=None,
    batch_size: int = 16,
) -> dict:
    """Run ASR; optionally call on_batch_done(batch_pairs, results) and flush cache each batch."""
    results = load_json(cache_path) if cache_path.exists() else {}
    if results:
        print(f"Loaded ASR cache: {len(results)} entries from {cache_path}", flush=True)

    if use_cache:
        missing = [(w, j) for w, j in sampled if not results.get(str(w))]
        if not missing:
            print(f"ASR cache covers all {len(sampled)} items.", flush=True)
            return results
        print(
            f"ASR cache partial: {len(sampled) - len(missing)}/{len(sampled)} hit, "
            f"running ASR on {len(missing)} remaining",
            flush=True,
        )
        sampled = missing
    elif results:
        sampled = [(w, j) for w, j in sampled if not results.get(str(w))]
        if not sampled:
            print("All items already in ASR cache.", flush=True)
            return results

    def _flush_cache():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if not sampled:
        return results

    print("Loading Qwen3-ASR model...", flush=True)
    sys.path.insert(0, "/root/code/github_repos/Qwen3-ASR")
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = Qwen3ASRModel.from_pretrained(
        QWEN3_ASR_LOCAL,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )
    print(f"Model loaded (ASR batch_size={batch_size}).\n", flush=True)

    results = dict(results)
    for i in tqdm(range(0, len(sampled), batch_size), desc="ASR"):
        batch = sampled[i:i + batch_size]
        audio_inputs = []
        valid_paths = []
        for wav_path, _ in batch:
            try:
                speech = extract_speech(wav_path)
                audio_inputs.append((speech, 16000))
                valid_paths.append(wav_path)
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                results[str(wav_path)] = ""

        if not audio_inputs:
            continue

        try:
            hypos = asr.transcribe(
                audio=audio_inputs,
                language="Chinese",
                return_time_stamps=False,
            )
            for wav_path, h in zip(valid_paths, hypos):
                results[str(wav_path)] = h.text
        except Exception as e:
            print(f"ASR batch error: {e}", flush=True)
            for wav_path in valid_paths:
                results[str(wav_path)] = ""

        if on_batch_done is not None:
            on_batch_done(batch, results)
        _flush_cache()

    del asr
    torch.cuda.empty_cache()

    _flush_cache()
    print(f"ASR cache saved to {cache_path} ({len(results)} entries)", flush=True)
    return results


def transcribe_asr_batch(asr, batch_pairs: list, results: dict) -> None:
    """Run Qwen3-ASR on one batch; merge hypos into results dict."""
    audio_inputs = []
    valid_paths = []
    for wav_path, _ in batch_pairs:
        try:
            speech = extract_speech(wav_path)
            audio_inputs.append((speech, 16000))
            valid_paths.append(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}", flush=True)
            results[str(wav_path)] = ""

    if not audio_inputs:
        return

    try:
        hypos = asr.transcribe(
            audio=audio_inputs,
            language="Chinese",
            return_time_stamps=False,
        )
        for wav_path, h in zip(valid_paths, hypos):
            results[str(wav_path)] = h.text
    except Exception as e:
        print(f"ASR batch error: {e}", flush=True)
        for wav_path in valid_paths:
            results[str(wav_path)] = ""


def load_asr_model(batch_size: int = 16, gpu_id: int = 0):
    """Load Qwen3-ASR model."""
    print("Loading Qwen3-ASR model...", flush=True)
    sys.path.insert(0, "/root/code/github_repos/Qwen3-ASR")
    from qwen_asr import Qwen3ASRModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    asr = Qwen3ASRModel.from_pretrained(
        QWEN3_ASR_LOCAL,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )
    print(f"Model loaded (ASR batch_size={batch_size}, gpu={gpu_id}).\n", flush=True)
    return asr


def llm_itn_batch_fetch(batch: list[dict]) -> dict[str, dict]:
    """Call LLM ITN API for one batch; return wav -> {ref_final, hypo_final}."""
    if not batch:
        return {}
    batch_input = [
        {"id": j, "ref": it["ref_manual_prep"], "hypo": it["hypo_manual_prep"]}
        for j, it in enumerate(batch)
    ]
    raw = call_llm_itn_batch(batch_input)
    by_id = {r["id"]: r for r in raw}
    entries = {}
    for j, it in enumerate(batch):
        r = by_id.get(j, {})
        ref_f, hyp_f = llm_itn_postprocess(
            r.get("ref_final", ""),
            r.get("hypo_final", ""),
            it["ref_manual"],
            it["hypo_manual"],
        )
        entries[it["wav"]] = {"ref_final": ref_f, "hypo_final": hyp_f}
    return entries


def llm_itn_one_batch(
    batch: list[dict],
    cache: dict,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, dict]:
    """Run LLM ITN synchronously on one batch; update cache and optional cache file."""
    pending = [it for it in batch if not (use_cache and it["wav"] in cache)]
    if pending:
        cache.update(llm_itn_batch_fetch(pending))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return {it["wav"]: cache[it["wav"]] for it in batch if it["wav"] in cache}


def _extract_json_array(raw_text: str) -> list:
    text = raw_text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0].strip()
            break
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return data


def call_llm_itn_batch(batch_items: list[dict], max_retries: int = 6, endpoint: str = None) -> list[dict]:
    from openai import OpenAI
    
    api_key = os.environ.get("ITN_LLM_API_KEY") or os.environ.get("LLM_API_KEY", "EMPTY")
    model = os.environ.get("ITN_LLM_MODEL") or os.environ.get("LLM_MODEL", "qwen3.6-27b")
    if endpoint is None:
        endpoint = _next_llm_endpoint()
    
    # endpoint格式: http://localhost:8000/v1
    client = OpenAI(
        api_key=api_key,
        base_url=endpoint.rstrip("/"),
    )

    user_lines = [
        "以下 pairs 为第一阶段手工 ITN 结果（已去标签/部分数字/去标点/小写，**保留空格**）。",
        "手工 ITN 不完备，请继续 ITN + ref/hyp 交叉对齐，返回 JSON 数组：",
    ]
    for item in batch_items:
        user_lines.extend([
            f"\n[id={item['id']}]",
            f"ref: {item['ref']}",
            f"hypo: {item['hypo']}",
        ])

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ITN_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(user_lines)},
                ],
                max_tokens=4096,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            return _extract_json_array(content)
        except Exception as e:
            last_err = e
            if attempt + 1 >= max_retries:
                break
            if "429" in str(e):
                time.sleep(min(60, 5 * (2 ** attempt)))
            else:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM ITN failed after {max_retries} retries: {last_err}")


def run_llm_itn(
    items: list[dict],
    cache_path: Path,
    batch_size: int = 16,
    concurrency: int = 1,
    use_cache: bool = True,
    on_batch_done=None,
) -> dict:
    cache = {}
    if use_cache and cache_path.exists():
        cache = load_json(cache_path)
        print(f"Loaded LLM ITN cache: {len(cache)} results from {cache_path}", flush=True)

    pending = [it for it in items if it["wav"] not in cache]
    if not pending:
        print("All LLM ITN results cached.", flush=True)
        return cache

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    num_endpoints = len(_LLM_ENDPOINTS)
    effective_concurrency = max(concurrency, num_endpoints * 2)  # 2 workers per endpoint
    print(
        f"Running LLM ITN on {len(pending)} items "
        f"(batch_size={batch_size}, {len(batches)} requests, "
        f"{num_endpoints} endpoints, concurrency={effective_concurrency})...",
        flush=True,
    )

    cache_lock = threading.Lock()
    error_list = []

    def process_batch(batch_idx, batch):
        try:
            entries = llm_itn_one_batch(batch, cache, cache_path, use_cache=use_cache)
            if on_batch_done is not None:
                with cache_lock:
                    on_batch_done(batch, entries, dict(cache))
        except Exception as e:
            error_list.append(e)
            print(f"LLM ITN batch {batch_idx} error: {e}", flush=True)

    if num_endpoints > 1 and len(batches) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = []
            for i, batch in enumerate(batches):
                futures.append(executor.submit(process_batch, i, batch))
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM ITN"):
                f.result()
    else:
        for i, batch in enumerate(tqdm(batches, desc="LLM ITN")):
            process_batch(i, batch)

    if error_list:
        print(f"WARNING: {len(error_list)} LLM ITN errors", flush=True)

    print(f"LLM ITN cache saved to {cache_path} ({len(cache)} entries)", flush=True)
    return cache


def summarize_cer(details: list[dict], ref_key: str, hyp_key: str) -> dict:
    total_sub = total_ins = total_del = total_chars = 0
    cers = []
    for item in details:
        ref = item[ref_key]
        hyp = item[hyp_key]
        cer, sub, ins, dele, chars = calc_cer(ref, hyp)
        item[f"{ref_key}_cer"] = cer
        total_sub += sub
        total_ins += ins
        total_del += dele
        total_chars += chars
        cers.append(cer)
    weighted = (total_sub + total_ins + total_del) / total_chars * 100 if total_chars else 0
    return {
        "weighted_cer": weighted,
        "avg_cer": float(np.mean(cers) * 100) if cers else 0,
        "median_cer": float(np.median(cers) * 100) if cers else 0,
        "min_cer": float(min(cers) * 100) if cers else 0,
        "max_cer": float(max(cers) * 100) if cers else 0,
        "total_chars": total_chars,
        "total_ins": total_ins,
        "total_del": total_del,
        "total_sub": total_sub,
    }


def write_details(path: Path, title: str, summary: dict, details: list[dict],
                  ref_key: str, hyp_key: str, cer_key: str, evaluated_at: str):
    lines = [
        title,
        f"Evaluated at: {evaluated_at}",
        "",
        "─" * 100,
        "SUMMARY",
        "─" * 100,
        f"Weighted CER: {summary['weighted_cer']:.2f}%",
        f"Average CER:  {summary['avg_cer']:.2f}%",
        f"Median CER:   {summary['median_cer']:.2f}%",
        f"Min CER:      {summary['min_cer']:.2f}%",
        f"Max CER:      {summary['max_cer']:.2f}%",
        f"Char Errors:  {summary['total_ins']} ins, {summary['total_del']} del, "
        f"{summary['total_sub']} sub / {summary['total_chars']} chars",
        "",
    ]
    sorted_details = sorted(details, key=lambda x: x[cer_key], reverse=True)
    for rank, item in enumerate(sorted_details, start=1):
        lines.extend([
            "=" * 100,
            f"Rank {rank}/{len(sorted_details)} | CER: {item[cer_key]*100:.2f}% | "
            f"Sub: {item['substitutions']} Ins: {item['insertions']} Del: {item['deletions']} | "
            f"Chars: {item['chars']} | {item['name']}",
            "=" * 100,
            f"WAV:  {item['wav']}",
            "",
            "Reference (start):",
            item["ref_start"],
            "",
            "Hypothesis (start):",
            item["hypo_start"],
            "",
            "Reference (final):",
            item[ref_key],
            "",
            "Hypothesis (final):",
            item[hyp_key],
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison(path: Path, manual_summary: dict, llm_summary: dict,
                     details: list[dict], evaluated_at: str, sample_size: int):
    lines = [
        f"Manual ITN vs LLM ITN Comparison ({sample_size} fixed samples)",
        f"Evaluated at: {evaluated_at}",
        "",
        "─" * 100,
        "OVERALL",
        "─" * 100,
        f"{'Method':<20} {'Weighted CER':>14} {'Avg CER':>10} {'Median CER':>12}",
        f"{'Manual rules':<20} {manual_summary['weighted_cer']:>13.2f}% "
        f"{manual_summary['avg_cer']:>9.2f}% {manual_summary['median_cer']:>11.2f}%",
        f"{'LLM (deepseek)':<20} {llm_summary['weighted_cer']:>13.2f}% "
        f"{llm_summary['avg_cer']:>9.2f}% {llm_summary['median_cer']:>11.2f}%",
        f"{'Delta (LLM-Manual)':<20} "
        f"{llm_summary['weighted_cer'] - manual_summary['weighted_cer']:>+13.2f}% "
        f"{llm_summary['avg_cer'] - manual_summary['avg_cer']:>+9.2f}% "
        f"{llm_summary['median_cer'] - manual_summary['median_cer']:>+11.2f}%",
        "",
    ]

    by_delta = sorted(
        details,
        key=lambda x: x["llm_cer"] - x["manual_cer"],
        reverse=True,
    )
    lines.append("Top cases where LLM ITN differs most from Manual (by CER delta):")
    lines.append("")
    for rank, item in enumerate(by_delta[:30], start=1):
        delta = (item["llm_cer"] - item["manual_cer"]) * 100
        lines.extend([
            "=" * 100,
            f"#{rank} | Manual CER: {item['manual_cer']*100:.2f}% | "
            f"LLM CER: {item['llm_cer']*100:.2f}% | Delta: {delta:+.2f}% | {item['name']}",
            "=" * 100,
            "Reference (start):",
            item["ref_start"],
            "",
            "Hypothesis (start):",
            item["hypo_start"],
            "",
            "Manual ref/hypo (CER baseline):",
            item["ref_manual"],
            item["hypo_manual"],
            "",
            "LLM input (manual prep, spaces kept):",
            item["ref_manual_prep"],
            item["hypo_manual_prep"],
            "",
            "LLM ref/hypo (stage-2 ITN, CER):",
            item["ref_llm"],
            item["hypo_llm"],
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Number of audios in fixed sample (default: 200)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed when creating sample list (default: 42 for 200, 43 for 500, else sample-size)")
    parser.add_argument("--skip-asr", action="store_true", help="Use cached ASR results")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM ITN")
    parser.add_argument("--asr-batch-size", type=int, default=16, help="ASR + LLM ITN batch size")
    parser.add_argument("--refresh-llm-cache", action="store_true", help="Re-run LLM ITN")
    args = parser.parse_args()

    sample_size = args.sample_size
    seed = args.seed
    if seed is None:
        seed = 42 if sample_size == 200 else (43 if sample_size == 500 else sample_size)

    paths = eval_paths(sample_size)
    sampled = load_fixed_sample(sample_size, paths["sample_list"], seed=seed)
    if not sampled:
        print("No valid samples found.")
        return

    asr_results = run_asr(
        sampled,
        use_cache=args.skip_asr,
        cache_path=paths["asr_cache"],
        batch_size=args.asr_batch_size,
    )
    validate_asr_cache(sampled, asr_results)

    detailed = []
    llm_input_items = []

    print(f"\n{'='*95}")
    print(f"{'File':<30} {'Manual':>8} {'LLM':>8}")
    print(f"{'='*95}")

    for wav_path, json_path in sampled:
        hypo_raw = asr_results.get(str(wav_path), "")
        item = build_eval_item(wav_path, json_path, hypo_raw)
        manual_cer = item["manual_cer"]
        detailed.append(item)
        llm_input_items.append(item)
        print(f"{wav_path.name:<30} {manual_cer*100:>7.1f}%")

    manual_summary = summarize_cer(
        [{**d, "ref_final": d["ref_manual"], "hypo_final": d["hypo_manual"]} for d in detailed],
        "ref_final", "hypo_final",
    )

    llm_summary = None
    if not args.skip_llm:
        use_cache = not args.refresh_llm_cache
        llm_cache = run_llm_itn(
            llm_input_items,
            cache_path=paths["llm_cache"],
            batch_size=args.asr_batch_size,
            use_cache=use_cache,
        )
        for item in detailed:
            llm = llm_cache.get(item["wav"], {})
            item["ref_llm"], item["hypo_llm"] = llm_itn_postprocess(
                llm.get("ref_final", ""),
                llm.get("hypo_final", ""),
                item["ref_manual"],
                item["hypo_manual"],
            )
            item["llm_cer"], _, _, _, _ = calc_cer(item["ref_llm"], item["hypo_llm"])

        llm_summary = summarize_cer(
            [{**d, "ref_final": d["ref_llm"], "hypo_final": d["hypo_llm"]} for d in detailed],
            "ref_final", "hypo_final",
        )

        print(f"{'='*95}")
        print(f"\nManual Weighted CER: {manual_summary['weighted_cer']:.2f}%")
        print(f"LLM    Weighted CER: {llm_summary['weighted_cer']:.2f}%")
        print(f"Delta: {llm_summary['weighted_cer'] - manual_summary['weighted_cer']:+.2f}%")
        llm_worse = sum(
            1 for d in detailed if d.get("llm_cer", 0) > d["manual_cer"] + 1e-9
        )
        print(f"LLM worse than manual: {llm_worse}/{len(detailed)}")

    evaluated_at = datetime.now().isoformat()
    model_name = os.environ.get("ITN_LLM_MODEL", "deepseek-v4-flash")

    summary = {
        "sample_list": str(paths["sample_list"]),
        "sample_size": len(sampled),
        "seed": seed,
        "manual": manual_summary,
        "llm": llm_summary,
        "llm_model": model_name if llm_summary else None,
        "evaluated_at": evaluated_at,
        "details": sorted(detailed, key=lambda x: x["manual_cer"], reverse=True),
    }
    summary_path = paths["summary"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    write_details(
        paths["details_manual"],
        f"Batch Eval — Manual ITN ({len(sampled)} fixed samples, sorted by CER high → low)",
        manual_summary, detailed, "ref_manual", "hypo_manual", "manual_cer", evaluated_at,
    )
    print(f"Manual details saved to: {paths['details_manual']}")

    if llm_summary:
        write_details(
            paths["details_llm"],
            f"Batch Eval — LLM ITN ({model_name}, {len(sampled)} fixed samples)",
            llm_summary, detailed, "ref_llm", "hypo_llm", "llm_cer", evaluated_at,
        )
        write_comparison(
            paths["comparison"],
            manual_summary, llm_summary, detailed, evaluated_at, sample_size,
        )
        print(f"LLM details saved to: {paths['details_llm']}")
        print(f"Comparison saved to: {paths['comparison']}")

    write_details(
        paths["details_legacy"],
        f"Batch Eval Details ({len(sampled)} fixed samples, sorted by CER high → low)",
        manual_summary, detailed, "ref_manual", "hypo_manual", "manual_cer", evaluated_at,
    )
    print(f"Details saved to: {paths['details_legacy']}")


if __name__ == "__main__":
    main()
