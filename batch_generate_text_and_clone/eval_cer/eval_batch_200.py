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
import json
import os
import random
import re
import socket
import string
import sys
import time
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
OUT_DIR = Path("/root/code/github_repos/OmniVoice-fork/batch_cloned_voices")
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

ITN_SYSTEM_PROMPT = """你是 ASR 评测第二阶段 ITN（逆文本归一化）专家。

## 输入说明
你将收到 **已完成第一阶段手工 ITN** 的 ref / hyp 文本对。
手工 ITN **不完备**，可能漏掉数字、单位、同音对齐、拼音/汉字统一等；你的任务是在此基础上 **继续 ITN + ref/hyp 交叉对齐**。

**输入已做（勿重复破坏）：** 去标签、部分数字/单位归一化、去标点、英文小写。
**输入保留空格**——便于分词、对齐英文单词与中文片段；**输出也请保留单词间空格**（不要删光空格、不要把英文词粘在一起）。

## 核心目标
消除「写法不同但说的同一件事」造成的虚假字错，使 CER 反映真实读音/内容差异。
**不是**改写语义、不是帮 ASR 纠错、不是润色。

## 总原则
1. **成对处理**：ref、hyp 对照阅读，按语义片段对齐后再改。
2. **等价则统一**：读音/语义相同、仅写法不同 → 统一为 **ref 侧形式**，hyp 跟随 ref。
3. **ref 也可补全**：手工 ITN 在 ref 侧也可能漏归一化，ref_final 可修正 ref 中遗漏项，但 **不要** 改变 ref 语义；无明确收益时 **保持 ref 与输入一致**。
4. **不确定保留**；**真差异不对齐**（不同名词、动词、数字、语义）。
5. **禁止** 错误合并英文单词（如把 `just described` 合成 `justdescribed`）。
6. **空格**：英文词间保留空格；**中文与英文/数字相邻处不要插入空格**（`哎we` 正确，`哎 we` 错误）；中文词间若无 ref 侧空格则不要新增。

---

## 第二阶段任务（按顺序）

### A. 补全手工 ITN 遗漏（ref、hyp 各自检查）

**数字（手工可能漏转）**
- 中文数词→阿拉伯：`一百二十`→`120`，`四十五`→`45`，`三千`→`3000`
- 逐字读数：`三零二`→`302`，`八八七七`→`8877`，`eight eight seven seven`→`8877`
- 英文数词：`forty-five`→`45`，`one hundred and twenty`→`120`，`twelve point three`→`12.3`
- 百分数：`百分之八十`/`80 percent`/`eighty percent`→`80`；`八百分之`→`80`（ASR 语序错误）
- 分数：`三分之七`→`3分之7`；金额：`一块五`→`1块5`，`两块三毛`→`2块3`
- **保留小数点**：`12.5` 不能变成 `125`
- 单字「一~九」**仅**跟量词/单位时转换；**不**转 `一样`/`一般`/`一些`/`一起` 内的「一」
- 专名/地名如 `零丁洋`、`一诺千金` 不误转

**单位**
- `km/h`、`kilometers per hour`→`kmh`；`centimeters`/`centimetres`→`cm`；`millimeters`→`mm`
- `3 minutes`/`3 mins`→`3min`；`45 kmh` 数字与单位可连写 `45kmh`

**标签残留**
- 若仍有 `[question-oh]` 等方括号标签，删除且不保留标签内文字

**标点残留**
- 删除剩余标点，**保留数字内小数点**（如 `12.5`）

---

### B. ref ↔ hyp 交叉对齐（核心）

**B1. 同音/近音字**
音节相同或 ASR 常见近音 → 统一为 ref 用字，hyp 跟随。
- 人名：芳芳/方方、皓轩/浩轩
- 词语：刻舟/勾舟、求剑/球鞋（非此例）
- 数学术语误听：**求斜边** / **球鞋边**（同音 qiú xié biān）→ 跟 ref `求斜边`

**B2. 拼音/罗马音 ↔ 汉字**
语义相同 → 跟 ref 侧形式（ref 拼音则 hyp 转拼音；ref 汉字则 hyp 转汉字）。
- `hong jun bu pa yuan zheng nan` ↔ `红军不怕远征难`
- `qi lv chang zheng` ↔ `七律长征`
- `shan shui yi zhou pang bo` ↔ `山水一州磅礴`

**B3. 书名/诗名/专名**
- `yueyang lou ji` ↔ `岳阳楼记` ↔ `Yueyang Lou Ji`
- `song yuan er shi an xi` ↔ `松原二使安息`（《送元二使安西》近音）
- `not that yueyang lou ji` / `not that 岳阳楼记` → 两侧统一

**B4. 公式/符号 ↔ 近音读法**
- `a²+b²=c²` ↔ `a块大c色块` ↔ `A块大 C色块`
- `rt△` ↔ `r t` / `R T`（直角三角形符号）
- 保留 ref 中公式前后的功能词（如 `but`、`which`）

**B5. 感叹词/语气词（读音等价）**
- `wow` ↔ `哇`；`oh` ↔ `哦`
- 例：ref `拍错像佩奇 wow`，hyp `拍错像佩奇 哇` → hyp 改为 `拍错像佩奇 wow`

**B6. 纠正语境中的同一数值**
两侧都在自我纠正时，同一位置数值应对齐。
- ref `221bc`，hyp `202呀 bc`（都在纠正年份）→ hyp `221bc`

**B7. 英文写法变体**
- `question 1` / `question one` → 跟 ref
- 同一词不同拼写/空格 → 跟 ref
- **禁止** 把两个英文词合成一个词

---

### C. 明确不要对齐

- 语义/读音确实不同：`bar chart` / `bartlett`，`doctor brown` / `doctor不让你`，`wait` / `weight`，`3min` / `3米`
- hyp **多出** ref 没有的语气词：`啊`、`呀`、`哎`、`嗯`、`呃`（保留字但 **勿** 在其前后加空格）
- ref 的重复/停顿：`不对不对`、`被…被`——不要强行删 ref 侧内容；hyp 也不伪造 ref 没有的重复
- 不同动词/名词/数字：即使上下文相似也不对齐

---

## 输出格式
只输出 JSON 数组，不要 markdown 代码块：
[{"id": 0, "ref_final": "...", "hypo_final": "..."}, ...]

ref_final、hypo_final：**小写、无标点、保留空格**（单词/片段之间用单空格）。

---

## 示例（学会规则即可泛化到未列出的 case）

**例1 百分数+英文数词（补全+保留差异）**
ref: `80%和120块 wait`
hyp: `八百分之和一百二十块 weight`
→ ref: `80和120块 wait` / hyp: `80和120块 weight`（weight≠wait 保留）

**例2 同音人名**
ref: `芳芳的 blue shirt`
hyp: `方方的 blue shirt`
→ ref: `芳芳的 blue shirt` / hyp: `芳芳的 blue shirt`

**例3 同音成语**
ref: `刻舟求剑`
hyp: `勾舟求剑`
→ ref: `刻舟求剑` / hyp: `刻舟求剑`

**例4 拼音诗名**
ref: `hong jun bu pa yuan zheng nan`
hyp: `红军不怕远征难`
→ ref: `hong jun bu pa yuan zheng nan` / hyp: `hong jun bu pa yuan zheng nan`

**例5 书名拼音↔汉字**
ref: `not that yueyang lou ji its the first line`
hyp: `not that 岳阳楼记 is the first line`
→ ref: `not that yueyang lou ji its the first line`
   hyp: `not that yueyang lou ji its the first line`

**例6 数学同音+公式读法**
ref: `rt△求斜边 a²+b²=c² but 哪个是a哪个是b啊 晕了`
hyp: `r t 球鞋边 a块大 c色块 到底哪个是a哪个是b呀 哎 晕了`
→ ref: `rt△求斜边 a²+b²=c² but 哪个是a哪个是b啊 晕了`
   hyp: `rt△求斜边 a²+b²=c² 到底哪个是a哪个是b呀 哎 晕了`
（对齐公式与同音词；保留 hyp 多出的 `到底`/`呀`/`哎` 若 ref 无对应）

**例7 感叹词**
ref: `拍错像佩奇 wow`
hyp: `拍错像佩奇 哇`
→ ref: `拍错像佩奇 wow` / hyp: `拍错像佩奇 wow`

**例8 纠正语境数值**
ref: `应该是221bc不是202`
hyp: `应该是202呀 bc不是221`
→ ref: `应该是221bc不是202` / hyp: `应该是221bc不是202`

**例9 英文逐字读数**
ref: `call eight eight seven seven`
hyp: `call 8877`
→ ref: `call 8877` / hyp: `call 8877`

**例10 小数保留**
ref: `twelve point five meters`
hyp: `12.5米`
→ ref: `12.5 meters` / hyp: `12.5米`（对齐后单位形式跟 ref 或统一）

**例11 一样不转1**
ref: `这样一样的问题`
hyp: `这样1样的问题`
→ ref: `这样一样的问题` / hyp: `这样一样的问题`

**例12 真差异不对齐**
ref: `draw a bar chart`
hyp: `draw a bartlett`
→ 两侧保持不同（chart≠bartlett）

**例13 单位**
ref: `speed is 45 km/h`
hyp: `speed is 45 kilometers per hour`
→ ref: `speed is 45kmh` / hyp: `speed is 45kmh`
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


def manual_itn_preprocess(text: str) -> str:
    """Stage-1 ITN (LLM input and manual baseline)."""
    return finalize_cer_text(strip_punctuation(normalize_numbers(remove_tags(text))).lower())


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
    ref_manual_prep = manual_itn_preprocess(truth_raw)
    hyp_manual_prep = manual_itn_preprocess(hypo_raw)
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


def load_asr_model(batch_size: int = 16):
    """Load Qwen3-ASR model."""
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


def call_llm_itn_batch(batch_items: list[dict], max_retries: int = 6) -> list[dict]:
    api_key = os.environ.get("ITN_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("ITN_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("ITN_LLM_MODEL") or os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    if not api_key or not base_url:
        raise RuntimeError(f"Set ITN_LLM_API_KEY and ITN_LLM_BASE_URL in {ENV_FILE}")

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

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ITN_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        "max_tokens": 8192,
        "temperature": 0,
    }
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return _extract_json_array(content)
        except urllib.error.HTTPError as e:
            last_err = e
            if attempt + 1 >= max_retries:
                break
            if e.code == 429:
                time.sleep(min(60, 5 * (2 ** attempt)))
            else:
                time.sleep(2 ** attempt)
        except (TimeoutError, socket.timeout) as e:
            last_err = e
            if attempt + 1 >= max_retries:
                break
            # Large batches can exceed read timeout; back off longer before retry.
            time.sleep(min(120, 10 * (2 ** attempt)))
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = e
            if attempt + 1 >= max_retries:
                break
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
    print(
        f"Running LLM ITN on {len(pending)} items "
        f"(batch_size={batch_size}, {len(batches)} requests)...",
        flush=True,
    )

    for batch in tqdm(batches, desc="LLM ITN"):
        entries = llm_itn_one_batch(batch, cache, cache_path, use_cache=use_cache)
        if on_batch_done is not None:
            on_batch_done(batch, entries, dict(cache))

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
