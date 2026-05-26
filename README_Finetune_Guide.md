# OmniVoice 新手微调完全指南

> **目标读者**：刚接触 TTS（文本转语音）和 Voice Clone（声音克隆）的新手，想用自己的数据微调 OmniVoice 模型。
>
> **本文目标**：手把手带你从 0 到 1，完成环境搭建 → 跑通推理 → 准备数据 → 微调训练 → 测试效果的完整流程。
>
> **特别说明**：本指南不仅覆盖基础 TTS 微调，更重点解释**微调与 Voice Clone 的关系**——这是新手最容易困惑的地方。

---

## 目录

1. [OmniVoice 是什么？](#1-omnivoice-是什么)
2. [你需要什么硬件？](#2-你需要什么硬件)
3. [环境安装（一步一步）](#3-环境安装一步一步)
4. [Step 0：先跑通预训练模型（建立信心）](#4-step-0先跑通预训练模型建立信心)
5. [什么是 Finetune？为什么要 Finetune？](#5-什么是-finetune为什么要-finetune)
6. [Step 1：准备你的数据](#6-step-1准备你的数据)
7. [Step 2：数据预处理（音频转 Token）](#7-step-2数据预处理音频转-token)
8. [Step 3：配置训练参数](#8-step-3配置训练参数)
9. [Step 4：启动训练](#9-step-4启动训练)
10. [Step 5：监控训练进度](#10-step-5监控训练进度)
11. [Step 6：用微调后的模型推理](#11-step-6用微调后的模型推理)
12. [参数调优指南](#12-参数调优指南)
13. [常见问题 FAQ](#13-常见问题-faq)

---

## 1. OmniVoice 是什么？

**一句话解释**：OmniVoice 是一个 AI 语音合成模型，你给它一段文字，它就能生成人声朗读这段文字。

**更厉害的是**：
- **声音克隆（Voice Cloning）**：你给它 3~10 秒的某人说话录音，它就能模仿这个人的声音朗读任何文字
- **支持 600+ 种语言**：中文、英文、日语、韩语......几乎所有语言都能合成
- **语音设计（Voice Design）**：不用录音，直接描述 "女声、低沉、英式口音"，它就能生成对应的声音
- **速度超快**：生成速度是实时播放的 40 倍

**技术架构（简单理解）**：
- 底层是一个叫 **Qwen3-0.6B** 的大语言模型（类似 ChatGPT 但是专门做语音的）
- 音频通过一个 **Tokenizer** 转成 8 层数字编码（就像把语音"压缩"成数字）
- 训练时模型学习"看到文字 + 参考音频，预测对应的音频编码"
- 推理时通过 **扩散模型（Diffusion）** 一步步"去噪"生成音频编码，再转回声音

---

## 2. 你需要什么硬件？

### 最低配置（能跑起来）

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU，显存 **≥ 16GB**（如 RTX 4090, A100 40GB, V100 32GB） |
| 内存 | **≥ 32GB** |
| 硬盘 | **≥ 100GB** 空闲空间（模型文件 + 数据） |
| 系统 | Linux（推荐 Ubuntu 22.04）或 Windows WSL2 |

### 推荐配置（训练更舒服）

| 项目 | 要求 |
|------|------|
| GPU | **2~4 张 A100 40GB/80GB** 或 RTX 4090 |
| 内存 | **≥ 64GB** |
| 硬盘 | **≥ 500GB** SSD |

> **新手提示**：如果你只有 1 张 24GB 显存的卡（如 RTX 4090），也能训练，只是 batch size 要小一些，训练时间更长。

### 检查你的 GPU

```bash
nvidia-smi
```

如果显示了 GPU 信息，说明有 NVIDIA 显卡。记下显存大小（Memory-Usage 那一栏）。

### 检查 CUDA 版本

```bash
nvcc --version
```

需要 CUDA 版本 **≥ 11.8**。

---

## 3. 环境安装（一步一步）

### 3.1 创建虚拟环境（重要！避免搞乱系统）

```bash
# 安装 conda（如果还没有）
# 去 https://docs.conda.io/en/latest/miniconda.html 下载安装

# 创建一个名为 omnivoice 的环境，Python 3.10
conda create -n omnivoice python=3.10 -y

# 激活环境
conda activate omnivoice
```

> **为什么要用虚拟环境？** 因为不同项目依赖不同版本的库，虚拟环境能隔离它们，避免冲突。

### 3.2 安装 PyTorch（带 CUDA 支持）

去 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择你的 CUDA 版本，例如 CUDA 12.8：

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

**验证 PyTorch 能识别 GPU**：

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

如果输出 `CUDA available: True`，恭喜你！可以继续了。

### 3.3 下载 OmniVoice 代码

```bash
# 选一个目录存放代码，比如 ~/projects
cd ~
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
```

> **如果下载慢**：`git clone https://ghproxy.com/https://github.com/k2-fsa/OmniVoice.git`

### 3.4 安装 OmniVoice

```bash
# 方式一：直接安装（推荐新手）
pip install -e .

# 方式二：如果需要评估功能，一并安装
pip install -e ".[eval]"
```

安装完成后验证：

```bash
python -c "from omnivoice import OmniVoice; print('安装成功！')"
```

### 3.5 安装 HuggingFace 相关工具

```bash
pip install huggingface_hub
```

> **国内用户注意**：如果访问 HuggingFace 慢，设置镜像：
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> # 建议加到 ~/.bashrc 里永久生效
> echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
> ```

---

## 4. Step 0：先跑通预训练模型（建立信心）

在微调之前，先用官方预训练模型跑通推理，确认环境没问题。

### 4.1 写一个测试脚本

创建文件 `test_inference.py`：

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

print("正在加载模型（首次会下载约 2GB 模型文件，请耐心等待）...")

# 加载预训练模型
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",  # 用第一张 GPU
    dtype=torch.float16   # 半精度省显存
)

print("模型加载完成！")

# 测试 1：自动声音生成（Auto Voice）
print("\n[测试 1] 自动生成声音...")
audio = model.generate(text="你好，这是 OmniVoice 的测试。")
sf.write("test_auto.wav", audio[0], 24000)
print("已保存到 test_auto.wav")

# 测试 2：声音设计（Voice Design）
print("\n[测试 2] 声音设计...")
audio = model.generate(
    text="Hello, this is a voice design test.",
    instruct="female, low pitch, british accent"
)
sf.write("test_design.wav", audio[0], 24000)
print("已保存到 test_design.wav")

# 测试 3：声音克隆（Voice Cloning）
# 你需要准备一段 3~10 秒的参考音频
# print("\n[测试 3] 声音克隆...")
# audio = model.generate(
#     text="这是用参考音频克隆出来的声音。",
#     ref_audio="your_reference.wav",  # 替换为你的音频路径
#     ref_text="参考音频的文本内容",    # 参考音频说了什么
# )
# sf.write("test_clone.wav", audio[0], 24000)
# print("已保存到 test_clone.wav")

print("\n全部测试完成！请用播放器听听生成的音频。")
```

运行：

```bash
python test_inference.py
```

如果听到了合成语音，说明环境完全 OK！可以进入微调环节了。

---

## 5. 什么是 Finetune？为什么要 Finetune？

### 5.1 通俗解释

预训练模型（`k2-fsa/OmniVoice`）是用海量数据训练出来的**通用模型**，它能克隆各种声音、说各种语言。

**Finetune（微调）** 就是在这个通用模型的基础上，用你的**专属数据**再训练一段时间，让它：

1. **更适配你的场景**：比如你是做有声书的，用有声书数据微调后，生成的语音更像专业播音员
2. **更好地克隆特定人声**：用某人的大量录音微调，clone 效果更逼真
3. **支持特定术语**：比如医学、法律领域的专业词汇，微调后发音更准确
4. **改善特定语言/方言**：比如让你的模型说某种方言更地道

### 5.2 微调需要多少数据？

| 目标 | 推荐数据量 | 训练时间（2x A100） |
|------|-----------|-------------------|
| 改善某个说话人的克隆效果 | **10~30 分钟** 该说话人的音频 | 30~60 分钟 |
| 让模型适应某个领域（如新闻播报） | **5~10 小时** 该领域音频 | 1~2 小时 |
| 改善某种语言/方言效果 | **10+ 小时** 该语言音频 | 2~4 小时 |

> **新手建议**：先从 **30 分钟~1 小时** 的单人数据开始，快速看到效果，建立信心！

### 5.3 微调后还能做 Voice Clone 吗？（核心问题）

**完全可以，而且这正是微调最有价值的用法。**

OmniVoice 的微调不是只训练 TTS（文字→声音），它训练的是**完整的 Voice Clone 能力**。训练时，模型从每段音频的前缀部分提取"参考音频"，学习如何根据这段参考音频的声音特征来生成后续音频。

微调后，三种生成模式**全部保留**：

```python
# 模式 1：Voice Clone（用参考音频）—— 微调后效果通常更好
audio = model.generate(
    text="你好，这是克隆的声音。",
    ref_audio="reference.wav",
    ref_text="参考音频的转写"
)

# 模式 2：Voice Design —— 也完全可用
audio = model.generate(
    text="Hello world",
    instruct="female, low pitch"
)

# 模式 3：Auto Voice —— 同样可用
audio = model.generate(text="自动生成声音")
```

### 5.4 你的数据内容决定微调后提升的方向

| 你的微调数据 | 对 Voice Clone 的影响 |
|-------------|---------------------|
| **同一个人** 的大量音频（如某主播 2 小时录音） | 对这个人的 clone **更稳定、更逼真** |
| **多种说话人** 的同一领域音频（如 100 个新闻播音员） | 该领域风格的 clone **音质更好、更自然** |
| **某种语言/方言** 的大量音频 | 该语言的 clone **口音更地道** |

> **关键点**：预训练模型已经能 zero-shot clone 任何人的声音。微调只是让它在**特定场景下**clone 得更好，不会丧失通用 clone 能力（除非严重过拟合）。

---

## 6. Step 1：准备你的数据

数据准备是微调**最重要**的一步。数据质量直接决定最终效果。

### 6.1 数据格式

你需要准备一个 **JSONL** 文件，每行是一个 JSON 对象：

```jsonl
{"id": "sample_001", "audio_path": "/data/audio/001.wav", "text": "你好，欢迎使用语音合成系统。", "language_id": "zh"}
{"id": "sample_002", "audio_path": "/data/audio/002.wav", "text": "今天天气真不错，适合出去散步。", "language_id": "zh"}
{"id": "sample_003", "audio_path": "/data/audio/003.wav", "text": "Hello, welcome to the voice synthesis system.", "language_id": "en"}
```

**字段说明**：

| 字段 | 是否必填 | 说明 |
|------|---------|------|
| `id` | **是** | 唯一标识符，不能重复 |
| `audio_path` | **是** | 音频文件的**绝对路径** |
| `text` | **是** | 音频对应的文本（ transcript / 转写） |
| `language_id` | 否 | 语言代码，如 `zh`（中文）、`en`（英文）、`ja`（日语） |

#### 为什么没有 speaker_id 字段？

**因为不需要。** OmniVoice 的训练机制是**自监督**的：

```
一段 10 秒的音频样本
  ├── 前 3 秒 → 作为"参考音频"（prompt，就像推理时的 ref_audio）
  └── 后 7 秒 → 作为"目标音频"（被 mask，让模型预测）
```

模型学习的是：**"给定这段参考音频的声音特征 + 这段文本，生成对应的音频"**。

所以即使 JSONL 里没有 `speaker_id`：
- 如果你的数据**全是同一个人**，模型自然会熟悉这个人的声音分布
- 如果你的数据**是多个人混合**，模型会学到平均分布
- 无论哪种情况，**Voice Clone 的能力都保留**——推理时你仍然需要提供 `ref_audio`

### 6.2 音频要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 格式 | WAV / FLAC / MP3 | 推荐 WAV |
| 采样率 | **24 kHz** | 如果不是 24kHz，脚本会自动重采样 |
| 时长 | **3 秒 ~ 30 秒** | 太短信息不够，太长训练效率低 |
| 质量 | **干净、清晰** | 避免背景噪音、混响、音乐 |
| 音量 | **适中、一致** | 避免忽大忽小 |
| 内容 | **朗读风格** | 避免多人对话、歌声、喊叫 |

### 6.3 数据准备的实际步骤

#### 方式 A：你有现成的音频 + 文本（最简单）

如果你已经有配好对的音频和文本文件，只需写一个 Python 脚本生成 JSONL：

```python
import json
import os

# 配置你的数据目录
audio_dir = "/path/to/your/audio"  # 音频文件夹
text_dir = "/path/to/your/text"    # 文本文件夹（或文本在文件名里）
output_jsonl = "my_data_train.jsonl"

with open(output_jsonl, "w", encoding="utf-8") as f:
    for audio_file in sorted(os.listdir(audio_dir)):
        if not audio_file.endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, audio_file)

        # 方式 1：从同名文本文件读取
        text_file = audio_file.replace(".wav", ".txt")
        text_path = os.path.join(text_dir, text_file)
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as tf:
                text = tf.read().strip()
        else:
            # 方式 2：从文件名中提取（不推荐，除非确实在文件名里）
            text = audio_file.replace(".wav", "")

        sample = {
            "id": audio_file.replace(".wav", ""),
            "audio_path": os.path.abspath(audio_path),
            "text": text,
            "language_id": "zh"
        }
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"已生成 {output_jsonl}")
```

#### 方式 B：只有音频，没有文本（需要用 ASR 工具转写）

如果你的音频没有对应的文本，需要用语音识别（ASR）工具来转写。推荐：

- **中文**：使用 [FunASR](https://github.com/alibaba-damo-academy/FunASR) 或 [Whisper](https://github.com/openai/whisper)
- **英文**：使用 Whisper

示例（使用 Whisper）：

```bash
pip install openai-whisper

# 转写单个文件
whisper your_audio.wav --model medium --language Chinese --output_format json

# 批量转写（写一个 shell 脚本）
for f in /path/to/audios/*.wav; do
    whisper "$f" --model medium --language Chinese --output_dir /path/to/transcripts/
done
```

转写完成后，再按方式 A 组织成 JSONL。

#### 方式 C：从长音频切分（如播客、有声书）

如果你的原始素材是长音频（如 30 分钟的有声书），需要先用 VAD（语音活动检测）工具切分成短段：

```bash
pip install silero-vad
```

```python
import torch
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps

# 加载 VAD 模型
model = load_silero_vad()

# 读取音频
wav, sr = sf.read("long_audio.wav")
if sr != 16000:
    import librosa
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

# 检测语音片段
timestamps = get_speech_timestamps(
    torch.tensor(wav),
    model,
    min_speech_duration_ms=3000,   # 最少 3 秒
    max_speech_duration_s=30,      # 最多 30 秒
    return_seconds=True
)

# 切分保存
for i, ts in enumerate(timestamps):
    start, end = int(ts['start'] * sr), int(ts['end'] * sr)
    segment = wav[start:end]
    sf.write(f"segment_{i:04d}.wav", segment, sr)
```

切分后再用 Whisper 转写。

### 6.4 划分训练集和验证集

**必须**划分训练集和验证集（建议比例 9:1 或 8:2）。验证集用来监控模型有没有"学偏"。

```python
import json
import random

# 读取所有数据
with open("my_data_train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

# 划分 90% 训练，10% 验证
split_idx = int(len(lines) * 0.9)
train_lines = lines[:split_idx]
dev_lines = lines[split_idx:]

with open("my_data_train.jsonl", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("my_data_dev.jsonl", "w", encoding="utf-8") as f:
    f.writelines(dev_lines)

print(f"训练集: {len(train_lines)} 条")
print(f"验证集: {len(dev_lines)} 条")
```

---

## 7. Step 2：数据预处理（音频转 Token）

### 7.1 为什么需要这一步？

模型看不懂原始的音频波形，需要先把音频转成一种叫 **Token** 的数字表示。这个过程就像把语音"压缩编码"成模型能理解的 8 层数字矩阵。

### 7.2 运行 Tokenization 脚本

```bash
# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES="0,1"  # 根据你的实际情况修改

# 处理训练集
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl my_data_train.jsonl \
    --tar_output_pattern "data/finetune/tokens/train/audios/shard-%06d.tar" \
    --jsonl_output_pattern "data/finetune/tokens/train/txts/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True

# 处理验证集
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl my_data_dev.jsonl \
    --tar_output_pattern "data/finetune/tokens/dev/audios/shard-%06d.tar" \
    --jsonl_output_pattern "data/finetune/tokens/dev/txts/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True
```

**参数解释**：

| 参数 | 说明 |
|------|------|
| `--input_jsonl` | 你的数据文件 |
| `--tar_output_pattern` | Token 数据（tar 包）的输出路径 |
| `--jsonl_output_pattern` | 元数据文件的输出路径 |
| `--tokenizer_path` | 音频 Tokenizer 模型（HuggingFace 上的 `eustlb/higgs-audio-v2-tokenizer`） |
| `--nj_per_gpu` | 每个 GPU 上并行处理的工作进程数 |
| `--shuffle` | 是否打乱样本顺序 |

### 7.3 处理完成后你会得到什么？

```
data/finetune/tokens/
├── train/
│   ├── audios/
│   │   ├── shard-000000.tar   # 包含约 1000 个样本的音频 Token
│   │   ├── shard-000001.tar
│   │   └── ...
│   ├── txts/
│   │   ├── shard-000000.jsonl  # 对应 tar 包的文本元数据
│   │   ├── shard-000001.jsonl
│   │   └── ...
│   └── data.lst               # 训练集清单（训练时引用这个文件）
└── dev/
    ├── audios/
    ├── txts/
    └── data.lst               # 验证集清单
```

**`data.lst` 文件格式**：

```
/path/to/shard-000000.tar /path/to/shard-000000.jsonl 1000 3600.5
/path/to/shard-000001.tar /path/to/shard-000001.jsonl 800 2880.2
```

每行格式：`<tar路径> <jsonl路径> <样本数> <总时长(秒)>`

### 7.4 如果预处理失败怎么办？

```bash
# 加上 --skip_errors 参数跳过错误样本
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl my_data_train.jsonl \
    ... \
    --skip_errors
```

错误样本会记录在 `errors.jsonl` 中，处理完后检查这个文件，修复问题音频重新处理。

---

## 8. Step 3：配置训练参数

### 8.1 创建数据配置文件

创建文件 `my_data_config.json`：

```json
{
    "train": [
        {
            "manifest_path": ["data/finetune/tokens/train/data.lst"]
        }
    ],
    "dev": [
        {
            "manifest_path": ["data/finetune/tokens/dev/data.lst"]
        }
    ]
}
```

> **路径必须是绝对路径**，或者相对于你运行训练命令的目录的相对路径。

### 8.2 创建训练配置文件

创建文件 `my_train_config.json`：

```json
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,

    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,

    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",

    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 5000,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.01,
    "warmup_steps": 0,

    "batch_tokens": 8192,
    "gradient_accumulation_steps": 1,
    "num_workers": 2,

    "mixed_precision": "bf16",
    "allow_tf32": true,

    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 500,
    "keep_last_n_checkpoints": -1
}
```

### 8.3 关键参数详解（新手必看）

| 参数 | 默认值 | 说明 | 新手建议 |
|------|--------|------|---------|
| `init_from_checkpoint` | `"k2-fsa/OmniVoice"` | 从官方预训练模型开始微调 | **不要改** |
| `learning_rate` | `1e-5` | 学习率，控制每次更新参数的步长 | 微调用小学习率，1e-5~5e-5 |
| `steps` | `5000` | 总训练步数 | 数据少用 3000~5000，数据多用 10000+ |
| `batch_tokens` | `8192` | 每个 GPU 每步处理的 Token 数 | 显存小（16GB）改 4096 或 2048 |
| `mixed_precision` | `"bf16"` | 混合精度训练 | 不支持 bf16 的 GPU 改 `"fp16"` |
| `logging_steps` | `50` | 每 50 步打印一次日志 | 可以不改 |
| `eval_steps` | `500` | 每 500 步在验证集上测试一次 | 可以不改 |
| `save_steps` | `500` | 每 500 步保存一次模型 | 可以不改 |
| `keep_last_n_checkpoints` | `-1` | 保留最近的几个 checkpoint，-1=保留全部 | 硬盘紧张可以改 3 |

#### Voice Clone 相关参数（特别重要）

这两个参数直接控制模型在训练时学习 Voice Clone 的方式：

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `drop_cond_ratio` | `0.1` | **条件丢弃比例**。训练时有 10% 的概率"忘记"参考音频，强制模型学会不依赖参考音频也能生成。这是 Classifier-Free Guidance (CFG) 的训练技巧。 | 想强化 clone 能力 → 改 **0.05**；想保留更多通用性 → 保持 **0.1** |
| `prompt_ratio_range` | `[0.0, 0.3]` | **参考音频占比范围**。训练时从样本中截取 0%~30% 作为参考音频（prompt），其余作为预测目标。 | 想强化 clone 能力 → 改 **[0.2, 0.5]**；一般场景 → 保持 **[0.0, 0.3]** |

**通俗解释**：
- `prompt_ratio_range` 越大，模型在训练时看到的"参考音频"越长，学得更依赖参考音频的声音特征 → **Voice Clone 效果更强**
- `drop_cond_ratio` 越小，模型越难"忘记"参考音频 → **Voice Clone 更稳定**
- 但两个值都调得太极端，可能让模型丧失 Auto Voice / Voice Design 的能力，或导致过拟合

#### 关于 `batch_tokens` 的调整

这个参数**直接影响显存占用**：

| GPU 显存 | 建议 batch_tokens |
|---------|-------------------|
| 16 GB | 2048 ~ 4096 |
| 24 GB | 4096 ~ 8192 |
| 40 GB | 8192 ~ 16384 |
| 80 GB | 16384+ |

如果训练时报 `CUDA out of memory`，**第一步就是减小 `batch_tokens`**。

#### 关于 Attention 实现（重要）

默认使用 `flex_attention`，要求：
- PyTorch >= 2.5
- NVIDIA Ampere 架构或更新的 GPU（RTX 30 系列、A100、H100 等）

如果你的 GPU 比较老（如 RTX 2080Ti, V100），需要改用 SDPA：

```json
{
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 64
}
```

把这些加到配置文件中即可。

---

## 9. Step 4：启动训练

### 9.1 单卡训练

```bash
accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config my_train_config.json \
    --data_config my_data_config.json \
    --output_dir exp/my_first_finetune
```

### 9.2 多卡训练（推荐）

```bash
# 假设你有 2 张 GPU
accelerate launch \
    --gpu_ids "0,1" \
    --num_processes 2 \
    -m omnivoice.cli.train \
    --train_config my_train_config.json \
    --data_config my_data_config.json \
    --output_dir exp/my_first_finetune
```

### 9.3 使用现成的脚本

项目也提供了现成的微调脚本，你可以直接修改使用：

```bash
cp examples/run_finetune.sh my_run_finetune.sh
chmod +x my_run_finetune.sh

# 编辑 my_run_finetune.sh，修改里面的路径和参数
vim my_run_finetune.sh

# 运行
./my_run_finetune.sh
```

### 9.4 训练时你会看到什么？

正常训练日志大概长这样：

```
Step 50/5000 | Loss: 2.3456 | LR: 9.8e-06 | Tokens/s: 15234 | ETA: 2h15m
Step 100/5000 | Loss: 1.9876 | LR: 9.5e-06 | Tokens/s: 15890 | ETA: 2h10m
...
```

**重点关注 `Loss` 值**：
- Loss 应该**逐渐下降**（从 2.x 降到 1.x 甚至更低）
- 如果 Loss 一直不下降或震荡很大，可能需要调小学习率
- 如果 Loss 降到很低（< 0.5）但验证效果差，可能是过拟合

---

## 10. Step 5：监控训练进度

### 10.1 用 TensorBoard 查看

```bash
# 在另一个终端窗口运行
tensorboard --logdir exp/my_first_finetune/tensorboard --port 6006
```

然后在浏览器打开 `http://localhost:6006`，可以看到：
- 训练 Loss 曲线
- 学习率变化
- 验证 Loss（如果有验证集）

### 10.2 Checkpoint 文件

训练过程中会定期保存模型：

```
exp/my_first_finetune/
├── checkpoint-500/       # 第 500 步的模型
│   ├── model.safetensors
│   └── ...
├── checkpoint-1000/      # 第 1000 步的模型
├── checkpoint-1500/
├── ...
└── tensorboard/          # 日志文件
```

### 10.3 如何判断训练是否完成？

1. **达到设定的 steps**：比如设置 5000 步，跑到 5000 就停了
2. **Loss 收敛**：Loss 不再明显下降，连续几百步变化很小
3. **听验证效果**：用保存的 checkpoint 做推理，听起来满意就可以停

---

## 11. Step 6：用微调后的模型推理

### 11.1 加载微调后的模型

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

# 加载你微调后的 checkpoint
CHECKPOINT_PATH = "exp/my_first_finetune/checkpoint-5000"

model = OmniVoice.from_pretrained(
    CHECKPOINT_PATH,
    device_map="cuda:0",
    dtype=torch.float16
)

# 测试 1：Voice Clone（用参考音频）
print("\n[Voice Clone] 用参考音频克隆声音...")
audio = model.generate(
    text="你好，这是我微调后的模型克隆的声音。",
    ref_audio="your_reference.wav",      # 替换为你的参考音频
    ref_text="参考音频的转写文本",        # 参考音频说了什么
    language="zh"
)
sf.write("finetuned_clone.wav", audio[0], 24000)
print("已保存到 finetuned_clone.wav")

# 测试 2：Voice Design（不需要参考音频）
print("\n[Voice Design] 设计一个声音...")
audio = model.generate(
    text="Hello, this is voice design after fine-tuning.",
    instruct="female, low pitch, british accent"
)
sf.write("finetuned_design.wav", audio[0], 24000)
print("已保存到 finetuned_design.wav")

# 测试 3：Auto Voice（自动生成）
print("\n[Auto Voice] 自动生成声音...")
audio = model.generate(text="这是微调后自动生成的声音。")
sf.write("finetuned_auto.wav", audio[0], 24000)
print("已保存到 finetuned_auto.wav")
```

### 11.2 对比微调前后的效果

建议你：
1. 用同样的参考音频和文本，分别用预训练模型和微调模型生成
2. 保存两个音频文件
3. 仔细对比听，看微调后的模型是否有改善

对比脚本示例：

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

# 加载两个模型
pretrained = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
finetuned = OmniVoice.from_pretrained("exp/my_first_finetune/checkpoint-5000", device_map="cuda:0", dtype=torch.float16)

text = "你好，这是对比测试。"
ref_audio = "reference.wav"
ref_text = "参考音频的转写"

# 预训练模型生成
audio1 = pretrained.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language="zh")
sf.write("compare_pretrained.wav", audio1[0], 24000)

# 微调模型生成
audio2 = finetuned.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, language="zh")
sf.write("compare_finetuned.wav", audio2[0], 24000)

print("对比音频已生成，请听 compare_pretrained.wav 和 compare_finetuned.wav")
```

### 11.3 选择最好的 Checkpoint

如果有多个 checkpoint，可以在验证集上评估，选择 Loss 最低的那个。或者直接用耳朵听，选效果最好的。

---

## 12. 参数调优指南

### 12.1 常见场景的配置建议

#### 场景 A：单人声音克隆（让模型更好地模仿某人的声音）

**数据特征**：30 分钟~2 小时，全是同一个人的音频

**目标**：给一段这个人的参考音频，clone 效果比预训练模型更逼真

```json
{
    "learning_rate": 5e-6,
    "steps": 3000,
    "batch_tokens": 8192,
    "drop_cond_ratio": 0.05,
    "prompt_ratio_range": [0.2, 0.5]
}
```

- `learning_rate=5e-6`：小学习率，避免破坏预训练知识
- `drop_cond_ratio=0.05`：减少条件丢弃，让模型更依赖参考音频
- `prompt_ratio_range=[0.2, 0.5]`：增加参考音频占比，强化 clone 学习

**预期效果**：对这个人的 clone 更稳定、更像；对陌生人的 clone 仍然可用但可能略弱

#### 场景 B：领域适配（如新闻播报、有声书）

**数据特征**：5~10 小时，多种说话人，同一领域风格

**目标**：提升该领域风格的声音质量和自然度

```json
{
    "learning_rate": 1e-5,
    "steps": 10000,
    "batch_tokens": 8192,
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3]
}
```

- 保持默认 `drop_cond_ratio` 和 `prompt_ratio_range`
- 数据量大，可以多训一些步数

**预期效果**：新闻播报风格的声音更自然、更专业；Voice Clone 通用能力保留较好

#### 场景 C：小数据量快速实验（< 30 分钟音频）

**数据特征**：数据量很少，想快速验证效果

```json
{
    "learning_rate": 1e-5,
    "steps": 2000,
    "batch_tokens": 4096,
    "drop_cond_ratio": 0.05,
    "prompt_ratio_range": [0.1, 0.4],
    "save_steps": 250,
    "eval_steps": 250
}
```

- 步数少、保存频繁，方便快速验证
- 小学习率 + 小步数，防止过拟合

**预期效果**：可能有过拟合风险，建议多保存几个 checkpoint，逐个试听选最好的

### 12.2 如果训练效果不好的排查清单

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不降 | 学习率太大/太小 | 尝试 5e-6 ~ 5e-5 范围 |
| Loss 震荡 | batch_tokens 太小 | 增大 batch_tokens 或增加 gradient_accumulation_steps |
| 生成音频质量差 | 数据质量问题 | 检查音频是否干净、文本是否准确 |
| 过拟合（训练 Loss 低但效果差） | 数据太少/学习率太大 | 增加数据、减小学习率、早停 |
| CUDA OOM | 显存不足 | 减小 batch_tokens、改用 fp16 |
| 生成语音不像目标人 | 参考音频太短/质量差 | 用 5~10 秒高质量参考音频 |
| 微调后 Voice Design 失效 | drop_cond_ratio/prompt_ratio 太极端 | 回调到默认值 0.1 / [0.0, 0.3] |

---

## 13. 常见问题 FAQ

### Q1: 数据格式里没有 speaker_id，怎么做 Voice Clone？

**OmniVoice 不需要 speaker_id。** 它的训练方式是**自监督**的：从每段音频的前缀部分自动提取"参考音频"，让模型学习"模仿这段参考音频的声音"。

所以：
- 数据里没有 `speaker_id` **完全不影响** Voice Clone
- 推理时你仍然需要提供 `ref_audio`，模型根据这段音频做 clone
- 如果你的数据全是同一个人，模型会**隐式地**更熟悉这个人的声音

### Q2: 微调后预训练模型的 Voice Clone 能力会丢失吗？

**一般不会。** 预训练模型的 zero-shot Voice Clone 能力在微调后通常保留。

只有在以下情况可能退化：
- 数据量极少（< 10 分钟）且全是单一人
- 学习率太大（> 1e-4）
- 步数太多导致严重过拟合
- `drop_cond_ratio` 和 `prompt_ratio_range` 调得太极端

如果你担心，可以保存几个中间 checkpoint，测试不同阶段的通用 clone 能力。

### Q3: 我没有 GPU，可以用 CPU 训练吗？

**理论上可以，但不推荐**。CPU 训练速度比 GPU 慢几十到几百倍，微调可能需要几天甚至几周。建议：
- 使用云服务器（阿里云、腾讯云、AutoDL 等，租一张 RTX 4090 约 1~2 元/小时）
- 使用 Google Colab（有免费的 T4 GPU）

### Q4: 微调后的模型可以商用吗？

OmniVoice 使用 Apache-2.0 开源协议，**可以商用**。但请注意：
- 不要用于未经授权的声音克隆
- 遵守当地法律法规

### Q5: 微调后的模型文件有多大？

每个 checkpoint 约 **1.2 GB**（和预训练模型一样大）。

### Q6: 可以只微调模型的某一部分吗？

目前 OmniVoice 的训练代码是**全参数微调**（所有参数都更新）。未来可能会支持 LoRA 等高效微调方法。

### Q7: 训练中断后可以恢复吗？

可以！修改配置文件：

```json
{
    "resume_from_checkpoint": "exp/my_first_finetune/checkpoint-2500"
}
```

然后重新运行训练命令即可从中断处继续。

### Q8: 为什么我的验证集 Loss 比训练集高很多？

这是正常的！如果差距不大（< 30%），说明模型泛化能力 OK。如果差距很大，可能是过拟合，需要：
- 增加数据量
- 减小学习率
- 减少训练步数

### Q9: 如何评估微调效果？

除了听感评估，还可以用客观指标：

```bash
# 安装评估依赖
pip install jiwer s3prl funasr

# 运行评估（需要有测试集和参考音频）
# 具体方法参见 docs/evaluation.md
```

主要指标：
- **WER（词错误率）**：衡量发音准确度，越低越好
- **Speaker Similarity（说话人相似度）**：衡量克隆相似度，越高越好
- **UTMOS**：衡量自然度，越高越好

### Q10: 微调需要多久？

| 数据量 | GPU | 训练步数 | 预计时间 |
|--------|-----|---------|---------|
| 30 分钟 | 1x RTX 4090 | 3000 | ~1 小时 |
| 1 小时 | 2x A100 | 5000 | ~30 分钟 |
| 10 小时 | 4x A100 | 10000 | ~2 小时 |

### Q11: 如何同时训练多个语言？

在 JSONL 中给每个样本标注正确的 `language_id`，然后在 `data_config.json` 中：

```json
{
    "train": [
        {
            "language_id": "zh",
            "manifest_path": ["data/zh/data.lst"],
            "repeat": 1
        },
        {
            "language_id": "en",
            "manifest_path": ["data/en/data.lst"],
            "repeat": 1
        }
    ]
}
```

### Q12: 我想在 Windows 上训练，可以吗？

**推荐用 WSL2**（Windows Subsystem for Linux）：
1. 在 Windows 上安装 WSL2 + Ubuntu
2. 在 WSL2 中按照本指南的 Linux 步骤操作
3. 确保 WSL2 能访问你的 NVIDIA GPU（安装 WSL 版 CUDA 驱动）

直接在 Windows 命令行中运行可能会遇到各种路径、依赖问题。

### Q13: 模型下载到哪里了？如何修改下载位置？

**默认下载位置**：

```bash
# HuggingFace 默认缓存
~/.cache/huggingface/hub/models--k2-fsa--OmniVoice/

# 如果通过 ModelScope 镜像下载
~/.cache/modelscope/k2-fsa/OmniVoice/
```

模型文件约 **3.1GB**，包含 `model.safetensors`（2.3GB 权重）、tokenizer、audio_tokenizer 等。

**修改下载位置**：

```bash
# 方式1：修改 HuggingFace 缓存目录
export HF_HOME="/path/to/your/cache"
export HF_HUB_CACHE="/path/to/your/cache/hub"

# 方式2：修改 ModelScope 缓存目录
export MODELSCOPE_CACHE="/path/to/your/cache"

# 方式3：加载时直接指定本地路径（推荐）
model = OmniVoice.from_pretrained("/path/to/local/model")
```

### Q14: 微调后如何部署为服务？

```bash
# 使用 Gradio 启动 Web 服务
omnivoice-demo --model exp/my_first_finetune/checkpoint-5000 --ip 0.0.0.0 --port 8001
```

或者在 Python 中加载后封装为 FastAPI 服务：

```python
from fastapi import FastAPI
from omnivoice import OmniVoice
import torch

app = FastAPI()
model = OmniVoice.from_pretrained("exp/my_first_finetune/checkpoint-5000", device_map="cuda:0")

@app.post("/clone")
def clone(text: str, ref_audio: str, ref_text: str):
    audio = model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text)
    return {"audio": audio[0].tolist()}
```

---

## 附录：完整的训练流程速查表

```bash
# 1. 准备数据 → my_data_train.jsonl + my_data_dev.jsonl

# 2. Tokenize
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl my_data_train.jsonl \
    --tar_output_pattern "data/finetune/tokens/train/audios/shard-%06d.tar" \
    --jsonl_output_pattern "data/finetune/tokens/train/txts/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3

python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl my_data_dev.jsonl \
    --tar_output_pattern "data/finetune/tokens/dev/audios/shard-%06d.tar" \
    --jsonl_output_pattern "data/finetune/tokens/dev/txts/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3

# 3. 创建配置文件 → my_train_config.json + my_data_config.json

# 4. 启动训练
accelerate launch \
    --gpu_ids "0,1" \
    --num_processes 2 \
    -m omnivoice.cli.train \
    --train_config my_train_config.json \
    --data_config my_data_config.json \
    --output_dir exp/my_finetune

# 5. 监控训练
tensorboard --logdir exp/my_finetune/tensorboard

# 6. 推理测试（用微调后的 checkpoint）
python -c "
from omnivoice import OmniVoice
import soundfile as sf
import torch
model = OmniVoice.from_pretrained('exp/my_finetune/checkpoint-5000', device_map='cuda:0', dtype=torch.float16)
audio = model.generate(text='你好，微调成功！', ref_audio='ref.wav', ref_text='参考文本', language='zh')
sf.write('output.wav', audio[0], 24000)
"
```

---

## 参考资源

- **官方 README**: [README.md](README.md)
- **训练配置详解**: [docs/training.md](docs/training.md)
- **数据准备详解**: [docs/data_preparation.md](docs/data_preparation.md)
- **生成参数说明**: [docs/generation-parameters.md](docs/generation-parameters.md)
- **使用技巧**: [docs/tips.md](docs/tips.md)
- **声音设计指南**: [docs/voice-design.md](docs/voice-design.md)
- **在线 Demo**: [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)
- **Colab 笔记本**: [docs/OmniVoice.ipynb](docs/OmniVoice.ipynb)

---

**祝你微调顺利！如果遇到问题，欢迎在 [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues) 提问。**
