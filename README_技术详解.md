# OmniVoice 技术完全手册（深度版）

> 本文档是对 OmniVoice 项目的**超详细**技术总结，深入每一行代码和每一个设计决策。涵盖底层架构、数学原理、完整训练推理流程、配置文件示例、内存分析、调试技巧。适合进行二次开发、模型改进和论文复现。

---

## 目录

1. [项目概述](#1-项目概述)
2. [Qwen3-0.6B 基础模型剖析](#2-qwen3-06b-基础模型剖析)
3. [核心架构：OmniVoice 的 LLM 改造](#3-核心架构omnivoice-的-llm-改造)
4. [音频表示：Higgs Audio V2 + RVQ](#4-音频表示higgs-audio-v2--rvq)
5. [模型类逐行详解](#5-模型类逐行详解)
6. [训练流程：从原始音频到模型更新](#6-训练流程从原始音频到模型更新)
7. [推理流程：从文本到波形的完整链路](#7-推理流程从文本到波形的完整链路)
   - [7.5 副语言控制（停顿、笑声、语气）](#75-副语言控制停顿笑声语气)
   - [7.6 情绪控制](#76-情绪控制)
8. [数据流水线：逐模块解析](#8-数据流水线逐模块解析)
9. [微调实战：儿童语音案例](#9-微调实战儿童语音案例)
10. [评估体系完整指南](#10-评估体系完整指南)
11. [CLI 工具源码解析](#11-cli-工具源码解析)
12. [性能优化与内存分析](#12-性能优化与内存分析)
13. [完整配置文件示例](#13-完整配置文件示例)
14. [调试技巧与问题排查](#14-调试技巧与问题排查)
15. [与同类模型的深度对比](#15-与同类模型的深度对比)
16. [OmniVoiceGenerationConfig 完整参数详解](#16-omnivoicegenerationconfig-完整参数详解)
17. [Voice Clone 语音克隆机制](#17-voice-clone-语音克隆机制)
18. [多语言与指令系统](#18-多语言与指令系统)
19. [音频 Tokenizer 架构详解](#19-音频-tokenizer-架构详解)
20. [训练时的随机掩码与 CFG 训练原理](#20-训练时的随机掩码与-cfg-训练原理)
21. [Checkpoint 文件结构与模型加载](#21-checkpoint-文件结构与模型加载)

---

## 1. 项目概述

### 1.1 定位

OmniVoice 是一个**基于离散扩散的大规模多语言零样本 TTS 系统**。它不是自回归模型（如 VALL-E），而是将 LLM 改造为双向注意力的掩码预测模型，通过多步迭代去掩码生成音频 token。

### 1.2 核心创新点

| 创新点 | 传统 TTS | OmniVoice |
|--------|---------|-----------|
| 生成方式 | 自回归（逐 token） | **离散扩散（迭代去掩码，32 步）** |
| 注意力 | 因果（只能看左边） | **双向（BERT 式，全局可见）** |
| 说话人信息 | 显式 speaker_id / d-vector | **隐式 prompt（音频前缀）** |
| 推理速度 | RTF 0.5~1.0 | **RTF 0.025（40x 实时）** |
| Voice Design | 多数不支持 | **通过文本指令控制** |

### 1.3 完整技术栈

```
┌──────────────────────────────────────────────────────────────┐
│                        OmniVoice 技术栈                        │
├──────────────────────────────────────────────────────────────┤
│  音频编解码层                                                  │
│    Higgs Audio V2 Tokenizer (Encoder → RVQ → Decoder)         │
│    采样率 24kHz, 帧率 75Hz, 8 层 RVQ, 每层 1024 码本           │
├──────────────────────────────────────────────────────────────┤
│  语言模型层                                                    │
│    Qwen3-0.6B (双向注意力, 28 层, 1536 hidden, 12 头)         │
│    + Audio Embeddings (8×1025 → 1536, 共享表)                │
│    + Audio Heads (1536 → 8×1025, 线性投影)                    │
├──────────────────────────────────────────────────────────────┤
│  训练框架层                                                    │
│    HuggingFace Accelerate (DDP/DeepSpeed/bf16)               │
│    WebDataset (tar shard 格式)                                 │
│    Sequence Packing / Length Grouping                         │
├──────────────────────────────────────────────────────────────┤
│  推理引擎层                                                    │
│    迭代去掩码解码 (32 步)                                      │
│    Classifier-Free Guidance (guidance_scale=2.0)             │
│    长文本分块 (30s 阈值, 15s 块大小)                           │
│    后处理 (去静音, 音量匹配, 淡入淡出)                         │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 项目文件结构

```
omnivoice/
├── __init__.py                    # 导出 OmniVoice, OmniVoiceConfig
├── cli/                           # 命令行入口
│   ├── demo.py                    # Gradio Web UI (语音克隆 + 声音设计)
│   ├── infer.py                   # 单条推理 CLI
│   ├── infer_batch.py             # 批量推理 (多 GPU 分布式)
│   └── train.py                   # 训练入口 (读取 config, 启动 trainer)
├── data/                          # 数据流水线
│   ├── dataset.py                 # WebDataset/JSONL 读取, SampleDecoder
│   ├── processor.py               # 样本处理 (mask, tokenize, drop_cond)
│   ├── collator.py                # PaddingDataCollator / PackingDataCollator
│   └── batching.py                # PackingIterableDataset / StreamLengthGroupDataset
├── models/
│   └── omnivoice.py               # 核心模型 (~1600 行, 所有逻辑在此)
├── training/                      # 训练基础设施
│   ├── config.py                  # TrainingConfig 数据类
│   ├── builder.py                 # 构建 model/tokenizer/dataloader
│   ├── trainer.py                 # OmniTrainer (Accelerate 封装)
│   └── checkpoint.py              # save/load checkpoint, TrainLogger
├── eval/                          # 评估工具
│   ├── wer/                       # WER 计算 (多语言文本归一化 + ASR)
│   ├── speaker_similarity/        # ECAPA-TDNN + WavLM 说话人相似度
│   ├── mos/                       # UTMOS 自然度评分
│   └── models/                    # 预训练评估模型 (ECAPA, UTMOS)
├── scripts/                       # 数据预处理脚本
│   ├── extract_audio_tokens.py    # 音频 → RVQ token (多进程 GPU 加速)
│   ├── extract_audio_tokens_add_noise.py  # 带噪声增强的版本
│   ├── jsonl_to_webdataset.py     # JSONL → WebDataset tar shard
│   └── denoise_audio.py           # 音频降噪工具
└── utils/                         # 工具函数
    ├── audio.py                   # 加载/重采样/去静音/淡入淡出/cross-fade
    ├── text.py                    # 文本分块/标点处理
    ├── duration.py                # 基于规则的音频时长估计
    ├── lang_map.py                # 600+ 语言代码映射
    └── voice_design.py            # 声音设计指令解析与验证
```

---

## 2. Qwen3-0.6B 基础模型剖析

OmniVoice 不是从零训练，而是加载 Qwen3-0.6B 的权重，只新增两个模块。理解 Qwen3-0.6B 的架构是理解 OmniVoice 的基础。

### 2.1 Qwen3-0.6B 配置

```json
// 从 transformers 加载的 llm_config
{
    "model_type": "qwen3",
    "vocab_size": 151936,           // 词表大小
    "hidden_size": 1536,            // 隐藏层维度 (D=1536)
    "num_hidden_layers": 28,        // Transformer 层数
    "num_attention_heads": 12,      // 注意力头数
    "num_key_value_heads": 12,      // GQA (此处等于头数, 非 GQA)
    "intermediate_size": 4864,      // FFN 中间层维度
    "max_position_embeddings": 40960, // 最大序列长度
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,        // RoPE 基频
    "use_sliding_window": false,
    "attention_dropout": 0.0,
    "torch_dtype": "bfloat16"
}
```

### 2.2 参数量计算

| 组件 | 计算公式 | 参数量 |
|------|---------|--------|
| Token Embedding | 151936 × 1536 | 233.4M |
| 28 层 Transformer | 28 × (4 × 1536² + 2 × 1536 × 4864) | ~653M |
| RMSNorm / Bias | 小量 | ~1M |
| **Qwen3-0.6B 总计** | | **~0.6B** |
| Audio Embeddings | 8200 × 1536 | 12.6M |
| Audio Heads | 1536 × 8200 | 12.6M |
| **OmniVoice 新增** | | **~25M** |
| **OmniVoice 总计** | | **~0.625B** |

### 2.3 注意力机制：双向 vs 因果

**GPT（因果/自回归）**：
```
位置 i 只能看到位置 0~i（三角形 mask）
适用于：逐 token 预测下一个
```

**OmniVoice（双向/掩码预测）**：
```
位置 i 能看到所有位置（方形 mask）
适用于：填空（masked token prediction）
```

这是 OmniVoice 与 VALL-E/Bark 的根本区别。双向注意力让模型在每一步都能看到全部上下文，因此可以用更少的步数（32 步）完成生成，而自回归需要 750+ 步（对应 10 秒音频的 token 数）。

---

## 3. 核心架构：OmniVoice 的 LLM 改造

### 3.1 改造点一：Audio Embeddings

**文件位置**：`omnivoice/models/omnivoice.py:216-223`

```python
self.audio_embeddings = nn.Embedding(
    config.num_audio_codebook * config.audio_vocab_size,  # 8 * 1025 = 8200
    self.config.llm_config.hidden_size,                    # 1536
)
self.register_buffer(
    "codebook_layer_offsets",
    torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
    # tensor([0, 1025, 2050, 3075, 4100, 5125, 6150, 7175])
)
```

**输入处理**：`_prepare_embed_inputs`（第 360-380 行）

```python
def _prepare_embed_inputs(self, input_ids: torch.Tensor, audio_mask: torch.Tensor):
    # input_ids: [Batch, 8, SeqLen]
    # audio_mask: [Batch, SeqLen], True=音频位置, False=文本位置
    
    # 1. 文本位置：查 Text Embedding，只取第 0 层
    text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])  # [B, S, 1536]
    
    # 2. 音频位置：给每层 ID 加上偏移量，然后查表求和
    #    input_ids 中音频位置有真实 ID，文本位置是 0（会被 mask 掉）
    shifted_ids = (
        input_ids * audio_mask.unsqueeze(1)   # 文本位置清零
    ) + self.codebook_layer_offsets.view(1, -1, 1)  # [B, 8, S]
    
    # 8 层分别查表，然后 sum → [B, S, 1536]
    audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
    
    # 3. 根据 audio_mask 选择：文本位置用 text_embeds，音频位置用 audio_embeds
    return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
    # 输出: [B, S, 1536]
```

**为什么 8 层共享一个表？**

假设每层独立表：
- 参数量：8 × 1025 × 1536 = 12,595,200

共享一个大表：
- 参数量：1 × 8200 × 1536 = 12,595,200

参数量相同！但共享表的好处是：
1. **跨层知识共享**：Layer 0 和 Layer 1 学到的特征可以互相影响
2. **正则化效果**：相当于对 embedding 做了隐式的参数共享约束
3. **实现简洁**：只需要一个 nn.Embedding

### 3.2 改造点二：Audio Heads

**文件位置**：`omnivoice/models/omnivoice.py:225-228`

```python
self.audio_heads = nn.Linear(
    self.config.llm_config.hidden_size,          # 1536
    config.num_audio_codebook * config.audio_vocab_size,  # 8 * 1025 = 8200
    bias=False,
)
```

**前向传播**：`forward` 方法（第 382-461 行）

```python
# LLM 输出 hidden_states: [B, S, 1536]
hidden_states = llm_outputs[0]

# Audio Heads: [B, S, 1536] → [B, S, 8200]
logits_flat = self.audio_heads(hidden_states)

# reshape: [B, S, 8, 1025] → [B, 8, S, 1025]
audio_logits = logits_flat.view(
    batch_size, seq_len,
    self.config.num_audio_codebook,      # 8
    self.config.audio_vocab_size,        # 1025
).permute(0, 2, 1, 3)
```

### 3.3 损失计算详解

**文件位置**：`omnivoice/models/omnivoice.py:434-456`

```python
# audio_logits: [B, 8, S, 1025]
# labels: [B, 8, S], -100 表示不算 loss

# 1. 对每一层、每一个位置计算交叉熵
#    permute: [B, 8, S, 1025] → [B, 1025, 8, S]
per_token_loss = F.cross_entropy(
    audio_logits.permute(0, 3, 1, 2),  # [B, 1025, 8, S]
    labels,                             # [B, 8, S]
    reduction="none",
    ignore_index=-100
)  # 输出: [B, 8, S]

# 2. 只保留有效位置（labels != -100）
valid_mask = (labels != -100).float()   # [B, 8, S]

# 3. 每层分别求平均
#    sum over (Batch, Seq) / count → [8]
layer_means = (per_token_loss * valid_mask).sum(dim=(0, 2)) / \
              valid_mask.sum(dim=(0, 2)).clamp(min=1.0)

# 4. 加权求和
weights = torch.tensor(
    [8/40, 8/40, 6/40, 6/40, 4/40, 4/40, 2/40, 2/40],
    device=audio_logits.device
)
loss = (layer_means * weights).sum()
```

**权重设计原理**：

| 层 | 权重 | 占比 | 负责内容 | 为什么权重高 |
|---|------|------|---------|------------|
| 0 | 8 | 20% | 最粗粒度：基频 F0、整体频谱包络 | 决定"谁在说"和"说什么" |
| 1 | 8 | 20% | 粗粒度：共振峰结构 | 决定音色主体 |
| 2 | 6 | 15% | 中等粒度：过渡音、连读 | 影响流畅度 |
| 3 | 6 | 15% | 中等粒度：音节边界 | 影响清晰度 |
| 4 | 4 | 10% | 细粒度：摩擦音细节 | 补充细节 |
| 5 | 4 | 10% | 细粒度：爆破音细节 | 补充细节 |
| 6 | 2 | 5% | 最细粒度：噪声纹理 | 几乎不影响感知 |
| 7 | 2 | 5% | 最细粒度：高频噪声 | 几乎不影响感知 |

**Layer 0-1 决定了 80% 的人耳感知信息**，因此给予最高权重。这符合 RVQ 的残差结构：低层是主要信号，高层是残差补充。

---

## 4. 音频表示：Higgs Audio V2 + RVQ

### 4.1 Higgs Audio V2 Tokenizer 架构

```
原始波形 [1, T]
    │
    ▼
┌─────────────┐
│   Encoder   │  CNN + LSTM 编码器
│  (神经网络)  │
└──────┬──────┘
       ▼
连续向量 [D, T']  (T' ≈ T/320, 因为 hop_length=320 @ 24kHz)
       │
       ▼
┌─────────────┐
│    RVQ      │  8 层残差向量量化
│  (8码本)    │
└──────┬──────┘
       ▼
离散 Token [8, T']
```

**关键参数**：
- 采样率：24 kHz
- Hop length：320 样本 → 帧率 = 24000/320 = **75 Hz**
- 10 秒音频 → 750 帧 → Token 形状 `[8, 750]`

### 4.2 RVQ 量化过程

```python
# 伪代码
def rvq_encode(vector):
    """vector: [D] 连续向量"""
    residual = vector
    tokens = []
    
    for layer in range(8):
        # 在当前残差上找到最近的码本向量
        codebook = codebooks[layer]  # [1024, D]
        distances = torch.sum((codebook - residual) ** 2, dim=1)
        idx = torch.argmin(distances)  # 最近的码本索引
        
        tokens.append(idx.item())
        residual = residual - codebook[idx]  # 减去已量化部分
    
    return tokens  # [id0, id1, ..., id7]

def rvq_decode(tokens):
    """tokens: [8] 离散索引"""
    vector = torch.zeros(D)
    for layer, idx in enumerate(tokens):
        vector += codebooks[layer][idx]
    return vector
```

**关键理解**：
- Layer 0 量化了信号的主要部分（贡献最大能量）
- Layer 1 量化 Layer 0 的残差
- ...
- Layer 7 量化最微小的残差（接近噪声）
- 重建时：8 层码本向量相加 ≈ 原始连续向量

### 4.3 Token 与音频时长换算

| 音频时长 | Token 帧数 (T) | 备注 |
|---------|---------------|------|
| 1 秒 | 75 | 帧率 75Hz |
| 3 秒 | 225 | 最短有效参考音频 |
| 5 秒 | 375 | 推荐参考音频时长 |
| 10 秒 | 750 | 较长参考音频 |
| 30 秒 | 2250 | 自动分块阈值 |

---

## 5. 模型类逐行详解

### 5.1 类继承关系

```
PreTrainedModel (transformers)
    └── OmniVoice
        ├── llm: Qwen3Model (AutoModel)
        ├── audio_embeddings: nn.Embedding(8200, 1536)
        ├── audio_heads: nn.Linear(1536, 8200)
        └── [inference-only attributes]
            ├── text_tokenizer
            ├── audio_tokenizer (HiggsAudioV2TokenizerModel)
            ├── duration_estimator
            └── _asr_pipe (Whisper)
```

### 5.2 from_pretrained 加载逻辑

**文件位置**：`omnivoice/models/omnivoice.py:246-292`

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    train_mode = kwargs.pop("train", False)           # 是否训练模式
    load_asr = kwargs.pop("load_asr", False)          # 是否加载 ASR
    asr_model_name = kwargs.pop("asr_model_name", "openai/whisper-large-v3-turbo")
    
    # 1. 解析路径（本地路径或 HuggingFace Hub 下载）
    resolved_path = _resolve_model_path(pretrained_model_name_or_path)
    
    # 2. 加载模型权重
    model = super().from_pretrained(resolved_path, *args, **kwargs)
    
    # 3. 非训练模式下，加载 tokenizer 和 audio tokenizer
    if not train_mode:
        model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        
        # audio_tokenizer 路径：checkpoint/audio_tokenizer/ 或 HuggingFace 下载
        audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")
        if not os.path.isdir(audio_tokenizer_path):
            audio_tokenizer_path = _resolve_model_path("eustlb/higgs-audio-v2-tokenizer")
        
        model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_path,
            device_map="cpu" if mps else model.device  # MPS 不支持
        )
        model.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_tokenizer_path)
        model.sampling_rate = model.feature_extractor.sampling_rate  # 24000
        model.duration_estimator = RuleDurationEstimator()
        
        if load_asr:
            model.load_asr_model(model_name=asr_model_name)
    
    return model
```

**关键点**：
- `train=True` 时**不加载** audio_tokenizer 和 text_tokenizer（训练不需要）
- audio_tokenizer 默认在模型目录的 `audio_tokenizer/` 子目录
- 如果本地没有，自动从 HuggingFace 下载 `eustlb/higgs-audio-v2-tokenizer`

### 5.3 generate() 方法完整流程

**文件位置**：`omnivoice/models/omnivoice.py:475-601`

```python
def generate(self, text, language=None, ref_text=None, ref_audio=None,
             voice_clone_prompt=None, instruct=None, duration=None,
             speed=None, generation_config=None, **kwargs):
    
    # 1. 合并 generation_config 和 kwargs
    gen_config = generation_config or OmniVoiceGenerationConfig.from_dict(kwargs)
    
    # 2. 预处理所有输入 → GenerationTask
    #    包括：文本处理、参考音频转 token、时长估计、语言解析
    full_task = self._preprocess_all(text=text, language=language, ...)
    
    # 3. 分长短文本
    #    short_idx: 估计音频 ≤ 30 秒
    #    long_idx: 估计音频 > 30 秒（需要分块）
    short_idx, long_idx = full_task.get_indices(gen_config, frame_rate)
    
    # 4. 短文本：直接迭代生成
    if short_idx:
        short_task = full_task.slice_task(short_idx)
        short_results = self._generate_iterative(short_task, gen_config)
    
    # 5. 长文本：分块生成
    if long_idx:
        long_task = full_task.slice_task(long_idx)
        long_results = self._generate_chunked(long_task, gen_config)
    
    # 6. 解码 token → 波形，后处理
    for i in range(batch_size):
        audio = self._decode_and_post_process(
            results[i], full_task.ref_rms[i], gen_config
        )
    
    return generated_audios  # list of np.ndarray
```

### 5.4 _generate_iterative 核心算法

**文件位置**：`omnivoice/models/omnivoice.py:1145-1297`

这是 OmniVoice 推理的**核心算法**， worth 逐行解析：

```python
def _generate_iterative(self, task: GenerationTask, gen_config):
    B = task.batch_size
    
    # === Step 1: 准备输入序列 ===
    # 每个样本构造：
    #   [Style] + [Text] + [Ref_Audio] + [Target(全MASK)]
    inputs_list = [self._prepare_inference_inputs(...) for i in range(B)]
    
    # === Step 2: Padding + 构造 Conditional & Unconditional ===
    # 合并成 2B 的 batch：前 B 个是 conditional，后 B 个是 unconditional
    batch_input_ids = torch.full((2*B, 8, max_len), MASK_ID)
    batch_audio_mask = torch.zeros((2*B, max_len), dtype=torch.bool)
    batch_attention_mask = torch.zeros((2*B, 1, max_len, max_len), dtype=torch.bool)
    
    for i in range(B):
        c_len = inputs_list[i]["input_ids"].size(2)
        u_len = task.target_lens[i]  # 目标区域长度
        
        # Conditional (0~B-1)：完整序列
        batch_input_ids[i, :, :c_len] = inputs_list[i]["input_ids"]
        batch_audio_mask[i, :c_len] = inputs_list[i]["audio_mask"]
        batch_attention_mask[i, :, :c_len, :c_len] = True
        
        # Unconditional (B~2B-1)：只保留目标区域（去掉 ref_audio 和 style）
        batch_input_ids[B+i, :, :u_len] = inputs_list[i]["input_ids"][..., -u_len:]
        batch_audio_mask[B+i, :u_len] = inputs_list[i]["audio_mask"][..., -u_len:]
        batch_attention_mask[B+i, :, :u_len, :u_len] = True
    
    # === Step 3: 初始化目标 token（全 MASK）===
    tokens = torch.full((B, 8, max_target_len), MASK_ID)
    
    # === Step 4: 计算时间步调度 ===
    # 非线性调度：前期 reveal 少，后期 reveal 多
    timesteps = _get_time_steps(num_step=gen_config.num_step, t_shift=gen_config.t_shift)
    schedules = []  # 每个样本每步 reveal 多少
    for t_len in task.target_lens:
        total_mask = t_len * 8  # 总 mask 数
        rem = total_mask
        sched = []
        for step in range(num_step):
            if step == num_step - 1:
                num = rem  # 最后一步全部 reveal
            else:
                num = min(ceil(total_mask * (timesteps[step+1] - timesteps[step])), rem)
            sched.append(num)
            rem -= num
        schedules.append(sched)
    
    # === Step 5: N 步迭代 ===
    layer_ids = torch.arange(8).view(1, -1, 1)  # [1, 8, 1]
    
    for step in range(gen_config.num_step):
        # 5.1 前向传播（一次处理 2B 个样本）
        batch_logits = self(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)  # [2B, 8, S, 1025]
        
        for i in range(B):
            k = schedules[i][step]  # 本轮 reveal 数量
            if k <= 0: continue
            
            c_len, t_len = c_lens[i], task.target_lens[i]
            
            # 5.2 提取 conditional 和 unconditional 的 logits
            c_logits = batch_logits[i:i+1, :, c_len-t_len:c_len, :]      # [1, 8, T, 1025]
            u_logits = batch_logits[B+i:B+i+1, :, :t_len, :]             # [1, 8, T, 1025]
            
            # 5.3 CFG + 预测 token
            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_logits, u_logits, gen_config
            )  # pred_tokens: [1, 8, T], scores: [1, 8, T]
            
            # 5.4 Layer Penalty：给高层加惩罚
            #     Layer 0 分数不变, Layer 7 分数 -35
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            
            # 5.5 位置随机性（Gumbel 采样）
            if gen_config.position_temperature > 0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)
            
            # 5.6 只选当前还是 MASK 的位置
            sample_tokens = tokens[i:i+1, :, :t_len]
            scores.masked_fill_(sample_tokens != MASK_ID, -float("inf"))
            
            # 5.7 Top-k：选置信度最高的 k 个位置 reveal
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))
            
            # 5.8 更新 batch 输入，下一轮使用
            tokens[i:i+1, :, :t_len] = sample_tokens
            batch_input_ids[i:i+1, :, c_len-t_len:c_len] = sample_tokens
            batch_input_ids[B+i:B+i+1, :, :t_len] = sample_tokens
    
    return [tokens[i, :, :task.target_lens[i]] for i in range(B)]
```

### 5.5 _predict_tokens_with_scoring：CFG 实现

**文件位置**：`omnivoice/models/omnivoice.py:1299-1322`

```python
def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
    if gen_config.guidance_scale != 0:
        # log-space CFG 更稳定
        c_log_probs = F.log_softmax(c_logits, dim=-1)      # [1, 8, T, 1025]
        u_log_probs = F.log_softmax(u_logits, dim=-1)      # [1, 8, T, 1025]
        
        # CFG 公式：log P_cond + scale * (log P_cond - log P_uncond)
        log_probs = torch.log_softmax(
            c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
            dim=-1,
        )
    else:
        log_probs = F.log_softmax(c_logits, dim=-1)
    
    # 禁止选择 MASK token (ID=1024)
    log_probs[..., self.config.audio_mask_id] = -float("inf")
    
    # 采样策略
    if gen_config.class_temperature > 0:
        # Top-k 采样：只保留前 10% 的概率质量
        filtered_probs = _filter_top_k(log_probs, ratio=0.1)
        pred_tokens = _gumbel_sample(filtered_probs, gen_config.class_temperature).argmax(dim=-1)
    else:
        # 贪心：选概率最大的
        pred_tokens = log_probs.argmax(dim=-1)
    
    # 置信度分数（用于选择 reveal 位置）
    confidence_scores = log_probs.max(dim=-1)[0]
    
    return pred_tokens, confidence_scores
```

**CFG 的数学原理**：

标准分类器引导（Classifier Guidance）需要训练一个额外的分类器：
```
∇log p(x|y) = ∇log p(x) + ∇log p(y|x)
```

CFG（Classifier-Free Guidance）巧妙地避免了额外分类器：
```
∇log p(x|y) ≈ ∇log p(x) + s * (∇log p(x|y) - ∇log p(x))
```

其中 `s = guidance_scale`：
- s=1：等于无条件生成（不用条件）
- s=2：条件强度加倍（默认，平衡）
- s>2：更强的条件控制，但可能过度收敛

OmniVoice 在 log-probability 空间直接做：
```python
log_probs = c_log_probs + s * (c_log_probs - u_log_probs)
```

---

## 6. 训练流程：从原始音频到模型更新

### 6.1 完整数据流

```
原始音频文件 (.wav)
    │
    ▼
extract_audio_tokens.py  (GPU 多进程)
    │  1. 加载音频 → 重采样到 24kHz
    │  2. Higgs Audio V2 Tokenizer 编码 → [8, T]
    │  3. 保存为 .npy 到 tar shard
    ▼
WebDataset tar shards (audios/shard-*.tar)
    │
    ▼
WebDatasetReader + SampleDecoder
    │  1. 从 tar 中读取 .npy (audio_tokens) 和 label
    ▼
OmniVoiceSampleProcessor
    │  1. drop_cond? (10% 概率)
    │  2. prompt_ratio = random(0.0, 0.3)
    │  3. mask_ratio = random(0.0, 1.0)
    │  4. 构造 style tokens (<|lang_start|>zh<|lang_end|>...)
    │  5. 构造 text tokens (<|text_start|>文本<|text_end|>)
    │  6. mask 目标区域
    │  7. 拼接 input_ids + labels + audio_mask
    ▼
Collator (Packing / Padding)
    │  flex_attention: 拼接成长序列 [1, 8, batch_tokens]
    │  sdpa: padding 到 max_len [B, 8, max_len]
    ▼
OmniTrainer (Accelerate)
    │  1. forward → loss
    │  2. backward
    │  3. clip_grad_norm
    │  4. optimizer.step()
    │  5. lr_scheduler.step()
    ▼
Checkpoint 保存
```

### 6.2 Processor 详解

**文件位置**：`omnivoice/data/processor.py:66-172`

```python
def __call__(self, sample):
    # sample = {"audio_tokens": [8, T], "label": {"text": "...", "language_id": "zh"}}
    
    # 1. 决定是否 drop_cond
    drop_cond = random() < self.drop_cond_ratio  # 默认 10%
    
    if drop_cond:
        prompt_ratio = 0.0
        drop_text = True
        use_language = False
        use_instruct = False
    else:
        prompt_ratio = random.uniform(*self.prompt_ratio_range)  # [0.0, 0.3]
        drop_text = False
        use_language = random() < self.language_ratio  # 80%
        use_instruct = random() < self.instruct_ratio  # 100%
        if use_instruct and random() < self.only_instruct_ratio:  # 50%
            prompt_ratio = 0.0  # 只用 instruct，不用 prompt
    
    mask_ratio = random.uniform(*self.mask_ratio_range)  # [0.0, 1.0]
    
    # 2. 构造 Style
    language = sample["label"].get("language_id", "None") if use_language else "None"
    instruct = sample["label"].get("instruct", "None") if use_instruct else "None"
    
    style = f"<|lang_start|>{language}<|lang_end|>"
    style += f"<|instruct_start|>{instruct}<|instruct_end|>"
    
    style_tokens = text_tokenizer(style).input_ids.repeat(8, 1)  # [8, N1]
    style_labels = torch.full_like(style_tokens, -100)  # 不算 loss
    
    # 3. 构造 Text
    text = sample["label"]["text"]
    text_tokens = text_tokenizer(f"<|text_start|>{text}<|text_end|>").input_ids.repeat(8, 1)
    text_labels = torch.full_like(text_tokens, -100)
    
    # 4. 处理 Audio
    audio_tokens = sample["audio_tokens"]  # [8, T]
    prompt_length = int(audio_tokens.shape[1] * prompt_ratio)  # prompt 长度
    
    audio_inputs = audio_tokens.clone()
    audio_labels = audio_tokens.clone()
    
    # 5. Mask 目标区域
    maskable_region = audio_tokens[:, prompt_length:]  # prompt 之后可 mask
    token_mask = torch.rand(maskable_region.shape) < mask_ratio
    audio_inputs[:, prompt_length:][token_mask] = MASK_ID  # 1024
    audio_labels[:, prompt_length:][~token_mask] = -100   # 未 mask 不算 loss
    
    if not drop_cond:
        audio_labels[:, :prompt_length] = -100  # prompt 不算 loss
    
    # 6. 拼接
    if drop_text:
        input_ids = audio_inputs                          # [8, T]
        labels = audio_labels                             # [8, T]
        audio_mask = torch.ones(T, dtype=torch.bool)      # 全是音频
    else:
        input_ids = torch.cat([style_tokens, text_tokens, audio_inputs], dim=1)  # [8, N1+N2+T]
        labels = torch.cat([style_labels, text_labels, audio_labels], dim=1)
        audio_mask = torch.zeros(N1+N2+T, dtype=torch.bool)
        audio_mask[N1+N2:] = True  # 音频区域
    
    return {"input_ids": input_ids, "labels": labels, "audio_mask": audio_mask, "length": seq_len}
```

### 6.3 训练时的输入序列可视化

**正常样本**（90% 概率，drop_cond=False）：
```
input_ids [8, TotalLen]:
┌──────────────────────────────────────────────────────────────┐
│ [Style]       [Text]          [Prompt Audio]  [Masked Target] │
│ 20 tokens     50 tokens       200 tokens      600 tokens     │
└──────────────────────────────────────────────────────────────┘

labels [8, TotalLen]:
┌──────────────────────────────────────────────────────────────┐
│ [-100]        [-100]          [-100]          [真实Token]     │
│ (不算loss)    (不算loss)      (不算loss)      (算loss)       │
└──────────────────────────────────────────────────────────────┘

audio_mask [TotalLen]:
[False, False, ..., False, True, True, ..., True]
 ← style + text →      ←      audio 区域      →
```

**无条件样本**（10% 概率，drop_cond=True）：
```
input_ids [8, T]:
┌──────────────────────────────────────┐
│ [Prompt Audio]  [Masked Target]      │
│ 0 tokens        800 tokens           │  ← 注意：没有 style 和 text！
└──────────────────────────────────────┘

labels [8, T]:
┌──────────────────────────────────────┐
│ [-100]          [真实Token]           │
└──────────────────────────────────────┘

audio_mask [T]:
[True, True, ..., True]  ← 全是音频
```

### 6.4 Sequence Packing 详解

**适用场景**：`flex_attention`（PyTorch ≥ 2.5，NVIDIA Ampere+ GPU）

**原理**：将多个样本拼接成一个长序列，用 `document_ids` 区分不同样本，避免它们互相看到对方的 token。

```python
# 样本 1: [S1, T1, A1] 长度 100
# 样本 2: [S2, T2, A2] 长度 150
# 样本 3: [S3, T3, A3] 长度 120

# Packing 后：
input_ids: [1, 8, 8192]  ← 三个样本拼接
[样本1(100) | 样本2(150) | 样本3(120) | padding(8422)]

document_ids: [1, 8192]
[0,0,...,0, 1,1,...,1, 2,2,...,2, -1,-1,...]  ← -1 是 padding

# Attention Mask (flex_attention):
def _mask_mod_packed(document_ids, b, h, q_idx, kv_idx):
    return document_ids[q_idx] == document_ids[kv_idx]
    # 只关注同一个 document 内的 token
```

**优势**：
- 无 padding 浪费
- GPU 利用率更高
- 等效 batch_size 更大

**劣势**：
- 需要 PyTorch ≥ 2.5
- 需要 Ampere+ GPU（支持 flex_attention）

### 6.5 训练循环详解

**文件位置**：`omnivoice/training/trainer.py:243-354`

```python
def train(self):
    while self.global_step < self.config.steps:
        # 1. 取 batch
        batch = next(train_iterator)
        batch = _to_device(batch, self.accelerator.device)
        
        # 2. 梯度累积（accumulate 阶段不更新参数）
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.accelerator.backward(loss)  # 计算梯度
        
        # 3. 同步梯度时更新参数
        if self.accelerator.sync_gradients:
            # 梯度裁剪
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self.accelerator.log({"train/loss": avg_loss, ...})
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                self.evaluate()
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(self.global_step)
```

---

## 7. 推理流程：从文本到波形的完整链路

### 7.1 推理时的输入序列构造

**文件位置**：`omnivoice/models/omnivoice.py:1064-1143`

```python
def _prepare_inference_inputs(self, text, num_target_tokens, ref_text=None,
                               ref_audio_tokens=None, lang=None, instruct=None, denoise=True):
    
    # 1. Style Tokens
    style_text = ""
    if denoise and ref_audio_tokens is not None:
        style_text += "<|denoise|>"
    style_text += f"<|lang_start|>{lang or 'None'}<|lang_end|>"
    style_text += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    
    style_tokens = text_tokenizer(style_text).input_ids.repeat(8, 1).unsqueeze(0)  # [1, 8, N1]
    
    # 2. Text Tokens（ref_text 和 text 拼接！）
    full_text = ref_text.strip() + " " + text.strip() if ref_text else text.strip()
    wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
    text_tokens = text_tokenizer(wrapped_text).input_ids.repeat(8, 1).unsqueeze(0)  # [1, 8, N2]
    
    # 3. Target Audio（全 MASK）
    target = torch.full((1, 8, num_target_tokens), MASK_ID, dtype=torch.long)  # [1, 8, T_target]
    
    # 4. 拼接
    parts = [style_tokens, text_tokens]
    if ref_audio_tokens is not None:
        parts.append(ref_audio_tokens.unsqueeze(0))  # [1, 8, T_ref]
    parts.append(target)
    
    cond_input_ids = torch.cat(parts, dim=2)  # [1, 8, N1+N2+T_ref+T_target]
    
    # 5. Audio Mask
    audio_start = N1 + N2 + (T_ref if ref_audio_tokens else 0)
    audio_mask = torch.zeros(1, cond_input_ids.size(2), dtype=torch.bool)
    audio_mask[0, audio_start:] = True
    
    return {"input_ids": cond_input_ids, "audio_mask": audio_mask}
```

### 7.2 文本拼接的关键细节

**推理时，`ref_text` 和 `text` 会被拼接在一起！**

```python
ref_text = "今天天气真好"        # 参考音频说了什么
text = "欢迎使用语音克隆系统"   # 你想让他说什么

# 模型实际看到的文本：
"<|text_start|>今天天气真好 欢迎使用语音克隆系统<|text_end|>"
```

**为什么要这样设计？**

| 是否提供 ref_text | 模型能看到什么 | 效果 |
|------------------|-------------|------|
| **提供** | 既听到参考音频的声音，又知道参考音频说了什么 | **更好**：能理解语速、语调、发音习惯 |
| **不提供** | 只听到声音，不知道内容 | **可用但稍差**：缺少文本层面的对齐信息 |

**类比**：就像你模仿一个人说话，如果你**既听到他的声音又知道他说了什么**，模仿会更像；如果只听到声音但听不懂内容，模仿就比较模糊。

### 7.3 时长估计

**文件位置**：`omnivoice/utils/duration.py`

```python
class RuleDurationEstimator:
    """基于规则的音频时长估计器。
    
    原理：根据参考音频的"字符数/秒"来估算目标文本的时长。
    """
    def estimate_duration(self, text, ref_text, num_ref_audio_tokens):
        if not ref_text or num_ref_audio_tokens <= 0:
            return len(text) * 2.5  # fallback：每个字约 2.5 token
        
        # 参考音频的 token 速率（token/字符）
        ref_chars = len(ref_text)
        ref_tokens = num_ref_audio_tokens
        token_per_char = ref_tokens / ref_chars
        
        # 目标文本的估计 token 数
        target_tokens = len(text) * token_per_char
        
        # 加上参考音频长度（因为 prompt 会被保留）
        return target_tokens + ref_tokens
```

**使用时长控制**：
```python
# 方式 1：固定时长（覆盖 speed）
audio = model.generate(text="...", duration=10.0)  # 强制 10 秒

# 方式 2：语速控制
audio = model.generate(text="...", speed=1.5)  # 1.5x 语速（更快）
audio = model.generate(text="...", speed=0.8)  # 0.8x 语速（更慢）
```

### 7.4 后处理流程

**文件位置**：`omnivoice/models/omnivoice.py:710-785`

```python
def _decode_and_post_process(self, tokens, rms, gen_config):
    # 1. Token → 波形
    audio = audio_tokenizer.decode(tokens.unsqueeze(0)).audio_values[0]  # [1, T]
    audio = audio.cpu().numpy()
    
    # 2. 后处理
    audio = self._post_process_audio(audio, gen_config.postprocess_output, rms)
    return audio.squeeze(0)  # [T]

def _post_process_audio(self, audio, postprocess_output, ref_rms):
    if postprocess_output:
        # 2.1 去除过长静音（中间 >500ms, 首尾 >100ms）
        audio = remove_silence(audio, sample_rate=24000,
                               mid_sil=500, lead_sil=100, trail_sil=100)
    
    # 2.2 音量匹配
    if ref_rms is not None and ref_rms < 0.1:
        audio = audio * ref_rms / 0.1
    elif ref_rms is None:
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.5  # 默认归一化到 -6dB
    
    # 2.3 淡入淡出 + 边缘填充（防止爆音）
    audio = fade_and_pad_audio(audio, pad_duration=0.1, fade_duration=0.1)
    
    return audio
```

### 7.5 副语言控制（停顿、笑声、语气）

OmniVoice 支持通过**文本中的特殊标记**来控制副语言信息（paralinguistic），包括笑声、叹息、疑问语气、惊讶等。这是推理阶段独有的能力，在 `omnivoice/models/omnivoice.py:1521-1565` 中实现。

#### 7.5.1 Non-verbal Tags（非语言标记）

在输入文本中插入方括号标记，模型会将其识别为独立的副语言 token：

| 标记 | 含义 | 适用场景 |
|------|------|---------|
| `[laughter]` | 笑声 | "我今天考了满分[laughter]" |
| `[sigh]` | 叹息 | "唉，作业又写不完了[sigh]" |
| `[confirmation-en]` | 英语确认语气（嗯、对） | 英文对话中的肯定回应 |
| `[question-en]` | 英语疑问语气 | 英文疑问句末尾 |
| `[question-ah]` | "啊？"（疑问） | 中文疑问语气 |
| `[question-oh]` | "哦？"（疑问） | 恍然大悟式疑问 |
| `[question-ei]` | "诶？"（疑问） | 轻微惊讶的疑问 |
| `[question-yi]` | "咦？"（疑问） | 发现异常的疑问 |
| `[surprise-ah]` | "啊！"（惊讶） | 突发惊吓 |
| `[surprise-oh]` | "哦！"（惊讶） | 恍然大悟 |
| `[surprise-wa]` | "哇！"（惊讶） | 赞叹式惊讶 |
| `[surprise-yo]` | "哟！"（惊讶） | 轻松式惊讶 |
| `[dissatisfaction-hnn]` | "嗯..."（不满/犹豫） | 迟疑、不满、思考 |

#### 7.5.2 使用示例

```python
from omnivoice import OmniVoice
import soundfile as sf

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")

# 示例 1：笑声
text1 = "妈妈，我今天在学校得了第一名[laughter]"
audio1 = model.generate(text=text1, ref_audio="ref.wav", language="zh")
sf.write("output_laughter.wav", audio1[0], 24000)

# 示例 2：叹息
text2 = "唉，这次考试又没考好[sigh]"
audio2 = model.generate(text=text2, ref_audio="ref.wav", language="zh")
sf.write("output_sigh.wav", audio2[0], 24000)

# 示例 3：疑问语气
text3 = "你说什么[question-ah]我没听清楚"
audio3 = model.generate(text=text3, ref_audio="ref.wav", language="zh")
sf.write("output_question.wav", audio3[0], 24000)

# 示例 4：惊讶
text4 = "[surprise-wa]这个礼物太漂亮了"
audio4 = model.generate(text=text4, ref_audio="ref.wav", language="zh")
sf.write("output_surprise.wav", audio4[0], 24000)

# 示例 5：多种副语言组合
text5 = "真的吗[question-oh]太好了[laughter]"
audio5 = model.generate(text=text5, ref_audio="ref.wav", language="zh")
sf.write("output_mixed.wav", audio5[0], 24000)
```

#### 7.5.3 实现原理

**源码位置**：`omnivoice/models/omnivoice.py:1521-1565`

```python
_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """将包含副语言标记的文本进行 tokenize。
    
    关键设计：标记与普通文本**分开 tokenize**，确保标记获得一致的 
    token ID，不受前后中文/英文上下文影响。
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        # 1. 标记前的普通文本
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        # 2. 标记本身（独立 tokenize）
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    # 3. 最后一个标记后的普通文本
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)
    
    # 合并所有 token IDs
    combined = []
    for p in parts:
        combined.extend(p)
    return torch.tensor([combined], dtype=torch.long)
```

**为什么要分开 tokenize？**

BPE tokenizer 的切分是上下文相关的。`[laughter]` 前面是中文时，tokenizer 可能把它切得和前面是英文时不同。分开 tokenize 保证了 `[laughter]` 在任何语言环境下都得到**完全相同的 token ID 序列**，模型才能稳定学习到"这个 token 序列 = 笑声"。

#### 7.5.4 停顿与呼吸控制

除了显式标记，停顿主要通过**文本层面的标点**间接控制：

| 控制方式 | 机制 | 效果 |
|---------|------|------|
| **标点符号** | 逗号、句号、问号 | 模型从训练数据中学到标点→停顿的映射 |
| **空格**（英文） | `duration.py:62` 映射为 0.2 时长因子 | 词边界/呼吸 |
| **中文空格** | `_combine_text()` 自动去除中文字符周围空格 | 中文不停顿靠空格 |

**注意**：OmniVoice **没有**显式的语速/停顿长度调节参数（如 `speed=0.8` 只影响总时长，不控制单个停顿）。副语言控制主要靠：
1. 文本中的标记和标点
2. 参考音频（`ref_audio`）隐式传递的说话风格

#### 7.5.5 训练数据中的副语言

当前版本的训练数据处理流水线（`omnivoice/data/processor.py`）**没有**对这些标记做特殊处理。这意味着：

- 如果训练数据中包含 `[laughter]` 等标记的文本-音频对，模型会自然学会
- 如果训练数据中**没有**这类标记，模型可能无法稳定生成对应的副语言效果
- **预训练模型（Emilia 数据集）**：可能包含少量自然对话中的笑声、叹息，但没有系统性的标记
- **微调时**：可以在训练文本中手动加入这些标记，让模型学习特定说话人的笑声/语气风格

### 7.6 情绪控制

OmniVoice **没有内置的情绪控制机制**——代码中不存在 emotion embedding、情绪标签或情绪分类器。与某些 TTS 系统（如 Emotional VITS）不同，你不能直接传入 `emotion="happy"` 来让模型用特定情绪说话。

但情绪可以通过**三种间接方式**影响生成结果：

#### 7.6.1 Non-verbal Tags（最直接）

在文本中插入副语言标记，让模型在特定位置产生对应的声音特征：

```python
# 开心+惊讶
audio = model.generate(
    text="妈妈我考了第一名[surprise-wa][laughter]",
    ref_audio="ref.wav", language="zh"
)

# 失望+叹息
audio = model.generate(
    text="我的玩具坏了[sigh][dissatisfaction-hnn]",
    ref_audio="ref.wav", language="zh"
)
```

**限制**：
- 只有 11 种预定义标记，覆盖范围有限
- 标记只在插入位置生效，不能控制整句话的情绪基调
- 效果取决于训练数据是否见过这些标记（checkpoint-63000 没有专门训练）

#### 7.6.2 参考音频的情绪（最强烈）

参考音频的情绪会**主导**生成结果。这是目前最有效的情绪控制方式：

| 参考音频情绪 | 生成效果 | 示例 |
|-----------|---------|------|
| 开心兴奋 | 即使文本中性，语音也偏轻快 | 用孩子笑着的录音做参考 |
| 哭泣委屈 | 即使文本普通，语音也可能带哭腔 | 用孩子哭的录音做参考 |
| 平静自然 | 标准语音，无特殊情绪 | 日常对话录音 |

```python
# 用哭泣的参考音频 → 生成带哭腔的语音
audio = model.generate(
    text="我要妈妈",              # 文本本身中性
    ref_audio="crying_child.wav",  # 哭泣的参考音频
    language="zh"
)
# 结果：声音可能带有哭泣的特征（哽咽、颤抖）
```

**关键机制**：参考音频的声学特征（基频变化、能量分布、频谱包络）通过 prompt 传入模型，双向注意力使这些特征影响整个生成过程。

**限制**：
- 需要找到对应情绪的参考音频
- 情绪效果随 guidance_scale 增强（scale 越高，越像参考音频）
- 无法在同一句话中切换情绪（除非用多个参考音频分段生成）

#### 7.6.3 文本内容的情绪暗示（最弱）

模型可能从文本词汇中隐式推断情绪：

```python
# 文本暗示开心 → 可能生成更轻快的语音
audio = model.generate(text="耶！太好了！我赢了！", ref_audio="ref.wav")

# 文本暗示难过 → 可能生成更低沉的语音
audio = model.generate(text="唉...我好难过，不想说话", ref_audio="ref.wav")
```

**但这是隐性学习**：
- 预训练时模型从海量文本-音频对中学到"开心词汇 ↔ 高基频音频"的关联
- 没有专门的情绪监督信号
- 效果**非常不稳定**，受参考音频影响很大

#### 7.6.4 三种方式的对比

| 控制方式 | 效果强度 | 可控性 | 实现难度 | 适用场景 |
|---------|---------|--------|---------|---------|
| **Non-verbal tags** | 中等 | 高（精确到词） | 简单（改文本） | 特定位置的笑声、叹息 |
| **参考音频情绪** | 强 | 低（取决于能找到什么参考） | 中等（需要找对应情绪的录音） | 整句话的统一情绪 |
| **文本暗示** | 弱 | 中 | 简单（改文本） | 辅助增强 |

#### 7.6.5 组合使用示例

实际使用中，**三种方式组合**效果最佳：

```python
# 目标：生成"开心且惊讶"的语音

# Step 1: 选择开心的参考音频
prompt = model.create_voice_clone_prompt(
    ref_audio="excited_child.wav",  # 孩子兴奋的录音
    ref_text="太好了！",
)

# Step 2: 文本中包含情绪词汇 + non-verbal tags
audio = model.generate(
    text="哇[surprise-wa]我考了100分[laughter]太厉害了！",
    voice_clone_prompt=prompt,
    language="zh",
    generation_config=OmniVoiceGenerationConfig(
        guidance_scale=2.5,  # 增强参考音频影响
    )
)
```

#### 7.6.6 如何实现真正的情绪控制

如果需要在 OmniVoice 中实现系统性的情绪控制，需要修改训练流程：

**方案 1：训练数据标注**
1. 在训练 JSONL 中增加 `emotion` 字段：
   ```json
   {"id": "001", "text": "你好", "audio_path": "...", "emotion": "happy"}
   ```
2. 修改 `processor.py`，将 emotion 加入 style token：
   ```python
   style += f"<|emotion_start|>{emotion}<|emotion_end|>"
   ```
3. 微调模型学习 emotion token 与声学特征的映射

**方案 2：Latent 空间插值**
1. 收集同一说话人不同情绪的参考音频
2. 提取各自的 voice_clone_prompt
3. 在 latent 空间做插值：
   ```python
   happy_prompt = model.create_voice_clone_prompt("happy.wav")
   sad_prompt = model.create_voice_clone_prompt("sad.wav")
   # 插值得到中间情绪
   mixed_prompt = VoiceClonePrompt(
       ref_audio_tokens=0.7 * happy_prompt.ref_audio_tokens + 0.3 * sad_prompt.ref_audio_tokens,
       ref_text=happy_prompt.ref_text,
       ref_rms=happy_prompt.ref_rms,
   )
   ```

**注意**：方案 2 是理论上的，实际效果取决于模型对 prompt 的线性插值是否平滑。

---

## 8. 数据流水线：逐模块解析

### 8.1 WebDataset 格式

**manifest 文件**（`data.lst`）：
```
/path/to/shard-000000.tar /path/to/shard-000000.jsonl 1000 3600.5
/path/to/shard-000001.tar /path/to/shard-000001.jsonl 800 2880.2
```
每行格式：`<tar路径> <jsonl路径> <样本数> <总秒数>`

**tar shard 内部结构**：
```
shard-000000.tar
├── sample_001.npy       # audio tokens [8, T]
├── sample_002.npy
└── ...

shard-000000.jsonl
{"id": "sample_001", "text": "你好", "language_id": "zh"}
{"id": "sample_002", "text": "Hello", "language_id": "en"}
```

### 8.2 SampleDecoder

**文件位置**：`omnivoice/data/dataset.py:189-249`

```python
class SampleDecoder:
    def __call__(self, sample):
        # sample 来自 WebDataset
        src = sample["__url__"]   # tar 文件路径
        key = sample["__key__"]   # 样本 key
        
        # 1. 加载 audio tokens
        if "npy" in sample:
            audio_tokens = torch.from_numpy(sample["npy"])  # [8, T]
            return {"audio_tokens": audio_tokens, "label": label}
        
        # 2. 或者加载原始音频（预处理阶段）
        for ext in ("flac", "wav", "mp3"):
            if ext in sample:
                audio = load_audio_webdataset(sample[ext], 24000)
                return {"audio": audio, "label": label, "audio_duration": dur}
```

### 8.3 两种 Collator 的详细对比

**PaddingDataCollator**（sdpa 模式）：
```python
def __call__(self, processed_samples):
    # processed_samples: List[Dict]，每个 Dict 来自 Processor
    
    max_len = max(s["length"] for s in processed_samples)
    B = len(processed_samples)
    
    # 对每个样本 padding 到 max_len
    for i, s in enumerate(processed_samples):
        pad = max_len - s["length"]
        input_ids[i] = F.pad(s["input_ids"], (0, pad), value=pad_token_id)
        labels[i] = F.pad(s["labels"], (0, pad), value=-100)
        audio_mask[i] = F.pad(s["audio_mask"], (0, pad), value=False)
    
    # 4D 双向注意力 mask
    # attention_mask[b, 0, i, j] = True 表示位置 i 可以看到位置 j
    attention_mask = valid[:, None, None, :].expand(B, 1, max_len, max_len)
    
    return {
        "input_ids": torch.stack(input_ids),      # [B, 8, max_len]
        "labels": torch.stack(labels),             # [B, 8, max_len]
        "audio_mask": torch.stack(audio_mask),     # [B, max_len]
        "attention_mask": attention_mask,          # [B, 1, max_len, max_len]
        "position_ids": torch.stack(position_ids), # [B, max_len]
    }
```

**PackingDataCollator**（flex_attention 模式）：
```python
def __call__(self, processed_samples):
    # 直接拼接，不 padding
    input_ids = torch.cat([s["input_ids"] for s in processed_samples], dim=1)   # [8, Total]
    labels = torch.cat([s["labels"] for s in processed_samples], dim=1)         # [8, Total]
    audio_mask = torch.cat([s["audio_mask"] for s in processed_samples], dim=0) # [Total]
    
    # document_ids 标记每个样本的归属
    document_ids = []
    for i, s in enumerate(processed_samples):
        document_ids.append(torch.full((s["length"],), i, dtype=torch.int32))
    document_ids = torch.cat(document_ids)  # [Total]
    
    # padding 到 batch_tokens（通常是 8192 或 16384）
    pad = target_length - input_ids.shape[1]
    input_ids = F.pad(input_ids, (0, pad), value=pad_token_id)
    ...
    
    return {
        "input_ids": input_ids.unsqueeze(0),       # [1, 8, L]
        "labels": labels.unsqueeze(0),              # [1, 8, L]
        "audio_mask": audio_mask.unsqueeze(0),      # [1, L]
        "document_ids": document_ids.unsqueeze(0),  # [1, L]
    }
```

---

## 9. 微调实战：儿童语音案例

### 9.1 项目背景

当前仓库正在进行**儿童语音微调**实验：
- **目标**：让模型更好地克隆儿童的声音
- **数据**：4 个儿童语音数据集（中英混合）
- **模型**：基于 `k2-fsa/OmniVoice` 预训练权重微调

### 9.2 数据集详情

| 数据集 | 语言 | 说明 |
|--------|------|------|
| BAAI-ChildMandarin41.25H | zh | 中文儿童语音 |
| Chinese_English_Scripted_Speech_Children | en/zh | 中英脚本语音 |
| King-ASR-EN-Kid | en | 英文儿童 |
| speechocean762 | en | 英文儿童 |

### 9.3 微调配置（checkpoint-63000）

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
    
    // 关闭不需要的功能（儿童数据没有 instruct 和 pinyin）
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,
    
    "init_from_checkpoint": "/root/.cache/modelscope/k2-fsa/OmniVoice",
    "learning_rate": 3e-6,          // 小学习率保护预训练知识
    "steps": 100000,
    "batch_tokens": 16384,          // 2x A800 80GB
    
    "mixed_precision": "bf16",
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000
}
```

### 9.4 关键决策：为什么 lr=3e-6？

| 学习率 | 适用场景 | 风险 |
|--------|---------|------|
| 1e-4 | 从零训练/预训练 | 微调会覆盖预训练知识 |
| 1e-5 | 大领域微调（10+ 小时） | 可能稍大 |
| **3e-6** | **小领域微调（保护知识）** | **推荐** |
| 5e-6 | 中等微调 | 适中 |

儿童语音微调使用 **3e-6** 是因为：
1. 数据量不大（~40 小时），需要避免过拟合
2. 预训练模型已经能 clone 儿童声音，微调只是让它**更稳定**
3. 小学习率保护预训练的通用能力

### 9.5 批量推理对比脚本

```python
# exp/children_finetune_20260519_1418/run_batch_inference.py
CKPT_DIR = Path("exp/children_finetune_20260519_1418/checkpoints")

# 遍历所有 checkpoint
for ckpt in sorted(checkpoints):
    model = OmniVoice.from_pretrained(str(ckpt), device_map="cuda:0")
    
    for sample in samples:
        audio = model.generate(
            text=sample["clone_text"],
            ref_audio=sample["ref_audio"],
            ref_text=sample["ref_text"],
            language="zh",
        )
        sf.write(out_path, audio[0], 24000)
    
    del model; gc.collect(); torch.cuda.empty_cache()
```

---

## 10. 评估体系完整指南

### 10.1 WER（词错误率）

**流程**：
1. 生成音频
2. ASR 转写（Whisper / SenseVoice / HuBERT）
3. 文本归一化（小写、去标点、数字转文字）
4. 计算编辑距离

**文件位置**：`omnivoice/eval/wer/`

```python
# 多语言文本归一化
text_normalize(text, iso_code="zh", lower_case=True, remove_numbers=True)

# 中文归一化：
# - 全角转半角
# - 数字转中文（123 → 一百二十三）
# - 去除标点
# - 统一繁简
```

### 10.2 说话人相似度（Speaker Similarity）

**流程**：
1. 用 ECAPA-TDNN + WavLM 提取 speaker embedding
2. 计算克隆音频与参考音频的余弦相似度

```python
# omnivoice/eval/speaker_similarity/sim.py
ref_embedding = extract_speaker_embedding(ref_audio)
clone_embedding = extract_speaker_embedding(cloned_audio)
similarity = cosine_similarity(ref_embedding, clone_embedding)
```

**评分标准**：
| 相似度 | 评价 |
|--------|------|
| > 0.85 | 非常像 |
| 0.75~0.85 | 比较像 |
| 0.65~0.75 | 有点像 |
| < 0.65 | 不太像 |

### 10.3 自然度（UTMOS）

**UTMOS**：基于深度学习的 MOS 预测模型

```python
# omnivoice/eval/mos/utmos.py
score = predict_utmos(audio)  # 范围 1.0 ~ 5.0
```

**评分标准**：
| UTMOS | 主观感受 |
|-------|---------|
| > 4.0 | 非常自然，像真人 |
| 3.5~4.0 | 自然，偶尔有瑕疵 |
| 3.0~3.5 | 可接受，有明显合成感 |
| < 3.0 | 不自然 |

---

## 11. CLI 工具源码解析

### 11.1 omnivoice-infer（单条推理）

```bash
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "Hello world" \
    --ref_audio ref.wav \
    --ref_text "Reference transcript" \
    --output out.wav \
    --num_step 32 \
    --guidance_scale 2.0
```

**源码**：`omnivoice/cli/infer.py`

### 11.2 omnivoice-infer-batch（批量推理）

**输入 JSONL**：
```jsonl
{"id": "001", "text": "Hello", "ref_audio": "/path/ref.wav", "ref_text": "Hello", "language_id": "en"}
{"id": "002", "text": "你好", "instruct": "female, low pitch", "language_id": "zh"}
```

**支持多 GPU 分布式推理**：
```bash
accelerate launch \
    --num_processes 4 \
    -m omnivoice.cli.infer_batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

### 11.3 omnivoice-demo（Web UI）

基于 Gradio 的交互式界面：
- Voice Clone 标签页
- Voice Design 标签页
- 实时试听

---

## 12. 性能优化与内存分析

### 12.1 GPU 显存占用

| 配置 | 显存占用 | 说明 |
|------|---------|------|
| 推理（单条, fp16） | ~4 GB | batch_size=1 |
| 推理（单条, fp32） | ~6 GB | 更高精度 |
| 训练（batch_tokens=8192, bf16） | ~40 GB | 单卡 |
| 训练（batch_tokens=16384, bf16） | ~70 GB | 单卡 |

### 12.2 推理速度优化

| 优化手段 | 效果 | 方法 |
|---------|------|------|
| 使用 fp16 | 速度 +50%，显存 -50% | `dtype=torch.float16` |
| 减少 num_step | 速度线性提升 | 16 步（快）vs 50 步（慢） |
| 复用 voice_clone_prompt | 避免重复编码参考音频 | `create_voice_clone_prompt()` |
| 批量推理 | GPU 利用率更高 | 一次处理多条 |

### 12.3 训练速度优化

| 优化手段 | 效果 | 方法 |
|---------|------|------|
| flex_attention | 速度 +20% | 需要 Ampere+ GPU |
| tf32 | 速度 +10% | `allow_tf32=True` |
| 增大 batch_tokens | 效率更高 | 受显存限制 |
| DeepSpeed ZeRO-2 | 支持更大 batch | 多卡场景 |

---

## 13. 完整配置文件示例

### 13.1 训练配置（train_config.json）

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
    "use_pinyin_ratio": 0.3,
    "instruct_ratio": 1.0,
    "only_instruct_ratio": 0.5,
    
    "init_from_checkpoint": "k2-fsa/OmniVoice",
    "resume_from_checkpoint": null,
    
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 5000,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.01,
    
    "batch_tokens": 8192,
    "gradient_accumulation_steps": 1,
    "num_workers": 4,
    
    "mixed_precision": "bf16",
    "allow_tf32": true,
    "attn_implementation": "flex_attention",
    
    "logging_steps": 50,
    "eval_steps": 500,
    "save_steps": 1000,
    "keep_last_n_checkpoints": -1
}
```

### 13.2 数据配置（data_config.json）

```json
{
    "train": [
        {
            "language_id": "zh",
            "manifest_path": ["/data/zh/train/data.lst"],
            "repeat": 1
        },
        {
            "language_id": "en",
            "manifest_path": ["/data/en/train/data.lst"],
            "repeat": 1
        }
    ],
    "dev": [
        {
            "language_id": "zh",
            "manifest_path": ["/data/zh/dev/data.lst"],
            "repeat": 1
        }
    ]
}
```

### 13.3 DeepSpeed 配置（ds_config_zero2.json）

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
```

---

## 14. 调试技巧与问题排查

### 14.1 训练调试

**检查 1：数据是否正确加载**
```python
# 在 trainer.py 中打断点
batch = next(iter(train_dataloader))
print(batch["input_ids"].shape)    # 应该是 [B, 8, L] 或 [1, 8, L]
print(batch["labels"].shape)       # 同上
print((batch["labels"] != -100).sum())  # 应该有 mask 的位置（非零）
```

**检查 2：Loss 是否合理**
```python
# 初始 loss 应该在 2.0~4.0 之间（随机猜 1025 类的交叉熵 ≈ ln(1025) ≈ 6.9，但有权重和 mask）
# 如果初始 loss < 1.0：可能权重加载错误
# 如果初始 loss > 10：可能是 label 构造错误
```

**检查 3：梯度是否正常**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 14.2 推理调试

**检查 1：参考音频长度**
```python
prompt = model.create_voice_clone_prompt(ref_audio="ref.wav")
print(prompt.ref_audio_tokens.shape)  # [8, T], T 应该在 225~750（3~10秒）
```

**检查 2：生成序列长度**
```python
task = model._preprocess_all(text="测试文本", ref_audio="ref.wav")
print(task.target_lens)  # 目标 token 数，对应音频时长
```

**检查 3：逐步 reveal**
```python
# 修改 _generate_iterative 打印每步 reveal 数量
for step in range(num_step):
    print(f"Step {step}: reveal {schedules[0][step]} tokens, "
          f"remaining={mask_count}")
```

### 14.3 常见问题速查

| 现象 | 诊断命令 | 解决方案 |
|------|---------|---------|
| CUDA OOM | `nvidia-smi` | 减小 `batch_tokens` 或改用 fp16 |
| Loss = nan | `torch.isnan(loss)` | 降低学习率，检查输入是否有 inf |
| 生成静音 | `sf.read("out.wav")` | 检查 ref_audio 是否有效，增大 guidance_scale |
| 克隆不像 | 听感对比 | 检查 ref_text 是否准确，使用 5~10s 参考音频 |
| 速度慢 | `time python infer.py` | 使用 fp16，减少 num_step，复用 prompt |
| 训练不收敛 | `tensorboard --logdir exp/` | 检查学习率、数据质量、是否加载了预训练权重 |

---

## 15. 与同类模型的深度对比

### 15.1 架构对比

| 维度 | OmniVoice | VALL-E | VALL-E X | Bark | XTTS v2 |
|------|-----------|--------|----------|------|---------|
| **基础架构** | 改造 LLM (Qwen3) | 自回归 Encoder-Decoder | 自回归 Encoder-Decoder | 自回归 Transformer | VAE + GPT |
| **注意力** | **双向** | 因果 | 因果 | 因果 | 因果 |
| **生成方式** | **离散扩散 (32步)** | 自回归 (~750步/10s) | 自回归 | 自回归 | 自回归 |
| **音频编码** | RVQ 8层 | RVQ 8层 | RVQ 8层 | EnCodec | Hifi-GAN |
| **Speaker 信息** | **隐式 (prompt)** | 显式 ( enrolled ) | 显式 ( enrolled ) | 隐式 (prompt) | 显式 (d-vector) |
| **语言覆盖** | **600+** | 英语为主 | 多语言 | 多语言 | 13 语言 |
| **推理 RTF** | **0.025** | ~1.0 | ~1.0 | ~2.0 | ~0.5 |
| **Voice Design** | **支持** | 不支持 | 不支持 | 有限 | 不支持 |
| **开源** | **完全开源** | 未开源 | 未开源 | 开源 | 开源 |
| **模型大小** | 0.6B | 1B | 1B | ~1.5B | 400M |

### 15.2 为什么 OmniVoice 更快？

**VALL-E（自回归）**：
```
生成 10 秒音频 = 750 个 token
需要 750 次前向传播（逐个生成）
每次处理长度递增：1, 2, 3, ..., 750
总计算量 ∝ 750²
```

**OmniVoice（离散扩散）**：
```
生成 10 秒音频 = 750 个 token × 8 层 = 6000 个 mask
只需 32 次前向传播
每次处理固定长度（~1000 token）
总计算量 ∝ 32 × 1000
```

**速度比**：
```
VALL-E 计算量 / OmniVoice 计算量 ≈ 750² / (32 × 1000) ≈ 17.6x
```

实际 RTF 差距更大（40x），因为：
1. OmniVoice 使用双向注意力，可以并行计算
2. OmniVoice 的序列长度固定，缓存效率更高
3. 32 步可以使用更大的 batch_size

### 15.3 为什么不需要 speaker_id？

| 模型 | Speaker 信息方式 | 缺点 |
|------|----------------|------|
| VALL-E | 3 秒 enrolled audio → 编码为 speaker embedding | 需要预处理，embedding 可能丢失细节 |
| XTTS | d-vector / speaker encoder | 需要额外的 speaker 编码器模块 |
| **OmniVoice** | **直接使用原始音频 token 作为 prompt** | **无信息损失，端到端** |

OmniVoice 的 prompt 是原始音频 token，包含了完整的声学信息，因此不需要额外的 speaker 编码器。

---

## 附录：核心公式汇总

### A. RVQ 重建
```
v ≈ Σᵢ codebook[i][token[i]]   (i = 0~7)
```

### B. Embedding 融合
```
embed[b, s] = audio_mask[b, s] ? Σₗ audio_embed[input[b, l, s] + offset[l]]
                                 : text_embed[input[b, 0, s]]
```

### C. 损失函数
```
L = Σₗ wₗ · mean(CE(logits[:, l, :, :], labels[:, l, :]) · valid_mask)
  = Σₗ wₗ · Lₗ

w = [8, 8, 6, 6, 4, 4, 2, 2] / 40
```

### D. CFG（推理）
```
log_probs = log_softmax(log P_cond + s · (log P_cond - log P_uncond))

其中 s = guidance_scale
```

### E. 时间步调度
```
tᵢ = shift · (i/N) / (1 + (shift - 1) · (i/N))

revealᵢ = total_mask · (tᵢ₊₁ - tᵢ)
```

### F. Layer Penalty
```
score[b, l, s] = confidence[b, l, s] - l · penalty

其中 l ∈ {0, 1, ..., 7}, penalty = 5.0
```

---

## 16. OmniVoiceGenerationConfig 完整参数详解

**源码位置**：`omnivoice/models/omnivoice.py:97-115`

```python
@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32              # 迭代去掩码步数
    guidance_scale: float = 2.0     # CFG 引导强度
    t_shift: float = 0.1            # 时间步非线性调度参数
    layer_penalty_factor: float = 5.0   # 层级惩罚系数
    position_temperature: float = 5.0   # 位置选择温度
    class_temperature: float = 0.0      # token 采样温度（0=greedy）
    denoise: bool = True            # 是否添加去噪标记
    preprocess_prompt: bool = True  # 是否预处理参考音频
    postprocess_output: bool = True # 是否后处理输出音频
    audio_chunk_duration: float = 15.0  # 长音频分块时长（秒）
    audio_chunk_threshold: float = 30.0 # 长音频分块阈值（秒）
```

### 16.1 参数逐项说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_step` | int | 32 | 迭代去掩码的总步数。步数越多，生成质量越高但速度越慢。常用值：16（快）、32（标准）、64（高质量） |
| `guidance_scale` | float | 2.0 | CFG 引导强度。0 = 无条件生成（更快但质量低）；1~3 = 标准语音克隆；>3 = 更强的风格跟随但可能过拟合到参考音频 |
| `t_shift` | float | 0.1 | 时间步非线性调度参数。控制早期步 reveal 多少 token。值越小，早期 reveal 越少，后期 reveal 越集中。通常不需要调整 |
| `layer_penalty_factor` | float | 5.0 | 层级惩罚系数。每层减去 `layer_id × factor` 的分数，迫使低层（高频）先被 reveal。值越大，层级顺序越严格 |
| `position_temperature` | float | 5.0 | 位置选择温度。Gumbel 采样温度，控制选择哪些位置去掩码。>0 时引入随机性；=0 时按置信度严格排序 |
| `class_temperature` | float | 0.0 | Token 采样温度。控制 predict token 时的随机性。0 = greedy（确定性）；>0 时先 filter top-10% 再 Gumbel 采样 |
| `denoise` | bool | True | 是否在 style token 前添加 `<\|denoise\|>`。用于去除参考音频中的噪声，提升克隆质量 |
| `preprocess_prompt` | bool | True | 是否对参考音频做预处理（去静音、截断、自动加标点）。建议保持 True |
| `postprocess_output` | bool | True | 是否对生成音频做后处理（去静音、音量归一化、淡入淡出）。建议保持 True |
| `audio_chunk_duration` | float | 15.0 | 长音频分块生成的每块时长。超过 threshold 的文本会被切分为多段生成 |
| `audio_chunk_threshold` | float | 30.0 | 长音频分块阈值。估计 token 数超过 `30s × frame_rate` 时启用分块生成 |

### 16.2 常用配置组合

```python
from omnivoice import OmniVoiceGenerationConfig

# 快速预览（牺牲质量换速度）
fast = OmniVoiceGenerationConfig(num_step=16, guidance_scale=1.0)

# 标准质量（平衡速度和质量）
standard = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)

# 高质量（慢但更自然）
high_quality = OmniVoiceGenerationConfig(
    num_step=64, guidance_scale=2.5, class_temperature=0.5
)

# 确定性生成（可复现结果）
deterministic = OmniVoiceGenerationConfig(
    num_step=32, guidance_scale=2.0,
    position_temperature=0.0, class_temperature=0.0
)

# 强风格跟随（更像参考音频）
strong_clone = OmniVoiceGenerationConfig(
    num_step=32, guidance_scale=3.0, denoise=True
)
```

### 16.3 温度参数详解

**`class_temperature`（token 采样温度）**：

```python
# class_temperature = 0.0（默认，greedy）
# 直接取概率最大的 token，结果确定性
pred_tokens = log_probs.argmax(dim=-1)

# class_temperature > 0.0
# 1. 先 filter top-10%（约 102 个候选）
filtered = _filter_top_k(log_probs, ratio=0.1)  # 只保留 top 10%
# 2. Gumbel 采样引入随机性
noisy = filtered / temperature + gumbel_noise
pred_tokens = noisy.argmax(dim=-1)
```

**`position_temperature`（位置选择温度）**：

```python
# position_temperature = 0.0
# 严格按置信度从高到低选择位置
selected = scores.argsort(descending=True)[:k]

# position_temperature > 0.0
# 引入 Gumbel 噪声，低置信度位置也有机会被选中
noisy_scores = scores / temperature + gumbel_noise
selected = noisy_scores.argsort(descending=True)[:k]
```

温度参数的作用：
- **温度 = 0**：完全确定性，每次生成结果相同
- **温度 > 0**：引入随机性，同一文本多次生成会有细微差异
- **温度越高**：随机性越强，多样性越高，但可能牺牲一致性

---

## 17. Voice Clone 语音克隆机制

语音克隆（Voice Clone）是 OmniVoice 的核心功能之一。它允许模型仅通过**3-10 秒的参考音频**，就合成出与该说话人音色、语调、语速高度一致的语音。

### 17.1 两种使用方式

**方式一：直接传入 ref_audio + ref_text（简单）**

```python
audio = model.generate(
    text="欢迎使用语音克隆系统",
    ref_audio="speaker_ref.wav",      # 参考音频路径
    ref_text="这是参考音频的文本",     # 参考音频的转录（可选，None 时自动识别）
    language="zh",
)
```

**方式二：先创建可复用的 VoiceClonePrompt（推荐，批量推理时更高效）**

```python
# 1. 一次性创建 prompt（包含编码后的音频 token、参考文本、音量信息）
prompt = model.create_voice_clone_prompt(
    ref_audio="speaker_ref.wav",
    ref_text="这是参考音频的文本",     # None 则自动 ASR 识别
    preprocess_prompt=True,
)

# 2. 多次复用（避免重复编码参考音频）
audio1 = model.generate(text="第一句", voice_clone_prompt=prompt, language="zh")
audio2 = model.generate(text="第二句", voice_clone_prompt=prompt, language="zh")
audio3 = model.generate(text="第三句", voice_clone_prompt=prompt, language="zh")
```

### 17.2 create_voice_clone_prompt 内部流程

**源码位置**：`omnivoice/models/omnivoice.py:603-708`

```python
def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
    # Step 1: 加载音频
    ref_wav = load_audio(ref_audio, sample_rate=24000)  # [1, T]
    
    # Step 2: 音量检测与归一化（防止过 quiet 的音频）
    ref_rms = sqrt(mean(ref_wav^2))
    if 0 < ref_rms < 0.1:
        ref_wav = ref_wav * 0.1 / ref_rms  # 提升到 -20dB
    
    # Step 3: 预处理（可选）
    if preprocess_prompt:
        # 3a: 超过 20s 的音频在最大静音处截断
        if ref_text is None:  # 只有无文本时才截断（避免文本不匹配）
            ref_wav = trim_long_audio(ref_wav, trim_threshold=20.0)
        # 3b: 去除静音（中间>200ms, 首尾>100ms）
        ref_wav = remove_silence(ref_wav, mid_sil=200, lead_sil=100, trail_sil=200)
    
    # Step 4: 警告（参考音频过长会降低克隆质量）
    if duration > 20.0:
        logger.warning("参考音频 %.1fs > 20s，建议截断到 3-10s")
    
    # Step 5: 自动转录（如果 ref_text 未提供）
    if ref_text is None:
        ref_text = self.transcribe(ref_wav)  # 使用内置 ASR
    
    # Step 6: 对齐到 tokenizer 的 hop_length 倍数
    chunk_size = audio_tokenizer.config.hop_length  # 通常 320
    ref_wav = ref_wav[:, :-(ref_wav.shape[-1] % chunk_size)]
    
    # Step 7: 编码为离散音频 token
    ref_audio_tokens = audio_tokenizer.encode(ref_wav).audio_codes  # [8, T]
    
    # Step 8: 自动加标点（如果文本没有标点结尾）
    if preprocess_prompt:
        ref_text = add_punctuation(ref_text)
    
    return VoiceClonePrompt(
        ref_audio_tokens=ref_audio_tokens,  # [8, T]
        ref_text=ref_text,
        ref_rms=ref_rms,
    )
```

### 17.3 VoiceClonePrompt 的数据结构

```python
@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor   # [8, T]，参考音频的 RVQ token
    ref_text: str                    # 参考音频的转录文本
    ref_rms: float                   # 参考音频的 RMS 音量（用于输出音量匹配）
```

### 17.4 语音克隆的推理输入序列

当使用语音克隆时，推理输入序列的构造方式：

```
[Style]              <|denoise|><|lang_start|>zh<|lang_end|><|instruct_start|>None<|instruct_end|>
[Text]               <|text_start|>{ref_text} {target_text}<|text_end|>
[Reference Audio]    [ref_token_1] [ref_token_2] ... [ref_token_T]     ← 参考音频作为 prompt
[Target Audio]       [MASK] [MASK] ... [MASK]                            ← 需要生成的目标
```

**关键设计**：
- 参考音频的 token 被放在目标音频之前，作为**声学 prompt**
- 模型通过双向注意力同时"看到"参考音频的声学内容和 ref_text 的文本内容
- 这种设计不需要显式的 speaker embedding，说话人信息完全由 prompt 音频隐式编码

### 17.5 最佳实践

| 场景 | 建议 |
|------|------|
| **参考音频长度** | 3-10 秒最优。太短（<2s）信息不足；太长（>20s）质量下降 |
| **参考音频质量** | 干净、无背景音乐、单人说话。嘈杂环境会降低克隆质量 |
| **ref_text 是否必须** | 不必须，但提供后克隆效果更好（模型知道参考音频说了什么） |
| **跨语言克隆** | 支持！可以用中文参考音频克隆说英文，但口音会带中文特征 |
| **批量克隆** | 使用 `create_voice_clone_prompt()` 缓存，避免重复编码参考音频 |
| **preprocess_prompt** | 保持 True（默认）。会自动去静音、截断、加标点 |

### 17.6 语音克隆 vs Voice Design

| 特性 | Voice Clone | Voice Design |
|------|------------|--------------|
| 输入 | 参考音频 + 文本 | 文本描述 + 目标文本 |
| 控制粒度 | 精确复制某个说话人 | 通过描述控制风格 |
| 示例 | "用张三的声音说这句话" | "用一个低沉有力的男声说这句话" |
| 实现 | ref_audio_tokens 作为 prompt | instruct 作为 style token |

---

## 18. 多语言与指令系统

### 18.1 多语言支持

OmniVoice 支持 **100+ 种语言**，语言标识使用 ISO 639-3 代码（3 字母）或完整语言名称。

**常用语言代码**：

| 语言 | 代码 | 完整名称 |
|------|------|---------|
| 中文 | `zh` | `Chinese` |
| 英语 | `en` | `English` |
| 日语 | `ja` | `Japanese` |
| 韩语 | `ko` | `Korean` |
| 德语 | `de` | `German` |
| 法语 | `fr` | `French` |
| 西班牙语 | `es` | `Spanish` |
| 俄语 | `ru` | `Russian` |
| 阿拉伯语 | `ar` | `Arabic` |
| 印地语 | `hi` | `Hindi` |

**查看所有支持的语言**：

```python
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
print(model.supported_language_ids())    # {'zh', 'en', 'ja', 'de', ...}
print(model.supported_language_names())  # {'Chinese', 'English', 'Japanese', ...}
```

**语言解析逻辑**：`omnivoice/models/omnivoice.py:1342-1359`

```python
def _resolve_language(language):
    if language is None or language.lower() == "none":
        return None  # 语言无关模式
    if language in LANG_IDS:
        return language  # 直接是 ISO 639-3 代码
    key = language.lower()
    if key in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[key]  # 从名称映射到代码
    logger.warning(f"语言 '{language}' 不被识别，回退到 None")
    return None
```

**使用方式**：

```python
# 方式 1：代码
audio = model.generate(text="Hello", language="en")

# 方式 2：完整名称
audio = model.generate(text="Bonjour", language="French")

# 方式 3：None（语言无关模式，性能稍差）
audio = model.generate(text="Hello", language=None)
```

**注意**：
- `language=None` 时模型不接收语言信息，仍能生成但性能稍差
- 训练时的 `language_ratio` 控制是否随机丢弃语言信息（用于增强泛化性）
- 语言信息通过 `<|lang_start|>{language}<|lang_end|>` 作为 style token 传入

### 18.2 指令系统（Voice Design）

除了语音克隆，OmniVoice 还支持**通过文本描述来控制说话风格**，无需参考音频。

**支持的指令维度**：`omnivoice/models/omnivoice.py:1362-1380`

| 维度 | 英文选项 | 中文选项 |
|------|---------|---------|
| 性别 | `male`, `female` | `男`, `女` |
| 年龄 | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` | `儿童`, `少年`, `青年`, `中年`, `老年` |
| 音高 | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` | - |
| 风格 | `whisper`（耳语） | - |
| 口音 | `american accent`, `british accent`, `australian accent`, ... | - |

**使用示例**：

```python
# 英文指令（逗号+空格分隔）
audio = model.generate(
    text="Welcome to the voice design system.",
    instruct="female, young adult, high pitch, british accent",
    language="en",
)

# 中文指令（全角逗号分隔）
audio = model.generate(
    text="欢迎使用语音设计系统。",
    instruct="女，青年，高音",
    language="zh",
)

# 仅使用指令（不提供 ref_audio）
audio = model.generate(
    text="这是一段由指令控制的语音。",
    instruct="男，中年，低音",
    language="zh",
)
```

**训练时的指令比例**：

```json
{
    "instruct_ratio": 0.0,        // 使用指令的概率
    "only_instruct_ratio": 0.0    // 仅用指令（不提供参考音频）的概率
}
```

- `instruct_ratio=0.5`：50% 的样本会加入 instruct 信息
- `only_instruct_ratio=0.2`：在使用 instruct 的样本中，20% 会完全丢弃参考音频，仅靠指令生成

**注意**：预训练模型的 `instruct_ratio` 和 `only_instruct_ratio` 通常设为 0（Emilia 数据集没有指令标注），需要微调才能启用 Voice Design 功能。

---

## 19. 音频 Tokenizer 架构详解

OmniVoice 使用 **Higgs Audio V2 Tokenizer** 作为音频编解码器。这是一个**独立的神经网络**，与 LLM 分开训练、分开加载。

### 19.1 架构位置

```
OmniVoice 系统 = LLM (Qwen3-0.6B 改造) + Audio Tokenizer (Higgs Audio V2)
                    ↑                          ↑
              生成音频 token               编码/解码波形
```

**文件结构**：

```
/root/.cache/modelscope/k2-fsa/OmniVoice/audio_tokenizer/
├── config.json              # Tokenizer 配置
├── model.safetensors        # Tokenizer 权重 (~800MB)
└── preprocessor_config.json # 音频预处理配置
```

### 19.2 核心参数

```json
{
    "model_type": "higgs_audio_v2",
    "sample_rate": 24000,
    "hop_length": 320,
    "num_quantizers": 8,
    "codebook_size": 1024,
    "codebook_dim": 256,
    "encoder_dim": 1024,
    "decoder_dim": 1024
}
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `sample_rate` | 24000 Hz | 音频采样率 |
| `hop_length` | 320 | 每帧对应的采样点数。24000Hz / 320 = 75 fps，即每帧 13.33ms |
| `num_quantizers` | 8 | RVQ 层数 |
| `codebook_size` | 1024 | 每层码本大小 |
| `codebook_dim` | 256 | 码本向量维度 |

### 19.3 编码流程（波形 → Token）

```python
# 输入：波形 [1, T]，T 为采样点数
audio_tokens = audio_tokenizer.encode(waveform)
# 输出：audio_codes [1, 8, T']，T' = T // 320

# 示例：10 秒音频
T = 24000 * 10 = 240000  # 采样点数
T' = 240000 // 320 = 750  # token 数（约 75 token/秒）
```

### 19.4 解码流程（Token → 波形）

```python
# 输入：audio_codes [1, 8, T']
audio = audio_tokenizer.decode(audio_codes)
# 输出：audio_values [1, T]，T = T' × 320
```

### 19.5 与 LLM 的关系

| 组件 | 功能 | 训练状态 | 显存占用 |
|------|------|---------|---------|
| **Audio Tokenizer** | 波形 ↔ 离散 token 的转换 | 预训练，**冻结** | ~2GB (encode+decode) |
| **LLM (Qwen3-0.6B)** | Token → Token 的预测 | 可训练 | ~4GB (fp16) |

**关键点**：
- Audio Tokenizer 在训练时**完全冻结**，不更新权重
- LLM 只学习"预测下一个音频 token"，不负责波形生成
- 推理时先由 LLM 生成音频 token，再由 Audio Tokenizer 解码为波形
- 这种分离设计使得音频编解码和语言建模可以独立优化

### 19.6 为什么不用 SoundStream/EnCodec？

Higgs Audio V2 相比传统神经音频编解码器的优势：

| 特性 | SoundStream/EnCodec | Higgs Audio V2 |
|------|---------------------|----------------|
| 帧率 | 50 Hz (20ms/帧) | 75 Hz (13.33ms/帧) |
| 码本层数 | 通常 4-8 层 | 8 层 |
| 重构质量 | 良好 | 更好（专为语音优化） |
| 与 LLM 配合 | 通用设计 | 针对离散扩散优化 |

---

## 20. 训练时的随机掩码与 CFG 训练原理

### 20.1 训练时的随机掩码策略

训练时，每个样本的掩码比例和提示比例都是**随机采样**的，这增强了模型的泛化能力。

**源码位置**：`omnivoice/data/processor.py:66-147`

```python
def __call__(self, sample):
    # 1. 是否丢弃条件（drop_cond）
    drop_cond = random.uniform(0, 1) < self.drop_cond_ratio  # 默认 10%
    
    if drop_cond:
        prompt_ratio = 0.0    # 无 prompt
        drop_text = True      # 无文本
        use_language = False  # 无语言
        use_instruct = False  # 无指令
    else:
        # 2. 随机采样 prompt 比例 [0.0, 0.3]
        prompt_ratio = random.uniform(*self.prompt_ratio_range)
        drop_text = False
        use_language = random.uniform(0, 1) < self.language_ratio
        use_instruct = random.uniform(0, 1) < self.instruct_ratio
    
    # 3. 随机采样掩码比例 [0.0, 1.0]
    mask_ratio = random.uniform(*self.mask_ratio_range)
    
    # 4. 计算 prompt 长度
    prompt_length = int(audio_tokens.shape[1] * prompt_ratio)
    
    # 5. 只对 prompt 之后的区域做掩码
    maskable_region = audio_tokens[:, prompt_length:]
    token_mask = torch.rand(maskable_region.shape) < mask_ratio
    
    # 6. 被掩码的位置设为 MASK_ID，其余保持不变
    audio_inputs[:, prompt_length:][token_mask] = MASK_ID
    
    # 7. 只有被掩码的位置计算 loss
    audio_labels[:, prompt_length:][~token_mask] = -100
```

### 20.2 不同掩码比例的效果

| mask_ratio | prompt 可见 | 目标可见 | 模型学习任务 |
|-----------|------------|---------|------------|
| 0% | 全部 | 全部 | 无掩码（简单，loss 很低） |
| 25% | 全部 | 75% | 少量补全 |
| 50% | 全部 | 50% | 中等难度补全 |
| 75% | 全部 | 25% | 大量补全 |
| 100% | 全部 | 0% | 完全生成（最难，类似推理） |

**为什么 mask_ratio_range = [0.0, 1.0]？**

- 掩码比例在 0%~100% 之间均匀随机采样
- 低掩码比例（如 10%）：模型学习局部修复能力
- 高掩码比例（如 90%）：模型学习几乎从头生成的能力
- 这种"课程学习"效果让模型逐步适应从简单到困难的生成任务

### 20.3 Prompt 比例的影响

| prompt_ratio | 可见音频前缀 | 效果 |
|-------------|------------|------|
| 0% | 无 | 类似 TTS，从文本生成音频 |
| 10% | 前面 10% 的 token | 语音克隆，有少量声学提示 |
| 30% | 前面 30% 的 token | 强语音克隆，声音更像参考 |

**注意**：prompt_ratio 只在**训练时随机采样**。推理时的 prompt 长度由参考音频的实际时长决定。

### 20.4 Drop Cond 与 CFG 训练

**drop_cond_ratio = 0.1**（默认）意味着 10% 的训练样本会**完全丢弃条件信息**（无文本、无语言、无指令、无 prompt）。

**为什么需要 drop cond？**

这是 **Classifier-Free Guidance (CFG)** 的训练基础：

```
训练时：
  90% 的样本：有条件（文本 + 参考音频）→ 学习条件分布 P(audio | text, ref)
  10% 的样本：无条件（无任何条件）→ 学习无条件分布 P(audio)

推理时：
  log_probs = log P_cond + guidance_scale × (log P_cond - log P_uncond)
             = 有条件预测 + 引导强度 × (有条件 - 无条件)
```

**类比**：就像画画时，90% 的时间有模特（有条件），10% 的时间凭想象（无条件）。训练后，模型知道"有模特时该怎么画"以及"没有模特时的平均画风"。CFG 就是用两者的差异来**增强风格跟随**。

### 20.5 训练配置参数汇总

```json
{
    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0, 0.3],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0
}
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audio_codebook_weights` | [8,8,6,6,4,4,2,2] | 8 层 RVQ 的 loss 权重，高层权重低（细节层不那么重要） |
| `drop_cond_ratio` | 0.1 | 丢弃条件的概率（CFG 训练用） |
| `prompt_ratio_range` | [0.0, 0.3] | 提示音频占总音频的比例范围 |
| `mask_ratio_range` | [0.0, 1.0] | 掩码比例范围 |
| `language_ratio` | 0.8 | 使用语言标识的概率 |
| `use_pinyin_ratio` | 0.0 | 使用拼音替代文本的概率（中文数据增强） |
| `instruct_ratio` | 0.0 | 使用指令的概率 |
| `only_instruct_ratio` | 0.0 | 仅用指令（无参考音频）的概率 |

---

## 21. Checkpoint 文件结构与模型加载

### 21.1 Checkpoint 目录结构

```
checkpoint-63000/
├── audio_tokenizer -> /root/.cache/.../audio_tokenizer/   # 符号链接
├── chat_template.jinja                                    # 对话模板
├── config.json                                            # 模型配置
├── model.safetensors                                      # 模型权重 (~2.3GB)
├── optimizer.bin                                          # 优化器状态 (~4.9GB)
├── scheduler.bin                                          # 学习率调度器状态
├── random_states_0.pkl                                    # 随机种子状态（GPU 0）
├── random_states_1.pkl                                    # 随机种子状态（GPU 1）
├── tokenizer.json                                         # 文本 tokenizer 词汇表
├── tokenizer_config.json                                  # Tokenizer 配置
└── train_config.json                                      # 训练配置（保存时的副本）
```

### 21.2 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.safetensors` | ~2.3GB | **模型权重**。HuggingFace safetensors 格式，可直接加载 |
| `optimizer.bin` | ~4.9GB | **优化器状态**。AdamW 的一阶/二阶动量，用于断点续训 |
| `scheduler.bin` | ~1.5KB | **学习率调度器状态**。当前 step、warmup 进度等 |
| `random_states_*.pkl` | ~15KB × N | **随机种子状态**。每个 GPU 一个，保证断点续训的可复现性 |
| `config.json` | ~2KB | **模型配置**。hidden_size、num_layers、audio_vocab_size 等 |
| `tokenizer.*` | ~11MB | **文本 tokenizer**。BPE 词汇表和配置 |
| `audio_tokenizer` | 符号链接 | **音频 tokenizer**。指向独立的 Higgs Audio V2 目录 |
| `train_config.json` | ~1.5KB | **训练配置备份**。lr、batch_size、steps 等（仅备份，不用于加载） |

### 21.3 加载流程

**`OmniVoice.from_pretrained()` 内部流程**：

```python
def from_pretrained(path, device_map="auto", dtype=torch.float16):
    # 1. 读取 config.json
    config = OmniVoiceConfig.from_json_file(f"{path}/config.json")
    
    # 2. 创建模型结构
    model = OmniVoice(config)
    
    # 3. 加载权重
    state_dict = load_file(f"{path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    
    # 4. 加载文本 tokenizer
    model.text_tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 5. 加载音频 tokenizer（从符号链接）
    tokenizer_path = Path(path) / "audio_tokenizer"
    model.audio_tokenizer = HiggsAudioV2Tokenizer.from_pretrained(tokenizer_path)
    
    # 6. 设置设备和精度
    model = model.to(device=device_map, dtype=dtype)
    
    return model
```

### 21.4 断点续训

**从 checkpoint 继续训练**：

```bash
# 方式 1：恢复 optimizer/scheduler/random_states（完全续训）
RESUME_FROM=exp/children_finetune/checkpoint-30000 bash finetune_children.sh --stage 2

# 方式 2：仅加载模型权重（新优化器/新 schedule）
# 在 train_config.json 中设置：
{
    "init_from_checkpoint": "exp/.../checkpoint-30000",
    "resume_from_checkpoint": null
}
```

**关键区别**：
- `resume_from_checkpoint`：恢复完整的训练状态（模型 + 优化器 + 调度器 + 随机种子），学习率从断点继续
- `init_from_checkpoint`：只加载模型权重，优化器和调度器重新初始化，学习率从 warmup 重新开始

### 21.5 手动导出和转换

**导出为 HuggingFace 格式（用于推理）**：

```python
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained("exp/children_finetune/checkpoints/checkpoint-63000")

# 保存为独立目录（包含所有必要文件）
model.save_pretrained("exported_model/")
# 会保存：model.safetensors, config.json, tokenizer, audio_tokenizer 符号链接
```

---

*文档版本：v2.1（深度版） | 基于 OmniVoice v0.1.5 | 覆盖 ~3000 行核心代码解析*
