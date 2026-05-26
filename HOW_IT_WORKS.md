# OmniVoice 底层原理详解：从训练到推理

> 本文用**尽量通俗的语言 + 图示**，解释 OmniVoice 到底是怎么训练和生成语音的。
>
> 读完这篇文章，你会明白：
> - 为什么微调不需要 speaker_id
> - 模型是怎么"学习克隆声音"的
> - 推理时那 32 步（或 50 步）到底在干什么
> - Classifier-Free Guidance 是什么东西

---

## 目录

1. [核心架构：把 LLM 改成语音模型](#1-核心架构把-llm-改成语音模型)
2. [音频是怎么表示的？——RVQ Token](#2-音频是怎么表示的rvq-token)
3. [训练过程详解](#3-训练过程详解)
4. [推理过程详解：Voice Clone 的每一步](#4-推理过程详解voice-clone-的每一步)
   - [4.1 推理时输入序列的构造](#41-推理时输入序列的构造)
   - [4.2 文本是如何拼接的？](#42-文本是如何拼接的)
   - [4.3 迭代去掩码解码](#43-迭代去掩码解码)
   - [4.4 时间步调度](#44-时间步调度)
   - [4.5 Classifier-Free Guidance](#45-classifier-free-guidance)
   - [4.6 Layer Penalty](#46-layer-penalty)
   - [4.7 长文本分块生成](#47-长文本分块生成)
5. [训练 vs 推理：一张图看懂区别](#5-训练-vs-推理一张图看懂区别)
6. [关键概念速查](#6-关键概念速查)

---

## 1. 核心架构：把 LLM 改成语音模型

### 1.1 整体结构

OmniVoice 不是从零训练的，而是**改造了一个现成的 LLM**（Qwen3-0.6B）。改造只加了两个东西：

```
┌─────────────────────────────────────────────────────────────┐
│                    OmniVoice 模型架构                          │
├─────────────────────────────────────────────────────────────┤
│  输入：Token IDs [Batch, 8, SeqLen]                           │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │  Text Embeddings │    │  Audio Embeddings │                │
│  │   (原有)         │    │   (新增)          │                │
│  │   词向量表       │    │   8层codebook     │                │
│  │   大小：Vocab    │    │   共享一个大表    │                │
│  │                  │    │   大小：8 × 1025  │                │
│  └────────┬─────────┘    └────────┬─────────┘                │
│           │                       │                          │
│           └───────────┬───────────┘                          │
│                       ▼                                      │
│              加和后送入 LLM (Qwen3-0.6B)                      │
│                       │                                      │
│                       ▼                                      │
│              输出 Hidden States [B, S, D]                     │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │   Audio Heads    │                             │
│              │   (新增线性层)    │                             │
│              │   输出 8×1025    │                             │
│              │   的 logits      │                             │
│              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Audio Embeddings：8 层 Codebook 共享一个表

音频用 **RVQ（Residual Vector Quantization）** 编码成 8 层离散 token。每层有 1024 个可能的值 + 1 个 mask token = 1025 个。

**关键设计**：8 层 codebook 不是各自独立的 embedding 表，而是**共享一个大表**：

```python
# 一个大的 embedding 表，大小 = 8 × 1025 = 8200
self.audio_embeddings = nn.Embedding(
    num_codebook * audio_vocab_size,  # 8 * 1025 = 8200
    hidden_size  # 和 LLM 的 hidden size 一样
)

# 每层 codebook 的 ID 有一个偏移量
# Layer 0: id + 0
# Layer 1: id + 1025
# Layer 2: id + 2050
# ...
codebook_layer_offsets = [0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]
```

**为什么这样设计？**
- 参数量更小（1 个表 vs 8 个表）
- 不同层之间可以共享知识（比如低音调的特征）

**输入时的处理**（`_prepare_embed_inputs`）：

```python
# input_ids 形状: [Batch, 8, SeqLen]
# 8 是 codebook 层数

# 1. 文本位置（audio_mask=False）：用 Text Embedding
#    只取第 0 层（因为文本只有一层）
text_embeds = text_embedding(input_ids[:, 0, :])  # [B, S, D]

# 2. 音频位置（audio_mask=True）：用 Audio Embedding
#    8 层分别查表，然后相加
shifted_ids = input_ids + codebook_layer_offsets  # [B, 8, S]
audio_embeds = audio_embeddings(shifted_ids).sum(dim=1)  # [B, S, D]

# 3. 根据 audio_mask 选择用 text_embeds 还是 audio_embeds
final_embeds = torch.where(audio_mask, audio_embeds, text_embeds)
```

**通俗理解**：
- 文本和音频都变成向量，但查的表不同
- 音频是 8 层向量**加在一起**（就像 RGB 三个通道加在一起显示彩色）
- `audio_mask` 就像一个开关：文本位置用文本向量，音频位置用音频向量

### 1.3 Audio Heads：从 Hidden State 预测 8 层 Token

LLM 输出的 hidden state 形状是 `[Batch, SeqLen, HiddenDim]`。需要把它映射回 8 层音频 token 的预测：

```python
# 线性层：HiddenDim → 8 × 1025
audio_heads = nn.Linear(HiddenDim, 8 * 1025)

# 前向传播
logits = audio_heads(hidden_states)  # [B, S, 8 * 1025]

# reshape 成 [B, 8, S, 1025]
audio_logits = logits.view(B, S, 8, 1025).permute(0, 2, 1, 3)
# 最终形状: [Batch, 8_Layers, SeqLen, 1025_Vocab]
```

**损失计算**（`forward` 中的关键代码）：

```python
# 对每一层分别计算交叉熵
# audio_logits.permute: [B, 8, S, 1025] → [B, 1025, 8, S]
per_token_loss = F.cross_entropy(
    audio_logits.permute(0, 3, 1, 2),  # [B, 1025, 8, S]
    labels,                             # [B, 8, S]
    reduction="none",
    ignore_index=-100
)  # 输出: [B, 8, S]

# 每层取平均
layer_means = (per_token_loss * valid_mask).sum(dim=(0, 2)) / valid_mask.sum(dim=(0, 2))
# layer_means 形状: [8]，即 8 层的平均 loss

# 加权求和（越高层权重越小）
weights = [8, 8, 6, 6, 4, 4, 2, 2]  # 归一化后使用
loss = (layer_means * weights).sum()
```

**为什么不同层权重不同？**

| Codebook 层 | 权重 | 负责什么 |
|------------|------|---------|
| Layer 0 | 8 | 最粗粒度（整体轮廓、基频） |
| Layer 1 | 8 | 粗粒度 |
| Layer 2 | 6 | 中等粒度 |
| Layer 3 | 6 | 中等粒度 |
| Layer 4 | 4 | 细粒度 |
| Layer 5 | 4 | 细粒度 |
| Layer 6 | 2 | 最细粒度（细节、噪声） |
| Layer 7 | 2 | 最细粒度 |

**低层（Layer 0-1）决定"说什么"和"谁在说"**，所以权重更高。高层只补充细节。

---

## 2. 音频是怎么表示的？——RVQ Token

### 2.1 从波形到 Token

原始音频是波形（连续的振幅值）。OmniVoice 用一个叫 **Higgs Audio V2 Tokenizer** 的模型把它转成离散 token：

```
波形 [1, 24000×T]  ──→  Encoder ──→  连续向量 ──→  RVQ量化 ──→  离散Token [8, T]
                              (神经网络)           (8层码本)
```

**RVQ（残差向量量化）** 的工作方式：

```
原始音频向量: v

Layer 0: 在码本0中找到最近的向量 c0
         残差 = v - c0

Layer 1: 在码本1中找到最近的向量 c1（匹配残差）
         残差 = 残差 - c1

Layer 2: 在码本2中找到最近的向量 c2
         ...

Layer 7: 在码本7中找到最近的向量 c7

最终表示: v ≈ c0 + c1 + c2 + ... + c7
Token IDs: [id0, id1, id2, ..., id7] 每层一个 ID
```

**帧率**：约 75 Hz（每秒 75 个 token）。所以 10 秒音频 ≈ 750 个 token。

### 2.2 从 Token 回波形（Decoder）

```
Token [8, T] ──→ 查表得到 8 个向量 ──→ 相加 ──→  Decoder ──→  波形 [1, 24000×T]
```

---

## 3. 训练过程详解

### 3.1 数据准备：从一段音频构造训练样本

这是**最关键**的部分，理解了这个就理解了 OmniVoice 的一切。

假设你有一段音频，已经被转成 token，形状是 `[8, 1000]`（8 层 codebook，1000 个时间步）。

**Step 1：随机决定 Prompt 长度**

```python
prompt_ratio = random.uniform(0.0, 0.3)  # 比如 0.2
prompt_length = int(1000 * 0.2)  # = 200
```

**Step 2：把音频分成两部分**

```
原始音频 Token [8, 1000]
├─ Prompt 部分:    [8, 0:200]   →  不 mask，作为条件输入
└─ 目标部分:       [8, 200:1000] →  部分 mask，让模型预测
```

**Step 3：对目标部分随机 Mask**

```python
mask_ratio = random.uniform(0.0, 1.0)  # 比如 0.6
target_region = audio_tokens[:, 200:]
mask = torch.rand(target_region.shape) < 0.6  # 60% 的位置被 mask
audio_inputs[:, 200:][mask] = MASK_TOKEN_ID  # 1024
```

Mask 后的样子：

```
Layer 0: [  5,  12,  88, ...,  42, 1024, 1024,  77, 1024, ...]  ← 1024 = MASK
Layer 1: [ 23,  55, 1024, ..., 1024,  33, 1024, 1024,  91, ...]
...
```

**Step 4：构造输入序列**

```
输入序列 input_ids [8, TotalLen]:
┌─────────────────────────────────────────────────────────────────┐
│ [Style Tokens]  +  [Text Tokens]  +  [Prompt Audio]  +  [Masked Audio] │
│    20 tokens         50 tokens          200 tokens        600 tokens   │
└─────────────────────────────────────────────────────────────────┘

Label (正确答案) [8, TotalLen]:
┌─────────────────────────────────────────────────────────────────┐
│ [ -100 ]        +  [ -100 ]       +  [ -100 ]      +  [真实Token]  │
│  (不算loss)         (不算loss)        (不算loss)      (算loss)      │
│                                    ↑ prompt部分    ↑ mask=1024的位置 │
└─────────────────────────────────────────────────────────────────┘
```

**audio_mask**（告诉模型哪些是音频位置）：

```
[False, False, ..., False, True, True, ..., True]
  ←  style+text  →    ←      audio 区域      →
```

### 3.2 条件丢弃（Classifier-Free Guidance 训练）

训练时有概率**扔掉所有条件**：

```python
if random.random() < drop_cond_ratio:  # 默认 10%
    # 只保留音频，扔掉 style 和 text
    input_ids = audio_inputs  # 只有 [8, 1000]
    labels = audio_labels
```

**为什么要这样？**

为了让模型学会"无条件生成"。推理时我们用 **CFG（Classifier-Free Guidance）**：
- 同时跑一次"有条件"（给了 text + ref_audio）
- 同时跑一次"无条件"（只给 text）
- 最终输出 = 有条件输出 + guidance_scale × (有条件 - 无条件)

这让生成结果更"贴合"条件，而不是随便生成。

### 3.3 完整的训练流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练流程                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 加载数据                                                          │
│     WebDataset → audio_tokens [8, T] + text + language_id            │
│                                                                      │
│  2. 每样本随机处理 (Processor)                                        │
│     ├─ 决定是否 drop_cond (10% 概率扔掉条件)                          │
│     ├─ 随机选 prompt_ratio (0%~30% 作为参考音频)                      │
│     ├─ 随机选 mask_ratio (0%~100% 的目标区域被 mask)                  │
│     ├─ 构造 style tokens (<|lang_start|>zh<|lang_end|>...)            │
│     ├─ 构造 text tokens (<|text_start|>你好<|text_end|>)              │
│     └─ 拼接: [Style] + [Text] + [Prompt] + [Masked_Target]           │
│                                                                      │
│  3. 组 Batch (Collator)                                               │
│     ├─ flex_attention: 把多个样本拼接成一个长序列 [1, 8, batch_tokens] │
│     └─ sdpa: 填充到 max_len [B, 8, max_len]                          │
│                                                                      │
│  4. 前向传播 (Forward)                                                │
│     ├─ Embedding: 文本查文本表，音频查音频表(8层相加)                   │
│     ├─ LLM: Qwen3-0.6B 处理序列                                       │
│     └─ Audio Heads: 输出 [B, 8, S, 1025] logits                       │
│                                                                      │
│  5. 计算 Loss                                                         │
│     ├─ 只在 mask=1024 的位置计算交叉熵                                │
│     ├─ 8 层分别计算，加权求和 (权重 [8,8,6,6,4,4,2,2])                 │
│     └─ 反向传播，更新参数                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 为什么不需要 speaker_id？

现在你应该明白了：

**训练时，Prompt 就是同一段音频的开头部分。**

模型学习的是：
> "给定这段音频的开头（prompt）+ 这段文字，预测被 mask 掉的音频 token"

所以：
- 如果你的数据全是**同一个人**，模型总是看到"这个人的声音开头"→"这个人的声音后续"，自然学会了这个人的特征
- 如果你的数据是**多个人混合**，模型学会了"各种声音开头"→"各种声音后续"的通用映射
- **推理时**，你提供 `ref_audio` 作为 prompt，模型根据这段音频的特征生成新音频

**不需要显式标注 speaker**，因为 prompt 本身就隐含了说话人信息！

---

## 4. 推理过程详解：Voice Clone 的每一步

### 4.1 推理时输入序列的构造

推理和训练最大的区别：**推理时目标区域全是 MASK**。

```python
# 1. 把参考音频转成 token
ref_audio_tokens = audio_tokenizer.encode(ref_wav)  # [8, T_ref]

# 2. 估计目标长度（根据文字多少 + 参考音频长度估算）
num_target_tokens = estimate_duration(text, ref_text, T_ref)

# 3. 构造目标区域（全 MASK）
target_audio_tokens = torch.full((8, num_target_tokens), MASK_ID)  # 全 1024

# 4. 拼接输入序列
input_ids = [Style] + [Text] + [ref_audio_tokens] + [target_audio_tokens]

audio_mask = [False...False, True...True]
              ← style+text →  ← audio 区域 →
```

### 4.2 文本是如何拼接的？

这是一个关键细节：**推理时，参考音频的文本和目标文本会被拼接在一起**。

```python
# 源码中的 _combine_text 函数
def _combine_text(text, ref_text=None):
    if ref_text:
        full_text = ref_text.strip() + " " + text.strip()
    else:
        full_text = text.strip()
    return full_text

# 然后包装成
wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
```

**实际例子**：

```python
ref_text = "今天天气真好"        # 参考音频说了什么
text = "欢迎使用语音克隆系统"   # 你想让他说什么

# 模型实际看到的文本部分：
"<|text_start|>今天天气真好 欢迎使用语音克隆系统<|text_end|>"
```

**为什么要这样设计？**

| 你提供 ref_text | 模型能看到什么 | 效果 |
|----------------|-------------|------|
| **提供** | 既听到参考音频的声音，又知道参考音频说了什么 | **更好**：模型能理解语速、语调、发音习惯 |
| **不提供** | 只听到声音，不知道内容 | **可用但稍差**：缺少文本层面的对齐信息 |

**类比**：就像你模仿一个人说话，如果你**既听到他的声音又知道他说了什么**，模仿会更像；如果只听到声音但听不懂内容，模仿就会比较模糊。

所以 `ref_text` 是**可选但推荐提供**的。如果不提供，模型会用 ASR（Whisper）自动转写参考音频。

### 4.3 迭代去掩码解码（Iterative Unmasking Decoding）

这是 OmniVoice 推理的核心算法。不是一次性预测所有 token，而是**N 步逐步 reveal**。

```python
# 初始化：目标区域全 mask
tokens = [[1024, 1024, 1024, ...],   # Layer 0
          [1024, 1024, 1024, ...],   # Layer 1
          ...
          [1024, 1024, 1024, ...]]  # Layer 7

# N 步迭代（默认 32 步）
for step in range(32):
    # 1. 把当前 token 填入输入序列
    input_ids[..., -num_target:] = tokens

    # 2. 前向传播，得到 logits [B, 8, S, 1025]
    logits = model(input_ids, audio_mask)

    # 3. 只对目标区域预测
    target_logits = logits[..., -num_target:, :]  # [B, 8, T, 1025]

    # 4. 用 CFG：同时跑 conditional 和 unconditional
    #    conditional: 有 ref_audio + text
    #    unconditional: 只有 text（没有 ref_audio）
    #    final_logits = cond + guidance_scale * (cond - uncond)

    # 5. 对每个 mask 位置，选择置信度最高的 token
    pred_tokens = argmax(logits)  # [B, 8, T]
    confidence = max(logits)      # [B, 8, T]

    # 6. 选择本轮要 reveal 的位置
    #    只选当前还是 mask (1024) 的位置
    #    按置信度排序，选最高的 k 个
    k = schedule[step]  # 每步 reveal 的数量，从多到少
    topk_positions = topk(confidence[mask == 1024], k)

    # 7. 把这些位置的 mask 替换为预测的 token
    tokens[topk_positions] = pred_tokens[topk_positions]

# 最终 tokens 全部填满，送入 audio tokenizer 解码成波形
```

### 4.4 时间步调度（Schedule）

32 步不是均匀 reveal 的。用一个非线性调度：

```
Step  0: reveal 10% of masks
Step  1: reveal 10% of remaining masks
Step  2: reveal  9% of remaining masks
...
Step 31: reveal all remaining masks (最后一步全部填满)
```

```python
def get_time_steps(num_step=32, t_shift=0.1):
    # 线性时间步 [0, 1/32, 2/32, ..., 1]
    timesteps = linspace(0, 1, num_step + 1)

    # 非线性变换：前期步长更小（更谨慎），后期更大
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps

# 每步 reveal 的数量 = total_masks × (timesteps[i+1] - timesteps[i])
```

### 4.5 Classifier-Free Guidance（CFG）详解

CFG 是 Voice Clone 质量的**关键**。

**训练时的准备**：
- 10% 的样本被"无条件训练"（drop_cond=True）
- 模型学会了：
  - `p(audio | text, ref_audio)` — 有条件分布
  - `p(audio | text)` — 无条件分布

**推理时的操作**：

```python
# 同时准备两组输入
# Conditional: 完整的 [Style] + [Text] + [Ref_Audio] + [Target]
# Unconditional: 只有 [Text] + [Target]（去掉 ref_audio 和 style）

# 两组输入拼成一个 batch
batch_input_ids = [cond_input, cond_input, ..., uncond_input, uncond_input, ...]
                  # 前 B 个是 conditional，后 B 个是 unconditional

# 一次前向传播
logits = model(batch_input_ids)  # [2B, 8, S, 1025]

# 分离 cond 和 uncond
c_logits = logits[0:B]    # 有条件
u_logits = logits[B:2B]   # 无条件

# CFG 公式
guidance_scale = 2.0  # 默认
final_logits = c_logits + guidance_scale * (c_logits - u_logits)

# 直观理解：
# (c_logits - u_logits) = "ref_audio 带来的方向"
# guidance_scale = 2.0 表示"沿着这个方向走 2 倍远"
# 结果：生成的音频更像参考音频
```

**guidance_scale 调参**：

| guidance_scale | 效果 |
|---------------|------|
| 1.0 | 几乎不用参考音频，比较自由 |
| 2.0 | 默认，平衡质量和多样性 |
| 2.5~3.0 | 更像参考音频，但可能过度收敛 |

### 4.6 Layer Penalty：让低层先 reveal

```python
# 选择 reveal 位置时，给高层加惩罚
scores = confidence - (layer_id * layer_penalty_factor)
# layer_id: 0, 1, 2, 3, 4, 5, 6, 7
# penalty: 默认 5.0

# Layer 0 的分数 = confidence - 0
# Layer 7 的分数 = confidence - 35

# 所以低层（Layer 0-1）更容易先被 reveal
```

**为什么这样？**
- 低层决定"说什么"和"谁在说"
- 高层只补充细节
- 先确定大框架，再填细节，更稳定

### 4.7 长文本分块生成

如果文本太长，VRAM 不够，会自动分块：

```python
# 估计音频超过 30 秒时触发分块
if estimated_duration > 30:
    chunks = split_text(text, chunk_size=15秒)

# 第 0 块：正常生成（可以带 ref_audio）
audio_0 = generate(chunks[0], ref_audio=ref_audio)

# 第 1~N 块：用第 0 块的音频作为 ref_audio
for chunk in chunks[1:]:
    audio_i = generate(chunk, ref_audio=audio_0)
```

这样 VRAM 占用几乎不变，可以生成任意长的音频。

---

## 5. 训练 vs 推理：一张图看懂区别

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│            训  练                    │            推  理                    │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  Prompt: 同段音频的前缀 (0~30%)      │  Prompt: 用户提供的 ref_audio        │
│  ─────────────────────────────      │  ─────────────────────────────       │
│  音频: [■■■■|□□□□□□□□□□]            │  音频: [ref■■■■|□□□□□□□□□□]          │
│        ↑prompt  ↑mask目标           │        ↑参考     ↑要生成的           │
│                                     │                                     │
│  Mask: 随机 mask 目标区域的一部分    │  Mask: 目标区域 100% mask            │
│  ─────────────────────────────      │  ─────────────────────────────       │
│  [■■■■|□□■□■□□□■□□]                 │  [ref■■■■|□□□□□□□□□□]                │
│        ↑已知  ↑部分 mask             │        ↑已知    ↑全部 mask           │
│                                     │                                     │
│  Text: 一定有                        │  Text: 一定有                        │
│  Style: 随机决定是否加 language/     │  Style: 用户指定 language/           │
│         instruct                     │         instruct（可选）              │
│                                     │                                     │
│  Drop Cond: 10% 概率扔掉所有条件     │  CFG: 同时跑 cond + uncond           │
│  ─────────────────────────────      │  ─────────────────────────────       │
│  让模型学会无条件生成                │  用 guidance_scale 控制条件强度      │
│                                     │                                     │
│  目标: 预测被 mask 的真实 token      │  目标: 逐步 reveal 所有 mask         │
│  ─────────────────────────────      │  ─────────────────────────────       │
│  一步完成（一次前向 + 一次 loss）    │  N 步迭代（32/50 次前向传播）        │
│                                     │                                     │
│  Loss: 只在 mask 位置算交叉熵        │  无 loss，每步选置信度最高的 token   │
│  ─────────────────────────────      │  ─────────────────────────────       │
│  8 层加权: [8,8,6,6,4,4,2,2]        │  每层独立预测，但受 layer_penalty    │
│                                     │  影响选择顺序                        │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
```

---

## 6. 关键概念速查

### 6.1 术语表

| 术语 | 解释 |
|------|------|
| **RVQ** | Residual Vector Quantization，残差向量量化。把音频向量用多层码本逐步逼近 |
| **Codebook** | 码本，一个向量查找表。每个 ID 对应一个固定向量 |
| **Token** | 离散整数 ID。音频被量化后的数字表示 |
| **Mask Token** | 特殊 token ID=1024，表示"这个位置需要预测" |
| **Prompt** | 参考音频/条件音频。训练时是音频前缀，推理时是 ref_audio |
| **CFG** | Classifier-Free Guidance，通过对比"有条件"和"无条件"输出来增强条件控制 |
| **Diffusion** | 扩散模型。OmniVoice 用"离散扩散"，通过逐步去 mask 生成 |
| **Unmasking** | 去掩码。把 mask token (1024) 替换为真实 token |
| **Frame Rate** | 帧率。音频 tokenizer 每秒输出多少个 token，约 75 Hz |
| **Step** | 推理时的迭代步数。默认 32 步，越高质量越好但越慢 |

### 6.2 核心超参数对照

| 参数 | 训练 | 推理 | 作用 |
|------|------|------|------|
| `prompt_ratio_range` | [0.0, 0.3] | — | 训练时 prompt 占音频的比例 |
| `mask_ratio_range` | [0.0, 1.0] | — | 训练时目标区域 mask 的比例 |
| `drop_cond_ratio` | 0.1 | — | 训练时无条件生成的概率 |
| `num_step` | — | 32 | 推理迭代步数 |
| `guidance_scale` | — | 2.0 | CFG 强度 |
| `t_shift` | — | 0.1 | 时间步调度偏移 |
| `layer_penalty_factor` | — | 5.0 | 高层 reveal 的惩罚 |
| `position_temperature` | — | 5.0 | 选择 reveal 位置的随机性 |
| `class_temperature` | — | 0.0 | 选择 token 值的随机性 |

### 6.3 数据维度速查

```
音频波形:        [1, 24000 * seconds]    1 通道，24kHz 采样率
Audio Tokens:    [8, T]                  8 层 codebook，T ≈ 75 * seconds
Text Tokens:     [1, L]                  文本 tokenizer 输出
Input IDs:       [Batch, 8, SeqLen]      8 层重复文本 token，音频位置用音频 token
Labels:          [Batch, 8, SeqLen]      -100 = 不算 loss
Audio Mask:      [Batch, SeqLen]         True = 音频位置，False = 文本位置
Logits:          [Batch, 8, SeqLen, 1025] 每层每个位置的 1025 类预测
Hidden States:   [Batch, SeqLen, 1536]   Qwen3-0.6B 的 hidden size
```

### 6.4 训练和推理的注意力机制

**训练时**：**双向注意力**（Bidirectional）
- 所有 token 都能看到所有其他 token
- 因为目标是"填空"（masked token prediction），不是"预测下一个"

**推理时**：也是**双向注意力**
- 但输入序列中只有目标区域是 mask
- 模型反复看到"已 reveal 的 token + 还未 reveal 的 mask"
- 每次前向传播更新一部分 mask

这和 GPT（自回归、因果注意力）完全不同！OmniVoice 更像 BERT 的 Masked LM，但是多步迭代的。

---

**现在你应该完全理解了：**
1. OmniVoice 是一个改造后的 LLM，用双向注意力做"音频填空"
2. 训练时从同段音频截取前缀作为 prompt，不需要 speaker_id
3. 推理时用迭代去 mask 的方式生成音频，配合 CFG 控制声音特征
4. Voice Clone 的本质是："用参考音频作为 prompt，生成风格一致的音频"
