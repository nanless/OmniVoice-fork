# 训练

> 英文版：[training.md](../training.md)

## 训练配置

训练由 JSON 训练配置文件与 JSON 数据配置文件控制。

可参考 [examples/config/](../examples/config/) 中的现成配置。

Emilia 训练配置：[examples/config/train_config_emilia.json](../examples/config/train_config_emilia.json)

Emilia 数据配置：[examples/config/data_config_emilia.json](../examples/config/data_config_emilia.json)

训练配置文件主要字段：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `llm_name_or_path` | 本地 LLM 路径或 Hugging Face ID | Qwen/Qwen3-0.6B |
| `steps` | 总训练步数 | 300,000 |
| `learning_rate` | 峰值学习率 | 1e-4 |
| `batch_tokens` | 每张 GPU 每批 token 数 | 8192 |
| `attn_implementation` | 注意力后端：`"flex_attention"` 或 `"sdpa"` | `"flex_attention"` |

`output_dir` 与 `data_config` 通过命令行传入（见下文）。

## 注意力实现

默认使用 `flex_attention`，需要 PyTorch ≥ 2.5 且兼容 GPU（如 NVIDIA Ampere 及更新）。若环境不支持，请在训练配置中将 `attn_implementation` 设为 `"sdpa"`。可参考 [examples/config/train_config_finetune_sdpa.json](../examples/config/train_config_finetune_sdpa.json)：

```json
{
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 64
}
```

`"sdpa"` 使用 PyTorch 内置缩放点积注意力，硬件兼容性更广。

以下字段仅在 `attn_implementation != "flex_attention"` 时生效：

| 字段 | 说明 | 默认值 |
|---|---|---|
| `max_sample_tokens` | 单样本最大 token 长度，超长样本丢弃 | 2000 |
| `min_sample_tokens` | 单样本最小 token 长度，过短样本丢弃 | 50 |
| `max_batch_size` | 每批样本数上限 | 64 |

`batch_tokens` 仍是显存的主要控制项——设定每批 token 预算。`max_batch_size` 为防止大量短样本凑成过大 batch 维度的安全上限。

### 批处理策略

两种后端**自动**选用不同批处理策略：

| 后端 | 批处理策略 | Batch 形状 | 说明 |
|---|---|---|---|
| `flex_attention` | 序列打包（sequence packing） | `[1, C, batch_tokens]` | 多样本拼成一条长序列；用 `document_ids` 标记文档边界 |
| `sdpa` | 按长度分组填充 | `[B, C, max_len]` | 相近 token 长度的样本同批，填充到该批最大长度 |

**为何不同？**

- `flex_attention` 下序列打包更省显存：用紧凑的 `BlockMask`（非稠密矩阵）描述跨文档的注意力范围。
- `sdpa` 下采用按长度分组填充：相似长度样本同批并填充到局部最大值，轻量 `[B, 1, max_len, max_len]` 布尔 mask 即可，浪费填充较少。

## 启动训练

```bash
accelerate launch \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config config/train_config_emilia.json \
    --data_config config/data_config_emilia.json \
    --output_dir exp/omnivoice_emilia
```

## 断点续训

在训练配置中设置 `resume_from_checkpoint`：

```json
{
    "resume_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## 从预训练模型初始化

从已有 OmniVoice 检查点微调时：

```json
{
    "init_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## 监控

训练日志写入 TensorBoard：

```bash
tensorboard --logdir exp/omnivoice_emilia/tensorboard
```
