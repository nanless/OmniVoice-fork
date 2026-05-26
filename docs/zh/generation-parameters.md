# 生成参数

> 英文版：[generation-parameters.md](../generation-parameters.md)

参数可通过 `model.generate(...)` 的关键字参数传入，或使用 `OmniVoiceGenerationConfig` 数据类。下文列出全部参数及所属类别。

```python
# 1) 直接传关键字参数
audio = model.generate(text="Hello world", num_step=32, guidance_scale=2.0)

# 2) 通过 OmniVoiceGenerationConfig
from omnivoice import OmniVoiceGenerationConfig

config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
audio = model.generate(text="Hello world", generation_config=config)
```

## 解码（Decoding）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `num_step` | int | 32 | 迭代去掩码步数。越大质量通常越好，但越慢；快速推理可用 16。 |
| `denoise` | bool | True | 在输入前添加 `<\|denoise\|>` token，提示模型生成更干净的语音。 |
| `guidance_scale` | float | 2.0 | 无分类器引导（CFG）强度。 |
| `t_shift` | float | 0.1 | 噪声调度的时间步偏移；越小越强调解码早期步骤。 |

## 采样（Sampling）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `position_temperature` | float | 5.0 | 掩码位置选择的温度。0 为贪心（确定性）；越大随机性越高。 |
| `class_temperature` | float | 0.0 | 每步 token 采样温度。0 为贪心；越大随机性越高。 |
| `layer_penalty_factor` | float | 5.0 | 对更深 codebook 层的惩罚，促使较低层先被去掩码。 |

## 时长与语速（Duration & Speed）

可传单个值（作用于全部样本），或按样本的列表（批处理时有用）：

```python
# 固定 10 秒输出
audio = model.generate(text="Hello, this is a test of duration control", duration=10.0)

# 更快（约为估计时长的 1.2 倍语速）
audio = model.generate(text="Hello, this is a test of duration control", speed=1.2)
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `duration` | float 或 list[float \| None] | None | 固定输出时长（秒）。设置后覆盖 `speed`。 |
| `speed` | float 或 list[float \| None] | None | 语速因子。>1 更短更快，<1 更长更慢。设置 `duration` 时忽略。二者均为 None 时默认 1.0。 |

优先级：`duration` > `speed`。

> **注意：** 使用 `duration` 时，默认后处理可能裁掉尾部静音，使实际时长略短于设定值。若需要**严格**等于指定时长，请设 `postprocess_output=False` 以关闭去静音。

## 前/后处理（Pre/Post Processing）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `preprocess_prompt` | bool | True | 是否对语音克隆参考音频做预处理（去除参考中的长静音、在参考文本末尾补标点）。 |
| `postprocess_output` | bool | True | 是否对生成音频做后处理（去除长静音等）。 |

## 长文本生成（Long-Form Generation）

为在较低显存下稳定生成长语音，当估计生成长度超过 `audio_chunk_duration` 时，文本会自动切分为多段，每段约生成 `audio_chunk_duration` 秒音频，从而支持任意长文本与近恒定的显存占用。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `audio_chunk_duration` | float | 15.0 | 长文本切分时的目标片段时长（秒）。 |
| `audio_chunk_threshold` | float | 30.0 | 估计音频时长超过该值（秒）时启用切分。 |
