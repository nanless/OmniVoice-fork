# 评测

> 英文版：[evaluation.md](../evaluation.md)

使用标准 TTS 指标评测 OmniVoice：WER（可懂度）、SIM-o（说话人相似度）、UTMOS（自然度）。

## 支持的测试集

| 测试集 | 语言 | WER 模块 | 指标 |
|---|---|---|---|
| **LibriSpeech-PC** | 英语 | HuBERT WER | WER + 说话人相似度 + MOS |
| **Seed-TTS (en)** | 英语 | Whisper WER | WER + MOS |
| **Seed-TTS (zh)** | 中文 | Paraformer WER | WER + MOS |
| **FLEURS** | 102 种语言 | Omnilingual-ASR WER | WER（按语言 + 宏平均） |
| **MiniMax Multilingual** | 24 种语言 | Whisper + Paraformer | WER + MOS |

## 环境准备

```bash
pip install omnivoice[eval]
# 或
uv sync --extra eval
```

## 快速开始

```bash
cd examples
bash run_eval.sh
# run_eval.sh 会：
# (1) 下载所需测试集与评测模型；
# (2) 对每个测试集进行推理与评测。
```

## 指标说明

### WER（词错误率）
用 ASR 转写生成语音并与参考文本对比，衡量可懂度。越低越好。部分语言实际使用 CER（字错误率）。

- **LibriSpeech-PC**：基于 HuBERT 的 ASR
- **Seed-TTS**：英语用 Whisper，中文用 Paraformer
- **MiniMax**：非中文用 Whisper，中文用 Paraformer
- **FLEURS**：Omnilingual-ASR 多语言模型

### 说话人相似度（Speaker Similarity）
参考音频与生成音频的说话人嵌入（ECAPA-TDNN + WavLM）余弦相似度。越高越好。

### UTMOS（预测 MOS）
从音频预测平均意见分（MOS）的神经网络。越高越好。
