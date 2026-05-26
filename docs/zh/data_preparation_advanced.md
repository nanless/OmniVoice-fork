# 高级数据准备

> 英文版：[data_preparation_advanced.md](../data_preparation_advanced.md)

高级流程在基础 token 化流程上增加**降噪**与**提示音噪声增强**。每个阶段均可选。

## 前置条件

- **降噪**：Sidon 模型检查点（`feature_extractor_cuda.pt`、`decoder_cuda.pt`），见 https://huggingface.co/sarulab-speech/sidon-v0.1/tree/main
- **噪声增强**：带 `data.lst` 清单的 noise + RIR tar 分片

## 流程概览

```
步骤 1（可选）：降噪
  原始音频 → Sidon 降噪 → 干净音频

步骤 2：Token 化（可选噪声增强）
  干净音频 + 前缀噪声增强 → audio tokenizer → tokens
```

## 降噪

使用 [Sidon](https://github.com/sarulab-speech/Sidon) 语音增强模型去除原始音频背景噪声。

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python -m omnivoice.scripts.denoise_audio \
    --input_jsonl data.jsonl \
    --tar_output_pattern data/denoised/audios/shard-%06d.tar \
    --jsonl_output_pattern data/denoised/txts/shard-%06d.jsonl \
    --feature_extractor_path /path/to/sidon_feature_extractor_cuda.pt \
    --decoder_path /path/to/sidon_decoder_cuda.pt \
    --target_sample_rate 24000 \
    --batch_duration 200.0
```

流程说明：
1. 读取 JSONL 清单
2. 对每条音频运行 Sidon 降噪
3. 输出降噪后的 WebDataset tar/jsonl 分片
4. 在 `data/denoised/` 生成 `data.lst`

> 若已有自定义 WebDataset，也可传 `--input_manifest /path/to/data.lst`。
> 下一步可将生成的 `data.lst` 用 `--input_manifest` 传给 `omnivoice.scripts.extract_audio_tokens` 做 token 提取。

### 带噪声增强的 Token 化

在 token 化时对**提示音频**加入环境噪声与房间混响，使模型在推理时对嘈杂参考音更鲁棒。注意：训练中仅对一小部分数据做噪声增强，以保证模型在干净参考下仍能生成高质量音频。

需要两个额外的 WebDataset 格式数据集：
- **噪声录音**：环境噪声 tar 分片及 `data.lst`
- **房间冲激响应（RIR）**：RIR tar 分片及 `data.lst`

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"
python -m omnivoice.scripts.extract_audio_tokens_add_noise \
    --input_jsonl data.jsonl \
    --tar_output_pattern data/tokens/shard-%06d.tar \
    --jsonl_output_pattern data/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --noise_manifest data/noise_shards/data.lst \
    --rir_manifest data/rir_shards/data.lst \
    --nj_per_gpu 3
```

> 若已有 WebDataset，也可传 `--input_manifest /path/to/data.lst`。
