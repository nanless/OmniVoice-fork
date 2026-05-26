# 数据准备

> 英文版：[data_preparation.md](../data_preparation.md)

OmniVoice 使用自定义 **WebDataset** 格式训练：音频打包为 **tar 分片**，并配有 **JSONL 元数据**。每个 tar 含数百至数千条样本（`.npy` 音频 token 数组），显著降低训练时磁盘 I/O；分离的 jsonl 便于修改元数据。本文说明数据格式与准备流程。

## 1. 输入格式

准备 JSONL，每行一个 JSON 对象：

```jsonl
{"id": "sample_001", "audio_path": "/data/audio/001.wav", "text": "Hello world", "language_id": "en"}
{"id": "sample_002", "audio_path": "/data/audio/002.wav", "text": "你好世界", "language_id": "zh"}
```

字段说明：
- `id` — 唯一样本 ID（用于跨分片与标签文件对齐）
- `audio_path` — 音频文件绝对路径（wav/flac/mp3，将重采样到 24 kHz）
- `text` — 转写文本
- `language_id` —（可选）语言代码，用于多语言训练，可省略

## 2. 处理

脚本 `extract_audio_tokens.py` 将音频编码为 8 层离散 token 并打包为 WebDataset 分片。

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"  # token 提取使用的 GPU
python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl data.jsonl \
    --tar_output_pattern output/audios/shard-%06d.tar \
    --jsonl_output_pattern output/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True
```

流程：
1. 读取 JSONL 清单
2. 用 audio tokenizer 将每条音频编码为离散 token
3. 打包为 WebDataset tar 分片及配套 jsonl
4. 生成 `data.lst` 清单

<details>
<summary><strong>备选：</strong>已有原始音频 WebDataset 输入</summary>

用 `data.lst` 清单代替 `--input_jsonl`：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,4"
python -m omnivoice.scripts.extract_audio_tokens \
    --input_manifest existing_data/data.lst \
    --tar_output_pattern output/audios/shard-%06d.tar \
    --jsonl_output_pattern output/txts/shard-%06d.jsonl \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 3 \
    --shuffle True
```

`existing_data/data.lst` 可由以下命令生成：

```bash
python -m omnivoice.scripts.jsonl_to_webdataset \
    --input data.jsonl \
    --output data/shards \
    --sr 24000 \
    --shard-size 1000
```

该步骤将音频重采样到目标采样率，并把 FLAC 打包为 tar 分片及配套 jsonl。

</details>

### 脚本参数说明

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--input_manifest` | None | 输入数据集清单 `data.lst`，与 `--input_jsonl` 互斥 |
| `--input_jsonl` | None | 原始 JSONL 路径，与 `--input_manifest` 互斥 |
| `--tar_output_pattern` | （必填） | tar 输出模式，如 `output/audios/shard-%06d.tar` |
| `--jsonl_output_pattern` | （必填） | JSONL 输出模式，如 `output/txts/shard-%06d.jsonl` |
| `--tokenizer_path` | `eustlb/higgs-audio-v2-tokenizer` | Hugging Face 或本地 tokenizer 路径 |
| `--nj_per_gpu` | 3 | 每 GPU 工作进程数 |
| `--loader_workers` | 24 | 流式 `IterableDataset` 的 DataLoader worker 数 |
| `--shuffle` | True | 分片前是否打乱 |
| `--shuffle-seed` | 42 | 打乱随机种子 |
| `--samples_per_shard` | 1000 | 每个 tar 分片最大样本数 |
| `--min_num_shards` | 32 | 最少输出分片数（保证分片数 ≥ GPU 数 × worker 数） |
| `--min_length` | 0.0 | 跳过短于该值（秒）的音频 |
| `--max_length` | inf | 跳过长于该值（秒）的音频 |
| `--skip_errors` | False | 出错时继续而非中止 |
| `--num_machines` | 1 | 分布式预处理机器总数 |
| `--machine_index` | 0 | 机器索引（从 0 起） |

### 输出目录结构

使用如下输出模式：

```bash
--tar_output_pattern output/audios/shard-%06d.tar \
--jsonl_output_pattern output/txts/shard-%06d.jsonl
```

将得到：

```
output/
├── audios/                    # WebDataset tar（音频 token）
│   ├── shard-000000.tar       # 每个 tar 约 1000 条样本
│   ├── shard-000001.tar
│   └── ...
├── txts/                      # 每分片配套的 JSONL 标签
│   ├── shard-000000.jsonl
│   ├── shard-000001.jsonl
│   └── ...
├── data.lst                   # 关联 tar ↔ jsonl 的清单
└── errors.jsonl               # 处理失败的样本（如有）
```

`data.lst` 与 `errors.jsonl` 写在 `audios/`、`txts/` 的**父目录**下。

### `data.lst` 清单格式

每行描述一个分片：

```
/path/to/shard-000000.tar /path/to/shard-000000.jsonl 1000 3600.500
/path/to/shard-000001.tar /path/to/shard-000001.jsonl 800 2880.200
```

格式：`<tar_path> <jsonl_path> <num_samples> <total_duration_seconds>`

- 路径为**绝对路径**
- `.tar` 含音频 token
- `.jsonl` 含元数据，便于不解压 tar 即可改标签
- 训练数据配置引用此清单

### tar 分片内部

每个 `.tar` 将**多条样本**（默认每分片 1000 条）打成一个归档，这是 WebDataset 的核心优势：dataloader 顺序读少量大文件，而非成千上万小文件。

每条样本为同名 key 的文件对：

```
shard-000000.tar:
  sample_001.npy    # 音频 token：numpy，shape [8, T]，dtype int16
  sample_002.npy
  ...
  sample_1000.npy
```

## 3. 训练用数据配置

创建 WebDataset 后，编写引用它们的 JSON 数据配置：

```json
{
    "train": [
        {
            "language_id": "en",
            "manifest_path": ["data/custom/tokens/train/data.lst"],
            "repeat": 1
        }
    ],
    "dev": [
        {
            "language_id": "en",
            "manifest_path": ["data/custom/tokens/dev/data.lst"],
            "repeat": 1
        }
    ]
}
```

- `manifest_path` — `data.lst` 文件列表（每个分片目录一个）
- `repeat` — 每个 epoch 重复次数（可用于语言平衡）
- `language_id` 不参与训练逻辑，仅便于组织数据

可参考 [examples/config/](../examples/config/) 中的数据配置示例。

> 降噪与噪声增强见 [data_preparation_advanced.md](data_preparation_advanced.md)。
