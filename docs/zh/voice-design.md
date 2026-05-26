# 声音设计（Voice Design）

> 英文版：[voice-design.md](../voice-design.md)

声音设计模式通过说话人属性（`instruct` 参数）描述目标音色，**无需参考音频**，模型会即时生成匹配的声音。

## 快速示例

```python
import torch
from omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

audio = model.generate(
    text="This is a test for voice design.",
    instruct="female, young adult, high pitch, british accent",
)
```

## 工作原理

`instruct` 接受逗号分隔的说话人属性字符串。每个属性属于一个**类别**（性别、年龄、音高、风格、口音或方言）。**同一类别内只能选一个属性**；不同类别可自由组合。

模型会自动检测 instruct 文本语言并在内部归一化——可用英文、中文或中英混合书写。

## 支持的属性

### 性别（Gender）

| English | 中文 |
|---------|---------|
| male | 男 |
| female | 女 |

### 年龄（Age）

| English | 中文 |
|---------|---------|
| child | 儿童 |
| teenager | 少年 |
| young adult | 青年 |
| middle-aged | 中年 |
| elderly | 老年 |

### 音高（Pitch）

| English | 中文 |
|---------|---------|
| very low pitch | 极低音调 |
| low pitch | 低音调 |
| moderate pitch | 中音调 |
| high pitch | 高音调 |
| very high pitch | 极高音调 |

### 风格（Style）

| English | 中文 |
|---------|---------|
| whisper | 耳语 |

### 英语口音（English Accent）

仅当合成文本为英语时生效。

| Accent |
|--------|
| american accent |
| british accent |
| australian accent |
| canadian accent |
| indian accent |
| chinese accent |
| korean accent |
| japanese accent |
| portuguese accent |
| russian accent |

### 中文方言（Chinese Dialect）

仅当合成文本为中文时生效。

| 方言 |
|---------|
| 河南话 |
| 陕西话 |
| 四川话 |
| 贵州话 |
| 云南话 |
| 桂林话 |
| 济南话 |
| 石家庄话 |
| 甘肃话 |
| 宁夏话 |
| 青岛话 |
| 东北话 |

## 编写 instruct 字符串

属性之间用逗号分隔（英文用半角 `,`，中文可用全角 `，`——模型会自动修正不一致）。

```
# 英文
"female, young adult, high pitch, british accent"

# 中文
"女，青年，高音调，四川话"

# 混合（会自动归一化）
"female, young adult, 四川话"
```

### 技巧

- **跨类别自由组合**：例如 `"male, elderly, low pitch, whisper"`。
- **可省略不关心项**：只写 `"female"` 也有效，其余由模型补全。
- **大小写不敏感**：`"Male"`、`"MALE"`、`"male"` 均可，代码会归一化为小写。
- **口音 vs 方言**：英语口音仅作用于英语合成；中文方言仅作用于中文合成。
- **组合限制**：受训练数据影响，部分属性组合可能效果不佳，模型可能忽略组合中的某些项；若不符合预期，可简化 instruct 字符串。
