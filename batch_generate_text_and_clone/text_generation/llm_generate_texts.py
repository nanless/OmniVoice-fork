#!/usr/bin/env python3
"""
OmniVoice Children Speech Text Generation v3
Major improvements over v2:
- Precise tag positioning: each tag has explicit placement rules
- More natural child speech: self-repairs, overlaps, prosodic patterns
- Post-generation validation: auto-correct tag density and position
- Scenario-aware tag patterns: contextual tag usage per scene type
- Better emotional nuance: tags reflect subtle emotion transitions
"""

import json
import os
import sys
import hashlib
import random
import difflib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEXT_GENERATION_DIR = Path(__file__).resolve().parent
_ENV_FILE = _TEXT_GENERATION_DIR / ".env"
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "batch_generated_text")


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not override existing vars)."""
    path = env_path or _ENV_FILE
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenConfig:
    total_target: int = 100000
    batch_size: int = 8
    max_workers: int = 10
    generate_text_tn: bool = True
    output_dir: str = _DEFAULT_OUTPUT_DIR
    seed: int = 42
    same_context_dup_threshold: float = 0.52
    semantic_dedup_threshold: float = 0.88
    reject_severe_length_mismatch: bool = True
    suppression_window_size: int = 800

    # Age distribution
    age_distribution: Dict[str, float] = field(default_factory=lambda: {
        "preschool":     0.20,
        "early_elem":    0.40,
        "upper_elem":    0.40,
    })

    # Length distribution (increased short/ultra_short for more concise child expressions)
    length_distribution: Dict[str, float] = field(default_factory=lambda: {
        "ultra_short": 0.20,
        "short":       0.30,
        "medium":      0.25,
        "long":        0.18,
        "very_long":   0.07,
    })

    # Language mix
    lang_mix_distribution: Dict[str, float] = field(default_factory=lambda: {
        "pure_cn":      0.28,
        "pure_en":      0.18,
        "cn_mostly":    0.24,
        "en_mostly":    0.10,
        "frequent_mix": 0.20,
    })

    stress_test_ratio: float = 0.15
    scenario_distribution: Dict[str, float] = field(default_factory=lambda: {
        "daily_family": 0.7,
        "school": 1.0,
        "classroom_interaction": 1.2,
        "emotions": 0.7,
        "social": 0.8,
        "play": 0.6,
        "learning": 1.3,
        "poetry_classics": 2.0,
        "primary_chinese": 3.2,
        "primary_english": 3.0,
        "primary_math": 3.2,
        "primary_science": 0.9,
        "morality_life": 0.8,
        "arts_music": 0.8,
        "pe_health": 0.8,
        "info_labor": 0.8,
        "knowledge_chat": 1.1,
        "fantasy": 0.8,
        "body_senses": 0.8,
        "requests": 0.8,
    })

    # LLM API (defaults from batch_generate_text_and_clone/text_generation/.env)
    model: str = ""
    api_key: Optional[str] = None
    base_url: str = ""
    max_retries: int = 3
    retry_base_delay: float = 1.0
    max_tokens: int = 8192
    temperature: float = 0.85
    truncate_overlength: bool = False


# ---------------------------------------------------------------------------
# OmniVoice Tag Definitions with Placement Rules
# ---------------------------------------------------------------------------

TAG_DEFINITIONS = {
    "[laughter]": {
        "description": "笑声、咯咯笑",
        "placement": "句尾或逗号后，表示听到/说完后笑。绝不在句首。",
        "examples": [
            "妈妈你快看，那只小狗在追自己的尾巴[laughter]",
            "我说我不是故意的，你信吗[laughter]",
        ],
        "forbidden": "不要在严肃、生气、哭泣场景使用。",
    },
    "[sigh]": {
        "description": "叹气、失望、疲倦",
        "placement": "句首表示情绪基调，或句中表示无奈。",
        "examples": [
            "[sigh]又要写作业了",
            "我明明都道歉了[sigh]她还是不理我",
            "[sigh]今天真的好累啊",
        ],
        "forbidden": "不要在兴奋、开心场景使用。",
    },
    "[question-ah]": {
        "description": '"啊？"表示没听清、困惑',
        "placement": "单独作为反应词，或在句首/逗号后。",
        "examples": [
            "[question-ah]你说什么？",
            "明天要考试[question-ah]我怎么不知道",
        ],
        "forbidden": "不要在陈述肯定事实时使用。",
    },
    "[question-oh]": {
        "description": '"哦？"表示恍然大悟、好奇',
        "placement": "听到新信息后的反应，通常单独或在逗号后。",
        "examples": [
            "[question-oh]原来是这样啊",
            "你是说[question-oh]那个红色的玩具吗",
        ],
        "forbidden": "不要和[surprise-oh]混淆，[question-oh]偏疑问。",
    },
    "[question-ei]": {
        "description": '"诶？"表示轻微惊讶、发现',
        "placement": "发现异常时的自然反应，短促。",
        "examples": [
            "[question-ei]我的橡皮怎么不见了",
            "[question-ei]这不是我的书包吗",
        ],
        "forbidden": "不要在平静叙述中使用。",
    },
    "[question-yi]": {
        "description": '"咦？"表示发现奇怪的事',
        "placement": "发现不合常理之物的反应。",
        "examples": [
            "[question-yi]这里怎么有个洞",
            "[question-yi]小猫怎么会开门的",
        ],
        "forbidden": "不要在日常普通场景使用。",
    },
    "[surprise-ah]": {
        "description": '"啊！"突然惊吓',
        "placement": "突发事件的第一反应，短促有力。",
        "examples": [
            "[surprise-ah]有虫子！",
            "[surprise-ah]你怎么突然出来了",
        ],
        "forbidden": "不要在预期之中的事件时使用。",
    },
    "[surprise-oh]": {
        "description": '"哦！"突然明白',
        "placement": "恍然大悟的瞬间反应。",
        "examples": [
            "[surprise-oh]我知道了！是这样做",
            "[surprise-oh]原来答案是这个",
        ],
        "forbidden": "不要和[question-oh]混淆，[surprise-oh]偏顿悟。",
    },
    "[surprise-wa]": {
        "description": '"哇！"惊叹、赞叹',
        "placement": "看到 impressive 事物时的自然反应。",
        "examples": [
            "[surprise-wa]这个蛋糕好大啊",
            "[surprise-wa]你居然会骑自行车了",
        ],
        "forbidden": "不要在负面场景使用。",
    },
    "[surprise-yo]": {
        "description": '"哟！"轻微惊讶、调侃',
        "placement": "轻松场合的调侃语气，不用于严肃场景。",
        "examples": [
            "[surprise-yo]今天打扮得不错嘛",
            "[surprise-yo]你还会做这个啊",
        ],
        "forbidden": "不要在负面情绪场景使用。",
    },
    "[dissatisfaction-hnn]": {
        "description": '"嗯..."犹豫、不满、思考',
        "placement": "句中表示犹豫，或句首表示不情愿。",
        "examples": [
            "[dissatisfaction-hnn]这个嘛，让我想想",
            "我[dissatisfaction-hnn]不太想去",
        ],
        "forbidden": "不要在果断、开心的场景使用。",
    },
    "[confirmation-en]": {
        "description": "英文语气词 'uh-huh', 'mm-hmm' 等确认",
        "placement": "回应对方时的自然确认声，简短。",
        "examples": [
            "[confirmation-en]嗯嗯",
            "[confirmation-en]好的好的",
        ],
        "forbidden": "不要在提问或否定场景使用。",
    },
}

# Valid tag names for regex
VALID_TAG_NAMES = "|".join(tag.strip("[]") for tag in TAG_DEFINITIONS.keys())

# ---------------------------------------------------------------------------
# Emotion-to-Tag Mapping with Intensity and Position
# ---------------------------------------------------------------------------

EMOTION_PROFILES = {
    # Happy emotions
    "excited": {
        "primary_tags": ["[surprise-wa]", "[laughter]"],
        "secondary_tags": ["[surprise-yo]"],
        "tag_density": "high",
        "position_bias": "end_weighted",
    },
    "happy": {
        "primary_tags": ["[laughter]"],
        "secondary_tags": ["[surprise-wa]", "[confirmation-en]"],
        "tag_density": "medium",
        "position_bias": "end_weighted",
    },
    "proud": {
        "primary_tags": ["[surprise-wa]", "[laughter]"],
        "secondary_tags": ["[surprise-yo]"],
        "tag_density": "medium",
        "position_bias": "mixed",
    },
    "playful": {
        "primary_tags": ["[laughter]", "[surprise-yo]"],
        "secondary_tags": ["[surprise-wa]"],
        "tag_density": "high",
        "position_bias": "scattered",
    },
    "mischievous": {
        "primary_tags": ["[laughter]", "[dissatisfaction-hnn]"],
        "secondary_tags": ["[surprise-yo]"],
        "tag_density": "medium",
        "position_bias": "end_weighted",
    },

    # Question emotions
    "curious": {
        "primary_tags": ["[question-oh]", "[question-ei]"],
        "secondary_tags": ["[question-yi]"],
        "tag_density": "high",
        "position_bias": "start_weighted",
    },
    "confused": {
        "primary_tags": ["[question-ah]", "[question-yi]"],
        "secondary_tags": ["[question-ei]"],
        "tag_density": "medium",
        "position_bias": "start_weighted",
    },
    "surprised": {
        "primary_tags": ["[surprise-ah]", "[surprise-oh]"],
        "secondary_tags": ["[surprise-wa]"],
        "tag_density": "high",
        "position_bias": "start_weighted",
    },
    "shocked": {
        "primary_tags": ["[surprise-ah]", "[surprise-wa]"],
        "secondary_tags": ["[question-ah]"],
        "tag_density": "very_high",
        "position_bias": "start_weighted",
    },

    # Negative emotions
    "angry": {
        "primary_tags": ["[sigh]", "[dissatisfaction-hnn]"],
        "secondary_tags": ["[surprise-ah]"],
        "tag_density": "medium",
        "position_bias": "start_weighted",
    },
    "frustrated": {
        "primary_tags": ["[sigh]", "[dissatisfaction-hnn]"],
        "secondary_tags": ["[question-ah]"],
        "tag_density": "medium",
        "position_bias": "mixed",
    },
    "jealous": {
        "primary_tags": ["[dissatisfaction-hnn]"],
        "secondary_tags": ["[sigh]"],
        "tag_density": "low",
        "position_bias": "end_weighted",
    },
    "sad": {
        "primary_tags": ["[sigh]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "low",
        "position_bias": "start_weighted",
    },
    "disappointed": {
        "primary_tags": ["[sigh]", "[dissatisfaction-hnn]"],
        "secondary_tags": [],
        "tag_density": "medium",
        "position_bias": "start_weighted",
    },
    "lonely": {
        "primary_tags": ["[sigh]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "low",
        "position_bias": "start_weighted",
    },

    # Anxious emotions
    "scared": {
        "primary_tags": ["[surprise-ah]"],
        "secondary_tags": ["[question-ah]"],
        "tag_density": "high",
        "position_bias": "start_weighted",
    },
    "nervous": {
        "primary_tags": ["[question-ah]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "medium",
        "position_bias": "scattered",
    },
    "worried": {
        "primary_tags": ["[question-ei]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "medium",
        "position_bias": "mixed",
    },

    # Low energy
    "bored": {
        "primary_tags": ["[sigh]", "[dissatisfaction-hnn]"],
        "secondary_tags": [],
        "tag_density": "low",
        "position_bias": "start_weighted",
    },
    "tired": {
        "primary_tags": ["[sigh]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "low",
        "position_bias": "start_weighted",
    },
    "sleepy": {
        "primary_tags": ["[sigh]"],
        "secondary_tags": ["[dissatisfaction-hnn]"],
        "tag_density": "low",
        "position_bias": "start_weighted",
    },

    # Silly
    "silly": {
        "primary_tags": ["[laughter]", "[surprise-yo]"],
        "secondary_tags": ["[surprise-wa]"],
        "tag_density": "medium",
        "position_bias": "scattered",
    },
}

EMOTIONS = list(EMOTION_PROFILES.keys())

TAG_DENSITY_MAP = {
    "very_high": (2, 4),
    "high": (1, 3),
    "medium": (0, 2),
    "low": (0, 1),
}

# ---------------------------------------------------------------------------
# Age-Tiered Language Features v3
# ---------------------------------------------------------------------------

AGE_FEATURES = {
    "preschool": {
        "name": "幼儿园（3-5岁）",
        "cn_features": [
            "大量使用叠词：吃饭饭、睡觉觉、喝水水、抱抱、亲亲",
            "句子极短，2-6个字为主，经常只是一个词",
            "大量使用语气词：嘛、呀、啦、呢、吧、哦、哼",
            "重复同一个词3次以上表达强烈情绪：要要要、不不不、好好好、抱抱抱",
            "用'最'字表达极端：最最喜欢、最最讨厌",
            "第三人称自我指代：宝宝要、明明不吃、亮亮害怕",
            "发音不准模拟：七饭(吃饭)、回水(喝水)、脑斧(老虎)",
            "短暂停顿后把话说完整，可以换话题但不能留下半句",
            "不会用复杂连词，只用'然后'、'还有'",
        ],
        "en_features": [
            "Very short phrases, 1-4 words",
            "Triple repetitions: 'want want want', 'no no no', 'mine mine mine'",
            "Baby talk: 'nana', 'baba', 'wawa'",
            "Subject + object only, no verbs sometimes: 'Me toy!', 'More juice!'",
            "Cannot form complex sentences, use 'and then' for everything",
        ],
        "mix_patterns": [
            "中文句子中插入英文名词：'我要吃apple'、'给我water'",
            "用英文数字：'我有three个'、'I want two'",
            "简短英文感叹词：'哇，so big!'、'Look，妈妈！'",
        ],
    },
    "early_elem": {
        "name": "小学低年级（6-8岁）",
        "cn_features": [
            "会说完整句子但经常跑题，说一半突然想到别的",
            "使用夸张词：超级好吃、巨好玩、超厉害",
            "模仿大人但用词不当：'我今天很忙碌'(想说我今天很忙)",
            "连环追问：'为什么...为什么...为什么...'",
            "'然后'连珠炮：'然后...然后...然后...'",
            "突然 shout 单个词打断句子",
            "自我修复：'我-我-我想要，不对，我不想要了'",
            "句子重叠：'那个那个'、'就是就是'",
        ],
        "en_features": [
            "Basic sentences with grammar errors",
            "Exaggerated words: 'sooooo', 'super duper', 'mega'",
            "'And then... and then...' storytelling",
            "Many 'why' questions in a row",
            "Self-repair with complete meaning: 'I-I-I want it, no, I don't want it now'",
            "False starts that still finish: 'Can we go now, wait, I mean after class?'",
        ],
        "mix_patterns": [
            "学校英文自然混入：'今天playground好好玩'",
            "英文情绪+中文叙述：'I'm so happy今天考了100分'",
            "英文短语+中文解释：'It's unfair不公平'",
            "中英交替叙述同一件事",
        ],
    },
    "upper_elem": {
        "name": "小学高年级（9-12岁）",
        "cn_features": [
            "可以说复杂长句但仍有孩子气，用词稚嫩",
            "网络流行语：'栓Q'、'绝绝子'、'yyds'、'emo了'",
            "模仿视频博主：'家人们谁懂啊'、'一整个大无语'",
            "反问句表达不满：'难道不是吗？'、'我哪里错了？'",
            "悄悄话传八卦的语气",
            "故意拖长音：'好——的——'、'知——道——了——'",
            "讽刺语气模仿大人",
        ],
        "en_features": [
            "Complex sentences with childish slang",
            "Internet slang: 'cringe', 'sus', 'no cap', 'fr fr', 'slay'",
            "Fluent code-switching mid-sentence",
            "Storytelling with proper structure",
            "Conditional: 'if I were...', 'what if...'",
            "Sarcasm: 'Oh, great. Just what I needed.'",
        ],
        "mix_patterns": [
            "流畅中英切换：'我觉得这个idea超棒的，totally agree'",
            "中文叙述+英文评价：'他这样做真的很过分，so annoying'",
            "英文开头中文展开：'You know what? 我今天发现一件超有趣的事'",
            "中英混合表达复杂情绪",
        ],
    },
}


# ---------------------------------------------------------------------------
# Scenarios v3
# ---------------------------------------------------------------------------

SCENARIOS = {
    "daily_family": {
        "name": "家庭日常",
        "subscenes": [
            "吃饭挑食讨价还价", "起床赖床撒娇", "洗澡玩水 refusal",
            "看电视抢遥控器", "要零食要玩具", "被批评顶嘴",
            "family出游兴奋", "睡前故事请求", "生病不舒服",
            "帮厨打鸡蛋", "宠物洗澡抗拒", "客人来访害羞",
            "收玩具讨价还价", "换牙害怕", "雨天室内闷",
        ],
        "typical_tags": ["[laughter]", "[sigh]", "[dissatisfaction-hnn]", "[surprise-wa]"],
        "typical_emotions": {
            "happy": 0.15, "excited": 0.12, "playful": 0.10, "mischievous": 0.10,
            "sad": 0.08, "angry": 0.08, "frustrated": 0.08, "disappointed": 0.08,
            "bored": 0.06, "tired": 0.05, "sleepy": 0.05, "worried": 0.05,
        },
    },
    "school": {
        "name": "学校生活",
        "subscenes": [
            "上课走神被发现", "举手回答错误", "考试紧张",
            "被老师表扬害羞", "传纸条被抓", "体育课偷懒",
            "食堂挑食", "放学冲出门", "作业忘带借口",
            "社团课迟到", "科学展讲解", "暴雨停课",
            "换座位不适应", "眼保健操偷看窗外",
        ],
        "typical_tags": ["[question-ah]", "[sigh]", "[surprise-ah]", "[dissatisfaction-hnn]"],
        "typical_emotions": {
            "nervous": 0.15, "worried": 0.12, "frustrated": 0.10, "confused": 0.10,
            "proud": 0.10, "happy": 0.08, "surprised": 0.08, "bored": 0.08,
            "scared": 0.05, "disappointed": 0.08, "silly": 0.06,
        },
    },
    "classroom_interaction": {
        "name": "师生课堂互动",
        "subscenes": [
            "老师点名回答", "同桌小声提醒", "小组讨论抢话",
            "黑板前讲题", "课堂纪律提醒", "举手但忘了要说什么",
            "老师追问为什么", "课代表收作业", "课堂小游戏",
            "下课前总结", "被老师纠正读音", "同学互评答案",
        ],
        "typical_tags": ["[question-ah]", "[question-oh]", "[sigh]", "[surprise-oh]", "[confirmation-en]"],
        "typical_emotions": {
            "nervous": 0.16, "curious": 0.14, "confused": 0.14, "proud": 0.12,
            "happy": 0.10, "worried": 0.10, "surprised": 0.08, "frustrated": 0.08,
            "bored": 0.04, "playful": 0.04,
        },
    },
    "emotions": {
        "name": "情绪爆发",
        "subscenes": [
            "开心到跳起来", "生气摔东西", "害怕不敢睡觉",
            "委屈哭泣", "惊讶大叫", "无聊发牢骚",
            "嫉妒兄弟姐妹", "自豪展示奖状", "恶作剧得逞偷笑",
        ],
        "typical_tags": ["[laughter]", "[sigh]", "[surprise-ah]", "[surprise-wa]", "[dissatisfaction-hnn]"],
        "typical_emotions": {
            "excited": 0.15, "happy": 0.12, "angry": 0.10, "scared": 0.10,
            "surprised": 0.10, "shocked": 0.08, "bored": 0.08, "jealous": 0.08,
            "proud": 0.08, "mischievous": 0.06, "playful": 0.05,
        },
    },
    "social": {
        "name": "社交互动",
        "subscenes": [
            "交朋友害羞", "抢玩具冲突", "分享不情愿",
            "说别人坏话", "秘密约定", "攀比炫耀",
            "被孤立难过", "小团体排挤", "道歉勉强",
        ],
        "typical_tags": ["[dissatisfaction-hnn]", "[sigh]", "[surprise-yo]", "[laughter]"],
        "typical_emotions": {
            "lonely": 0.12, "jealous": 0.12, "angry": 0.10, "frustrated": 0.10,
            "sad": 0.10, "nervous": 0.10, "happy": 0.08, "playful": 0.08,
            "mischievous": 0.08, "disappointed": 0.08, "bored": 0.04,
        },
    },
    "play": {
        "name": "游戏娱乐",
        "subscenes": [
            "电子游戏上瘾", "户外探险", "角色扮演入戏",
            "输掉游戏生气", "搭建积木倒塌", "捉迷藏被发现",
            "卡牌游戏作弊", "新玩具兴奋", "游戏规则争论",
        ],
        "typical_tags": ["[surprise-wa]", "[laughter]", "[surprise-ah]", "[sigh]"],
        "typical_emotions": {
            "excited": 0.18, "happy": 0.15, "playful": 0.12, "surprised": 0.10,
            "angry": 0.10, "frustrated": 0.08, "mischievous": 0.08, "proud": 0.08,
            "disappointed": 0.06, "bored": 0.05,
        },
    },
    "learning": {
        "name": "学习认知",
        "subscenes": [
            "背课文卡顿", "大声朗读课文", "读课后题题干",
            "数学题不会做", "英语单词记不住",
            "科学实验好奇", "历史故事入迷", "天文问题追问",
            "攀比成绩", "炫耀知识", "假装懂装懂",
        ],
        "typical_tags": ["[question-oh]", "[question-ei]", "[sigh]", "[surprise-oh]"],
        "typical_emotions": {
            "curious": 0.18, "confused": 0.15, "proud": 0.12, "surprised": 0.10,
            "frustrated": 0.10, "worried": 0.08, "nervous": 0.08, "happy": 0.08,
            "bored": 0.06, "mischievous": 0.05,
        },
    },
    "poetry_classics": {
        "name": "唐诗宋词与古诗文",
        "subscenes": [
            "背唐诗背串行", "宋词意境想象", "诗人名字记混",
            "古诗押韵发现", "解释诗句意思", "课前古诗朗读",
            "飞花令接不上", "把古诗编成故事", "古文断句困惑",
            "诗里景物联想",
        ],
        "typical_tags": ["[question-oh]", "[question-ei]", "[surprise-oh]", "[sigh]"],
        "typical_emotions": {
            "curious": 0.18, "confused": 0.16, "proud": 0.14, "happy": 0.12,
            "frustrated": 0.10, "surprised": 0.10, "bored": 0.08, "nervous": 0.07,
            "playful": 0.05,
        },
    },
    "primary_chinese": {
        "name": "小学语文",
        "subscenes": [
            "拔高-朗读文言文短篇", "拔高-跟读说明文科普段", "拔高-读议论文片段",
            "拔高-长篇阅读理解题干", "拔高-朗读七律古诗", "拔高-读成语典故文",
            "拔高-跟读名著节选", "拔高-读新闻评论短评", "拔高-朗读词牌短调",
            "拔高-读同步练习阅读材料", "拔高-跟读默写纸上的句子",
            "指读课文一字一顿", "大声朗读课文段落", "跟读老师纠正读音",
            "跟读录音/点读笔", "读阅读理解题干", "朗读古诗课文",
            "读课后题问题", "读注释和词语解释", "读看图说话提示语",
            "早读课集体朗读", "小组轮流读一段", "过关背诵检查",
            "拼音声调读错", "组词造句", "作文开头想不出",
        ],
        "typical_tags": ["[question-ah]", "[question-oh]", "[sigh]", "[surprise-oh]"],
        "typical_emotions": {
            "curious": 0.16, "confused": 0.16, "frustrated": 0.12, "proud": 0.12,
            "nervous": 0.10, "happy": 0.10, "worried": 0.08, "bored": 0.08,
            "surprised": 0.08,
        },
    },
    "primary_english": {
        "name": "小学英语",
        "subscenes": [
            "拔高-跟读长对话听力稿", "拔高-读完形填空长段落",
            "拔高-读定语从句复合句", "拔高-朗读科普英语短文",
            "拔高-读被动语态例句", "拔高-跟读演讲稿节选",
            "拔高-读历史类阅读材料", "拔高-跟读访谈对话脚本",
            "拔高-读图表说明英语短文", "拔高-朗读辩论正方观点",
            "跟读英语课文Repeat after me", "读课本对话Dialog",
            "读考试题目Read the question", "读听力材料跟读",
            "指读绘本Page by page", "读黑板板书句子",
            "读Word list并造句", "角色扮演跟读对话",
            "跟读外教示范句", "读试卷阅读Part短文",
            "逐句读课文漏词", "自然拼读卡住", "单词听写",
        ],
        "typical_tags": ["[question-ah]", "[question-oh]", "[surprise-oh]", "[laughter]"],
        "typical_emotions": {
            "nervous": 0.15, "confused": 0.15, "curious": 0.14, "happy": 0.12,
            "proud": 0.12, "frustrated": 0.10, "playful": 0.08, "surprised": 0.08,
            "worried": 0.06,
        },
    },
    "primary_math": {
        "name": "小学数学",
        "subscenes": [
            "拔高-读一元一次方程应用题", "拔高-读二元一次方程组题",
            "拔高-读一次函数图像题", "拔高-读勾股定理几何题",
            "拔高-读相似三角形题", "拔高-读概率统计题",
            "拔高-读不等式应用题", "拔高-读圆的周长面积题",
            "拔高-读百分数复杂应用题", "拔高-读坐标系描点题",
            "拔高-读统计图表题", "拔高-读几何证明已知条件",
            "朗读应用题题目", "读算式一个一个念", "读题目里的关键词",
            "读填空题题干", "读选择题选项", "读竖式计算步骤 aloud",
            "口算题逐题念出来", "读图形题条件", "读错题本上的题目",
            "同桌互读题目", "应用题读不懂",
        ],
        "typical_tags": ["[question-ah]", "[question-ei]", "[sigh]", "[surprise-oh]"],
        "typical_emotions": {
            "confused": 0.18, "frustrated": 0.14, "curious": 0.14, "proud": 0.12,
            "nervous": 0.10, "worried": 0.10, "happy": 0.08, "surprised": 0.08,
            "bored": 0.06,
        },
    },
    "primary_science": {
        "name": "小学科学",
        "subscenes": [
            "观察植物发芽", "磁铁吸不吸", "水会不会蒸发",
            "影子为什么变长", "月亮形状变化", "昆虫身体结构",
            "声音从哪里来", "简单电路灯泡不亮", "天气记录",
            "浮沉实验", "食物链", "人体呼吸心跳",
        ],
        "typical_tags": ["[question-oh]", "[question-ei]", "[surprise-wa]", "[surprise-oh]"],
        "typical_emotions": {
            "curious": 0.24, "surprised": 0.14, "confused": 0.14, "excited": 0.12,
            "proud": 0.10, "happy": 0.08, "worried": 0.06, "frustrated": 0.06,
            "silly": 0.06,
        },
    },
    "morality_life": {
        "name": "道法班会与生活常识",
        "subscenes": [
            "排队规则", "诚实承认错误", "帮助同学", "交通安全",
            "节约用水", "垃圾分类", "班干部发言", "国旗下讲话",
            "和朋友道歉", "借东西要归还", "校园安全提醒", "劳动值日分工",
        ],
        "typical_tags": ["[question-oh]", "[sigh]", "[dissatisfaction-hnn]", "[confirmation-en]"],
        "typical_emotions": {
            "worried": 0.14, "proud": 0.12, "confused": 0.12, "happy": 0.10,
            "nervous": 0.10, "frustrated": 0.10, "curious": 0.10, "disappointed": 0.08,
            "playful": 0.07, "sad": 0.07,
        },
    },
    "arts_music": {
        "name": "音乐美术",
        "subscenes": [
            "唱歌跑调", "节奏拍错", "认乐器声音", "合唱排练",
            "画画颜色调不出来", "手工剪纸", "泥塑做歪", "美术作品展示",
            "音乐课表演紧张", "画国旗和校园", "给画起名字",
        ],
        "typical_tags": ["[laughter]", "[surprise-wa]", "[question-oh]", "[sigh]"],
        "typical_emotions": {
            "happy": 0.16, "playful": 0.14, "proud": 0.12, "nervous": 0.10,
            "confused": 0.10, "frustrated": 0.10, "excited": 0.10, "surprised": 0.08,
            "silly": 0.08, "bored": 0.02,
        },
    },
    "pe_health": {
        "name": "体育与健康",
        "subscenes": [
            "跳绳计数", "跑步接力", "篮球传球", "足球射门",
            "立定跳远", "排队做操", "体育课热身", "健康课刷牙洗手",
            "运动后喘气", "扭到脚报告老师", "比赛输了不服气",
        ],
        "typical_tags": ["[surprise-wa]", "[sigh]", "[laughter]", "[dissatisfaction-hnn]"],
        "typical_emotions": {
            "excited": 0.16, "proud": 0.14, "frustrated": 0.12, "tired": 0.10,
            "happy": 0.10, "nervous": 0.10, "angry": 0.08, "surprised": 0.08,
            "worried": 0.06, "playful": 0.06,
        },
    },
    "info_labor": {
        "name": "信息科技与劳动实践",
        "subscenes": [
            "电脑打字找不到键", "保存文件忘记名字", "机器人小车不动",
            "画流程图", "认识键盘鼠标", "整理书包和课桌",
            "种绿豆记录", "做纸桥承重", "清洁值日", "缝扣子体验",
            "做简单早餐", "班级植物角浇水",
        ],
        "typical_tags": ["[question-ah]", "[question-ei]", "[surprise-oh]", "[sigh]"],
        "typical_emotions": {
            "curious": 0.18, "confused": 0.16, "proud": 0.12, "frustrated": 0.12,
            "happy": 0.10, "worried": 0.08, "surprised": 0.08, "nervous": 0.08,
            "playful": 0.08,
        },
    },
    "knowledge_chat": {
        "name": "儿童知识闲聊",
        "subscenes": [
            "恐龙和动物", "宇宙星球", "天气和四季",
            "身体为什么会这样", "交通工具", "节日习俗",
            "历史人物小故事", "植物怎么长大", "厨房里的科学",
            "生活安全常识", "地图和城市",
        ],
        "typical_tags": ["[question-oh]", "[question-ei]", "[surprise-wa]", "[surprise-oh]"],
        "typical_emotions": {
            "curious": 0.24, "surprised": 0.14, "happy": 0.12, "proud": 0.10,
            "confused": 0.10, "excited": 0.10, "playful": 0.08, "worried": 0.06,
            "silly": 0.06,
        },
    },
    "fantasy": {
        "name": "想象幻想",
        "subscenes": [
            "超级英雄幻想", "魔法咒语", "太空冒险",
            "恐龙世界", "童话改编", "拥有超能力",
            "冒险故事编造", "梦境描述", "假想朋友对话",
        ],
        "typical_tags": ["[surprise-wa]", "[laughter]", "[surprise-yo]"],
        "typical_emotions": {
            "excited": 0.20, "happy": 0.18, "playful": 0.15, "surprised": 0.12,
            "curious": 0.12, "proud": 0.08, "silly": 0.08, "mischievous": 0.07,
        },
    },
    "body_senses": {
        "name": "身体感知",
        "subscenes": [
            "饥饿要吃饭", "吃太饱打嗝", "发烧难受",
            "受伤哭喊", "犯困打哈欠", "痒痒大笑",
            "闻到臭味", "听到巨响吓哭", "晕车想吐",
        ],
        "typical_tags": ["[sigh]", "[surprise-ah]", "[dissatisfaction-hnn]"],
        "typical_emotions": {
            "tired": 0.15, "scared": 0.15, "sad": 0.12, "surprised": 0.12,
            "frustrated": 0.10, "disappointed": 0.10, "sleepy": 0.10,
            "happy": 0.08, "bored": 0.08,
        },
    },
    "requests": {
        "name": "请求追问",
        "subscenes": [
            "十万个为什么", "反复求同意", "讨价还价",
            "撒娇请求", "威胁反抗", "紧急求助",
            "礼貌请求", "假装可怜", "得寸进尺",
        ],
        "typical_tags": ["[dissatisfaction-hnn]", "[question-oh]", "[sigh]", "[confirmation-en]"],
        "typical_emotions": {
            "curious": 0.18, "frustrated": 0.15, "nervous": 0.12, "worried": 0.12,
            "happy": 0.10, "angry": 0.10, "disappointed": 0.10, "sad": 0.08,
            "playful": 0.05,
        },
    },
    # Stress tests
    "stress_numbers": {
        "name": "【压力测试】数字序列",
        "subscenes": ["电话号码", "门牌号", "日期时间", "数数字", "数学算式", "价格金额"],
        "is_stress_test": True,
        "typical_tags": ["[question-ei]", "[surprise-wa]"],
        "typical_emotions": {
            "curious": 0.25, "surprised": 0.20, "proud": 0.15, "confused": 0.15,
            "happy": 0.15, "excited": 0.10,
        },
    },
    "stress_tonguetwister": {
        "name": "【压力测试】绕口令",
        "subscenes": ["中文绕口令", "英文绕口令", "中英混合绕口令"],
        "is_stress_test": True,
        "typical_tags": ["[laughter]", "[surprise-yo]"],
        "typical_emotions": {
            "playful": 0.30, "happy": 0.25, "silly": 0.25, "excited": 0.20,
        },
    },
    "stress_emotion_shift": {
        "name": "【压力测试】情绪突变",
        "subscenes": ["开心转生气", "害怕转惊喜", "难过转开心", "生气转撒娇", "无聊转兴奋"],
        "is_stress_test": True,
        "typical_tags": ["[laughter]", "[sigh]", "[surprise-ah]", "[surprise-wa]"],
        "typical_emotions": {
            "surprised": 0.25, "shocked": 0.20, "happy": 0.20, "angry": 0.15,
            "sad": 0.10, "excited": 0.10,
        },
    },
    "stress_repetition": {
        "name": "【压力测试】重复与口吃",
        "subscenes": ["紧张口吃", "兴奋重复", "撒娇重复", "结巴解释", "快速连珠炮"],
        "is_stress_test": True,
        "typical_tags": ["[question-ah]", "[dissatisfaction-hnn]"],
        "typical_emotions": {
            "nervous": 0.30, "excited": 0.25, "frustrated": 0.20, "worried": 0.15,
            "happy": 0.10,
        },
    },
    "stress_whisper_shout": {
        "name": "【压力测试】耳语与大喊",
        "subscenes": ["悄悄话秘密", "大喊大叫", "远处呼喊", "突然惊吓", "压低声音威胁"],
        "is_stress_test": True,
        "typical_tags": ["[surprise-ah]", "[surprise-wa]", "[question-ah]"],
        "typical_emotions": {
            "scared": 0.25, "surprised": 0.20, "shocked": 0.15, "nervous": 0.15,
            "angry": 0.15, "excited": 0.10,
        },
    },
}


# ---------------------------------------------------------------------------
# Length & Language Specs
# ---------------------------------------------------------------------------

LENGTH_SPECS = {
    "ultra_short": {
        "cn": "极短，1到5个汉字，或1到3个英文单词。像刚学说话的孩子，一个词或一个短句即可",
        "en": "Very short, 1 to 3 words. Like a toddler learning to speak, just a word or short phrase",
        "examples": ["妈妈！", "Yay!", "不要！", "No way!", "哇！", "Oh no!"],
    },
    "short": {
        "cn": "短句，4到8个汉字/英文单词。简单直接",
        "en": "Short phrase, 4 to 8 words. Simple and direct",
        "examples": ["我要吃糖！", "Look at that!", "今天好开心呀", "I want it now!"],
    },
    "medium": {
        "cn": "中等，9到15个汉字/英文单词。一句完整但不复杂的话",
        "en": "Medium, 9 to 15 words. One complete but not complex sentence",
        "examples": [
            "妈妈，我今天在幼儿园得了小红花！",
            "Can we go to the park after school?",
        ],
    },
    "long": {
        "cn": "较长，16到25个汉字/英文单词。可以包含一个小转折",
        "en": "Longer, 16 to 25 words. Can include a small twist",
        "examples": [
            "小明说他有一个秘密基地，在小区的大树下，我可以去看看吗？",
            "My teacher said I did a great job on my drawing and put it on the wall!",
        ],
    },
    "very_long": {
        "cn": "长句，26到40个汉字/英文单词。一小段连续的表达，像孩子在讲故事",
        "en": "Long, 26 to 40 words. A short continuous expression, like a child telling a story",
        "examples": [
            "昨天晚上我做了一个梦，梦见我变成了奥特曼，正在和一只超级大的怪兽打架，然后妈妈叫我起床我就醒了。",
            "I was building the tallest Lego tower ever and it was even taller than my dad but then my little brother knocked it down and I was so mad!",
        ],
    },
}

LANG_SPECS = {
    "pure_cn": "纯中文，不要出现任何英文字母。如果有外来词概念，用中文表达",
    "pure_en": "Pure English, no Chinese characters at all",
    "cn_mostly": "以中文为主（80%+），自然地混入1-3个常见英文单词（OK、yeah、no、wow、bye、baby、apple等），英文词应该是孩子日常会说的",
    "en_mostly": "Mostly English (80%+), naturally mix in 1-3 common Chinese words a bilingual child would use (妈妈, 爸爸, 谢谢, 再见)",
    "frequent_mix": "自然的中英混杂(code-switching)，像从小双语环境长大的孩子一样说话。中英文切换流畅，可以在句子中间切换，不要刻意",
}


# ---------------------------------------------------------------------------
# Diversity Axes
# ---------------------------------------------------------------------------

MICRO_CONTEXTS = {
    "daily_family": [
        "具体物品: 遥控器、绘本、恐龙睡衣、小碗、牙刷、拖鞋、被子角",
        "说话对象: 妈妈、爸爸、奶奶、哥哥、妹妹、家里的小狗",
        "阻碍原因: 太晚了、饭太烫、玩具没收、灯关了、水进眼睛、零食被藏起来",
        "孩子策略: 撒娇、讲条件、假哭、转移话题、搬出另一个大人、自己制定规则",
        "时间锚点: 刚放学、晚饭前、洗澡后、睡前五分钟、周末早上",
        "地点锚点: 客厅、厨房、儿童房、阳台、车上安全座椅",
        "禁止雷同: 不要连续多条都是‘妈妈我要’或同一零食名",
    ],
    "school": [
        "具体物品: 铅笔盒、橡皮、课本、作业本、红领巾、饭卡、跳绳",
        "说话对象: 老师、同桌、前排同学、值日生、体育委员",
        "阻碍原因: 听错题、忘带东西、排队太慢、被点名、答案写歪、饭菜不喜欢",
        "孩子策略: 小声解释、找借口、装没听见、求同桌帮忙、突然举手补充",
        "地点变化: 走廊、操场、食堂、图书馆、专用教室、校门口",
        "禁止雷同: 不要连续多条都是忘带作业或同一科考试",
    ],
    "classroom_interaction": [
        "具体场景: 老师追问、同桌提醒、小组讨论、课代表收作业、黑板讲题、课堂小游戏",
        "说话对象: 班主任、语文老师、数学老师、英语老师、同桌、小组长、全班同学",
        "互动状态: 刚举手就忘词、答案被纠正、被老师表扬、被提醒坐好、抢着补充",
        "孩子策略: 先小声试探、求老师再说一遍、把同桌答案改一下、承认自己听岔、用例子解释",
    ],
    "emotions": [
        "具体触发: 积木倒了、贴纸没了、画被弄脏、糖掉地上、奖状被折、弟弟抢先",
        "说话对象: 妈妈、爸爸、朋友、玩偶、自己、犯错的弟弟妹妹",
        "身体反应: 鼻子酸、手握拳、眼泪快掉、肚子紧、声音变大、脚跺地",
        "孩子策略: 否认、甩锅、夸张控诉、突然后悔、边哭边讲条件",
    ],
    "social": [
        "具体物品: 卡片、贴纸、积木人、小车、发夹、秘密纸条、座位",
        "说话对象: 新朋友、好朋友、抢玩具的人、旁观同学、老师",
        "冲突原因: 轮到谁、谁先拿、有没有分享、谁被邀请、秘密被说出去",
        "孩子策略: 告状、拉同盟、假装不在乎、交换条件、嘴硬道歉",
    ],
    "play": [
        "具体物品: 游戏手柄、树枝宝剑、沙坑城堡、滑梯、球、卡牌、积木塔",
        "说话对象: 队友、对手、裁判一样的大人、假想怪兽、宠物",
        "冲突原因: 输了、规则变了、被发现、玩具坏了、轮不到自己、天快黑",
        "孩子策略: 改规则、重来一局、耍赖、吹牛、把普通东西想象成道具",
        "游戏类型: 桌游、手游、户外追逐、角色扮演、拼图、跳绳比赛",
        "禁止雷同: 不要连续多条都是 Minecraft 或同一游戏规则争论",
    ],
    "learning": [
        "具体物品: 单词卡、算术本、尺子、实验杯、地球仪、错题本、拼音表",
        "说话对象: 老师、家长、同桌、自己、假装懂的朋友",
        "卡住原因: 记混了、少看一行、把符号看错、读音不会、答案太像",
        "孩子策略: 猜答案、编口诀、转移到自己会的知识、问一串为什么",
    ],
    "poetry_classics": [
        "具体内容: 静夜思、春晓、咏鹅、悯农、登鹳雀楼、水调歌头、清明、江雪",
        "说话对象: 语文老师、妈妈、同桌、背诗小组、自己、想象中的诗人",
        "卡住原因: 上下句背反、诗人记混、字音读错、意思理解偏了、押韵听出来但说不清",
        "孩子策略: 把诗句画面化、编动作记诗、用现代话解释、偷看课本、把诗改成小故事",
    ],
    "primary_chinese": [
        "文体: 文言文、现代记叙/说明/议论、古诗、词、名著节选、新闻短评、同步阅读",
        "朗读方式: 跟读、指读、朗读、默读出声、读题干、读选项、读注释",
        "场景: 早读课、课堂跟读、在家写作业、小组过关、录音作业、黑板跟读",
        "卡住: 漏字跳行、断句错、同音字读错、文言字不会、长句喘气、读太快",
        "策略: 回读上一句、指字念、问老师、小声试读、读一半问对不对",
        "禁止雷同: 不要连续多条都读《岳阳楼记》或同一篇现代文",
    ],
    "primary_english": [
        "文体: Dialog、短文、完形、阅读理解、听力稿、演讲/辩论、科普、图表说明",
        "朗读方式: Repeat after me、指读、逐句跟读、读题干、读选项、读板书",
        "场景: 课堂跟读、外教示范、听力跟读、考试朗读、绘本指读、小组对话",
        "卡住: 漏词、连读错、时态念错、长句断不开、数字日期读错",
        "策略: 拆音节、重复短语、跟读最后三个词、读一半问老师",
        "禁止雷同: 不要连续多条都读 apple/cat 或同一段 Dialog",
    ],
    "primary_math": [
        "题型: 应用题、方程、函数、几何证明已知、概率统计、图表、口算、竖式、填空选择",
        "朗读方式: 念完整题干、念条件与设问、念算式、念选项、边指边读",
        "场景: 课堂读题、作业辅导、同桌互读、黑板讲题、口算比赛念题",
        "卡住: 漏单位、倍比关系念乱、符号念错、人物名读混、几何条件漏半句",
        "策略: 读两遍、指关键词、边读边问怎么列式、念到一半推翻重念",
        "禁止雷同: 不要连续多条都是同一类行程问题或同一道算术",
    ],
    "primary_science": [
        "具体内容: 植物发芽、磁铁、蒸发、影子、月相、昆虫、声音、电路、天气、浮沉、食物链",
        "说话对象: 科学老师、小组同学、实验搭档、家长、观察记录本",
        "观察误区: 把现象看反、以为磁铁什么都吸、忘记记录日期、实验步骤跳过、结论说太早",
        "孩子策略: 边看边猜、用生活经验解释、把发现告诉老师、要求再试一次、给实验起外号",
    ],
    "morality_life": [
        "具体内容: 排队、诚实、帮助同学、交通安全、节约用水、垃圾分类、班干部、值日",
        "说话对象: 班主任、道法老师、同桌、值日组长、犯错的同学、自己",
        "冲突原因: 规则和自己想法不一样、想帮忙但做错、怕承认错误、分工不公平、被提醒安全",
        "孩子策略: 讲道理但逻辑跳、先委屈再承认、帮同学找借口、把规则背成口号、问能不能补救",
    ],
    "arts_music": [
        "具体内容: 唱歌、节奏、乐器、合唱、调颜色、剪纸、泥塑、展示作品、起画名",
        "说话对象: 音乐老师、美术老师、同桌、合唱队友、看作品的同学",
        "卡住原因: 跑调、拍子慢半拍、颜色变脏、剪歪了、作品不像、上台紧张",
        "孩子策略: 笑着重来、把错画成新东西、给作品编故事、小声跟唱、请同桌帮忙看",
    ],
    "pe_health": [
        "具体内容: 跳绳、跑步、篮球、足球、跳远、做操、热身、刷牙洗手、运动安全",
        "说话对象: 体育老师、队友、对手、卫生老师、同桌、自己",
        "身体状态: 喘不上气、脚有点疼、手心出汗、跳绳绊住、跑到一半想停、赢了很得意",
        "孩子策略: 重新计数、求再来一次、给自己加油、解释不是故意犯规、边喘边报告老师",
    ],
    "info_labor": [
        "具体内容: 打字、保存文件、机器人小车、流程图、键盘鼠标、整理课桌、种植、纸桥、值日",
        "说话对象: 信息老师、劳动老师、小组同学、课代表、家长、自己的作品",
        "卡住原因: 找不到键、文件名忘了、线接反、小车不动、桌面越收越乱、绿豆没发芽",
        "孩子策略: 先乱试再问、给机器下命令、把步骤念出来、找同桌帮忙、把失败说成实验结果",
    ],
    "knowledge_chat": [
        "具体内容: 恐龙、猫狗昆虫、月亮太阳、下雨打雷、火车飞机、节日、植物、身体、地图",
        "说话对象: 家长、老师、朋友、博物馆讲解员、自己、玩具恐龙",
        "疑问来源: 刚看视频、课外书看到、路上突然发现、吃饭时想到、做实验时弄错",
        "孩子策略: 连环为什么、把知识和自己生活联系起来、用夸张比喻、先说错再自我修正",
    ],
    "fantasy": [
        "具体道具: 披风、魔法棒、纸盒飞船、恐龙蛋、枕头城堡、隐形药水",
        "说话对象: 假想朋友、怪兽、公主、机器人、队友、旁边的大人",
        "幻想规则: 咒语念错、超能力有冷却、怪兽怕饼干、飞船缺电、披风要充电",
        "孩子策略: 现场改设定、一本正经解释、突然出戏、邀请别人扮演角色",
    ],
    "body_senses": [
        "具体感觉: 肚子咕咕、膝盖疼、头晕、脚痒、眼睛酸、嗓子干、耳朵被吵",
        "说话对象: 妈妈、爸爸、医生、老师、朋友、自己的身体部位",
        "触发原因: 吃太快、摔了一下、坐车太久、被挠痒、声音太大、衣服标签扎人",
        "孩子策略: 夸张描述、要求抱抱、讨价还价、把身体拟人化、边笑边躲",
    ],
    "requests": [
        "具体请求: 多玩五分钟、买小蛋糕、再讲一页、带玩具出门、换座位、先不写作业",
        "说话对象: 妈妈、爸爸、老师、哥哥姐姐、售货员、同伴",
        "被拒原因: 太晚、太贵、要排队、规则不允许、已经答应过一次、明天要上学",
        "孩子策略: 礼貌请求、连环为什么、装可怜、讲条件、威胁不理人、反复确认",
    ],
    "stress_numbers": [
        "数字类型: 电话、门牌、楼层、日期、时间、价格、算式、游戏分数",
        "记忆状态: 记反了、漏一位、把两个号码混在一起、越数越乱、突然想起来",
        "说话对象: 老师、妈妈、同学、电话那头的人、自己",
        "孩子策略: 分段念、用手指数、编节奏、反复确认、念错后立刻改",
    ],
    "stress_tonguetwister": [
        "绕口令对象: 石狮子、红鲤鱼、贝壳、兔子、木头、青蛙",
        "卡顿位置: 开头卡住、中间绕晕、最后破功、越说越快、说完笑场",
        "说话对象: 同学、老师、妈妈、镜子里的自己",
        "孩子策略: 慢慢念、赌气重来、把错音说成对的、笑着求放过",
    ],
    "stress_emotion_shift": [
        "转折触发: 看错成绩、发现惊喜、玩具坏了又修好、误会解除、突然停电",
        "前后对比: 开心到生气、害怕到得意、委屈到偷笑、无聊到尖叫",
        "说话对象: 大人、朋友、自己、玩偶",
        "孩子策略: 先否认变化、突然改口、用标签标记转折、结尾还在回味",
    ],
    "stress_repetition": [
        "重复原因: 紧张、兴奋、撒娇、解释不清、怕被骂、急着抢话",
        "重复对象: 我我我、那个那个、不不不、要要要、wait wait、no no no",
        "说话对象: 老师、家长、朋友、自己",
        "孩子策略: 重复后改口、越说越小声、突然提高音量、边重复边动作",
    ],
    "stress_whisper_shout": [
        "音量变化: 先悄悄话后大喊、先大喊后捂嘴、远处喊、贴耳朵说、突然被吓到",
        "具体场景: 藏猫猫、秘密计划、看到虫子、远处叫人、怕被老师听见",
        "说话对象: 同伴、妈妈、老师、怪兽、自己",
        "孩子策略: 压低声音、突然破音、说完马上道歉、用动作代替语言",
    ],
}

CHILD_PROFILES = {
    "preschool": [
        "胆小黏人，说话短，常要抱抱",
        "嘴馋爱撒娇，喜欢把物品说成叠词",
        "动作比语言快，常说一半就换话题",
        "把玩偶当真的，会替玩偶说话",
        "发音不稳，偶尔把吃饭说成七饭、老虎说成脑斧",
        "情绪来得快去得快，刚哭完又好奇",
    ],
    "early_elem": [
        "话很多，喜欢用然后然后串故事",
        "爱找理由，模仿大人但用词不太对",
        "好胜心强，输赢和公平感特别明显",
        "容易紧张，解释时会结巴和自我修复",
        "好奇心强，会连续追问为什么",
        "喜欢夸张表达，超级、巨、最最经常出现",
    ],
    "upper_elem": [
        "开始嘴硬和反问，但仍有孩子气",
        "会用一点网络流行语，但不能像成年人",
        "爱装懂，发现错了会马上找补",
        "在同伴面前要面子，说话带一点表演感",
        "双语切换更自然，会夹英文评价和中文解释",
        "会讲条件和讲道理，但逻辑还会跳",
    ],
}

SPEECH_ERROR_PATTERNS = {
    "pure_cn": [
        "量词偶尔错用或省略",
        "代词指代有点乱，但能听懂",
        "因果关系跳跃，说完才补一句解释",
        "使用叠词或三连重复，但不要每条都用",
        "短句缺主语，像对话中间的一句",
        "偶尔发音不准模拟，如七饭、脑斧、灰机",
    ],
    "pure_en": [
        "child grammar error: tooked, drawed, falled, more better",
        "missing helper verbs or simple word order mistakes",
        "false start with wait/no/actually",
        "short but complete child utterances",
        "NO um/uh/like fillers — use child hesitations like 'wait' or repeat words instead",
        "repetition for emphasis: very very, no no no, wait wait",
    ],
    "cn_mostly": [
        "中文为主，只夹一个自然英文词，不要每条都夹 OK",
        "英文词放在孩子真的会说的位置，如 toy、phone、cake、bad idea",
        "中文句子可以缺主语或突然改口，但必须把意思说完整",
        "叠词和英文名词自然混用，如 blue的、toy车",
        "不要把中英混杂写得像翻译腔",
        "情绪词可以用英文，原因用中文解释",
    ],
    "en_mostly": [
        "English main sentence with one natural Chinese family word",
        "Chinese word should be embedded naturally, not appended mechanically",
        "use child grammar errors sparingly",
        "include one false start or self-repair",
        "avoid polished adult phrasing",
        "let the child sound bilingual, not translated",
    ],
    "frequent_mix": [
        "句中自然切换，不要前半全中文后半全英文",
        "同一个短语里可以混合，如这个idea、super生气、now怎么办",
        "切换点跟情绪或物品相关",
        "避免每条都使用同一组 English words",
        "允许中英语法互相影响一点点",
        "保持儿童口语感，但不要生成半句或省略号结尾",
    ],
}

DIALOGUE_STATES = [
    "这是被大人拒绝后的第二句话，但必须是完整可朗读的一句话",
    "这是抢话插进来的短句，开头可以口语化，但句意必须完整",
    "这是刚刚听错后的反应，先困惑再把正确意思说完整",
    "这是边做动作边说的话，要有动作痕迹且结尾完整",
    "这是已经哭/笑/急了一会儿后的补充，不要像作文但要说完",
    "这是对同伴说的悄悄话或告状，信息可以少但不能是半句",
    "这是突然想到另一件事，允许轻微跑题但要形成完整表达",
    "这是说到一半意识到不对，必须自我修复并收束成完整句",
    "这是隔了几秒才想起的补充，像孩子话没说完又补一句",
    "这是对着窗外/远处的人喊话，信息要具体",
    "这是边哭边解释，情绪词和事实混在一起",
    "这是假装没听见后的坦白，语气要心虚",
    "这是比赛/游戏刚结束的反应，带输赢感",
    "这是发现误会后的澄清，前后信息要不同",
]

OPENING_STYLES = [
    "直接喊人开头",
    "先用填充词/犹豫开头",
    "先描述动作或身体感觉",
    "先提一个问题",
    "先自我修复或说一半改口",
    "先用短促感叹",
    "先模仿大人或同学说话",
    "先讲刚刚发生的事",
    "先报时间地点",
    "先吐槽再解释",
    "先小声嘀咕",
    "先数东西或报分数",
    "先引用动画片台词",
    "先跟宠物/玩具说话",
    "先报天气或温度",
    "先抱怨再求帮忙",
    "先炫耀再被拆穿",
    "先引用同学的话",
    "先报分数或名次",
]

FOCUS_ANGLES = [
    "物品/玩具",
    "家人或老师",
    "同伴互动",
    "身体感受",
    "规则/要求",
    "时间或地点",
    "误会或记错",
    "想象夸张",
    "天气/季节",
    "食物/零食",
    "比赛/名次",
    "秘密/小声话",
    "手机/平板/游戏",
    "宠物/植物",
    "交通工具/出行",
    "节日/生日",
    "衣服/穿搭",
    "作业/考试",
    "邻居/亲戚",
    "动画片角色",
    "钱/零花钱",
]

SPEECH_PATTERNS = [
    "重复词",
    "拖长音",
    "完整自我修复",
    "连环追问",
    "小声抱怨",
    "突然兴奋",
    "讲条件",
    "自言自语",
    "夸张比喻",
    "数数或报时间",
    "模仿大人腔",
    "突然改口否定",
    "边笑边说不清楚",
    "用拟声词",
]

# Syntactic diversity: sentence type, subject, object, verb (per-row plan).
_SENTENCE_TYPE_CN = [
    "陈述句收束", "一般疑问句(吗/呢)", "特殊疑问句(什么/哪/怎么/为什么)",
    "反问句(难道/不是…吗)", "祈使/请求句(别/请/给我)", "感叹句(啊/呀/哇)",
    "选择疑问(A还是B)", "双重否定(不是不…)", "把字句", "被字句",
    "连动句(先…再…/一边…一边)", "兼语句(让/叫/请+人+做)",
    "并列(又…又…/也…也…)", "转折(可是/但是/不过)", "因果(因为/所以)",
    "条件(如果/要是…就)", "递进(不但…而且)", "话题前置(那个作业…)",
    "比拟(像…一样/好像)", "插入语(对了/其实/我说)", "无主句省略主语",
    "数量词开头(一个/两只…)", "时间地点开头(今天早上…)",
]

_SENTENCE_TYPE_EN = [
    "statement", "yes-no question", "wh-question (what/where/why)",
    "tag question (isn't it)", "imperative (don't/please)", "exclamation",
    "either-or question", "negative question (don't you)", "there is/are opener",
    "let's + verb", "want/wanna + to", "going to / gonna",
    "and/but compound", "because/so clause", "if/when clause",
    "comparative (bigger than)", "like/simile", "topic fronting (That game…)",
    "subject-less fragment (natural kid speech)", "two-part (wait— …)",
]

_SUBJECT_STYLE_CN = [
    "第一人称我", "第一人称我们", "第三人称自称(宝宝/名字/他称自己)",
    "第二人称你/您", "省略主语", "具体人名作主语(同桌/老师)",
    "事物作主语(这个/那本书)", "地点作主语(教室里…)", "时间作主语(刚才…)",
    "疑问词作主语(谁/什么)", "兼语前主语(妈妈让我…)",
]

_SUBJECT_STYLE_EN = [
    "I", "we", "you", "he/she name", "subject dropped",
    "thing as subject (This toy…)", "place (At school…)", "time (Yesterday…)",
    "who/what as subject", "Mom/Dad as subject",
]

_OBJECT_STYLE_CN = [
    "具体物品宾语", "抽象宾语(理由/秘密/办法)", "处所宾语(去操场/在家)",
    "人作宾语(帮弟弟/问老师)", "动宾(写作业/读书)", "数量+名宾语",
    "小句宾语(说他不对)", "无宾语不及物(跑了/哭了)", "双宾(给我一块糖)",
    "宾语前置强调(饭我不想吃)", "存现(桌上有个…)",
]

_OBJECT_STYLE_EN = [
    "concrete object", "abstract (idea/reason)", "place (to the park)",
    "person object (tell Mom)", "verb-ing object (doing homework)",
    "quantity + noun", "clause object (that he lied)", "intransitive (no object)",
    "double object (give me…)", "fronted object emphasis",
]

_VERB_FOCUS_CN = [
    "动作(跑拿扔推打开)", "心理(怕想喜欢担心)", "能愿(要会能敢应该)",
    "趋向(上来出去进来)", "判断(是像)", "状态变化(变成湿了倒了)",
    "感官(看见听见闻到)", "互动(问告诉吵答应)", "完成体(了/过)",
    "尝试(试着/差点)", "重复动作(又…了一次)",
]

_VERB_FOCUS_EN = [
    "action (run grab throw)", "mental (think scared love)",
    "modal (can/will/wanna)", "motion (come/go up down)",
    "be/look/seem", "change (got wet fell)", "perception (see hear)",
    "speech act (ask tell yell)", "past/perfect (did/have done)",
    "try/almost (tried to / almost)",
]

_CLAUSE_CONNECTOR_CN = [
    "只用短连接(然后/还有)", "转折连接(可是)", "因果连接(因为/所以)",
    "假设连接(如果)", "并列不用连词靠逗号", "单句不嵌套",
    "一层补语(我觉得…)", "两层信息(…但是…所以)",
]

_CLAUSE_CONNECTOR_EN = [
    "and then", "but/however", "because/so", "if/when",
    "comma splice (kid-like)", "single clause only", "I think (that)…",
    "two clauses chained",
]

_READING_MATERIAL_SENTENCE_MIX = [
    "材料中含疑问句", "材料中含感叹句", "材料中含把/被字句",
    "材料中含排比或对偶", "材料中含长定语", "材料中含数字列举",
    "材料中含对话引号句", "材料中含条件/转折复句", "材料中含文言判断句",
    "材料中含英文从句", "材料中含设问句", "材料中含并列分句",
]


def _syntactic_pools_for_lang(lang_key: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Return (sentence_types, subjects, objects, verbs, connectors) for lang_key."""
    use_en = lang_key in ("pure_en", "en_mostly")
    use_cn = lang_key in ("pure_cn", "cn_mostly")
    if use_en and not use_cn:
        return (
            list(_SENTENCE_TYPE_EN),
            list(_SUBJECT_STYLE_EN),
            list(_OBJECT_STYLE_EN),
            list(_VERB_FOCUS_EN),
            list(_CLAUSE_CONNECTOR_EN),
        )
    if use_cn and not use_en:
        return (
            list(_SENTENCE_TYPE_CN),
            list(_SUBJECT_STYLE_CN),
            list(_OBJECT_STYLE_CN),
            list(_VERB_FOCUS_CN),
            list(_CLAUSE_CONNECTOR_CN),
        )
    # Mixed: combine pools so rows can code-switch
    return (
        list(_SENTENCE_TYPE_CN) + list(_SENTENCE_TYPE_EN),
        list(_SUBJECT_STYLE_CN) + list(_SUBJECT_STYLE_EN),
        list(_OBJECT_STYLE_CN) + list(_OBJECT_STYLE_EN),
        list(_VERB_FOCUS_CN) + list(_VERB_FOCUS_EN),
        list(_CLAUSE_CONNECTOR_CN) + list(_CLAUSE_CONNECTOR_EN),
    )


def draw_syntactic_axes(
    lang_key: str,
    batch_size: int,
    seed_text: str,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Per-row sentence type / subject / object / verb / connector plan."""
    st, su, ob, vb, cc = _syntactic_pools_for_lang(lang_key)
    return (
        _pick_shuffled_pool(st, batch_size, seed_text, "syn_sent"),
        _pick_shuffled_pool(su, batch_size, seed_text, "syn_subj"),
        _pick_shuffled_pool(ob, batch_size, seed_text, "syn_obj"),
        _pick_shuffled_pool(vb, batch_size, seed_text, "syn_verb"),
        _draw_diverse_axis_list(
            cc,
            batch_size,
            random.Random(hashlib.md5(f"syn_conn|{seed_text}".encode()).hexdigest()),
        ),
    )


def build_syntactic_diversity_rules(
    lang_key: str,
    batch_size: int,
    is_reading: bool = False,
) -> str:
    """Batch-level hard rules for syntax variety."""
    min_sent = min(7, batch_size)
    min_subj = min(5, batch_size)
    min_verb = min(5, batch_size)
    lang_note = ""
    if lang_key == "frequent_mix":
        lang_note = "- 中英混杂时，句式计划可用中文或英文结构，但一条内切换要自然\n"
    elif lang_key == "pure_en":
        lang_note = "- 全部用英文句法，禁止中文句式骨架\n"
    elif lang_key == "pure_cn":
        lang_note = "- 全部用中文句法，禁止整句英文骨架\n"

    reading_note = ""
    if is_reading:
        reading_note = (
            "- 朗读正文里也要换句式：不要10条都是陈述句或都是问句\n"
            "- 读题时设问/条件/转折从句要有，动词类型随题型变化\n"
        )

    return f"""
=== 句式 / 主语 / 宾语 / 动词多样性（必须遵守） ===
{lang_note}{reading_note}- 本 batch 至少 {min_sent} 种不同「句式」、{min_subj} 种不同「主语类型」、{min_verb} 种不同「动词类型」
- 任意两条不得相同：句式 + 主语类型 + 核心动词类；宾语类型也尽量不同
- 以「我」开头的句子本 batch 最多 2 条；以「妈妈/老师」开头的最多 2 条
- 把字句、被字句、疑问句、感叹句、转折复句每种最多 2 条（有则分散）
- 核心动词不要10条都是「想/要/是」；换跑、拿、问、怕、发现、变成等
- 主语换：我/我们/你/他名/事物/时间/省略；宾语换：人/物/处所/小句/无宾语
- 逐条计划里的「句式/主语/宾语/动词」必须体现在正文语法上，不是只写标签
"""


# ---------------------------------------------------------------------------
# Prompt Builder v3
# ---------------------------------------------------------------------------

def build_prompt(
    scenario_key: str,
    subscene: str,
    length_key: str,
    lang_key: str,
    emotion: str,
    age_tier: str,
    batch_size: int,
    suppression_hint: str = "",
    task_id: Optional[int] = None,
) -> str:
    scenario = SCENARIOS[scenario_key]
    length_spec = LENGTH_SPECS[length_key]
    lang_spec = LANG_SPECS[lang_key]
    age = AGE_FEATURES[age_tier]
    emotion_profile = EMOTION_PROFILES[emotion]
    is_stress = scenario.get("is_stress_test", False)

    # Build tag guidance
    primary = ", ".join(emotion_profile["primary_tags"])
    secondary = ", ".join(emotion_profile["secondary_tags"]) if emotion_profile["secondary_tags"] else "无"
    density_min, density_max = TAG_DENSITY_MAP[emotion_profile["tag_density"]]

    # Build age feature text
    if "cn" in lang_key or lang_key in ("cn_mostly", "frequent_mix"):
        age_features_text = "\n".join(f"- {f}" for f in age["cn_features"])
    else:
        age_features_text = "\n".join(f"- {f}" for f in age["en_features"])

    mix_section = ""
    if lang_key in ("cn_mostly", "en_mostly", "frequent_mix"):
        mix_text = "\n".join(f"- {p}" for p in age["mix_patterns"])
        mix_section = f"""
=== 中英混杂模式 ===
{mix_text}
"""

    # Preschool-specific mandatory features
    preschool_note = ""
    if age_tier == "preschool":
        preschool_note = """
=== 幼儿园特别要求（必须遵守，缺一项视为失败） ===
每条文本必须包含至少1项幼儿语言特征（不要写成小学生/青少年口吻）：
- 叠词: 吃饭饭、喝水水、睡觉觉、抱抱抱、走走走、洗手手
- 三连重复: 要要要、不不不、好好好、怕怕怕
- 第三人称自我指代: 宝宝要、明明不吃、朵朵不想
- 发音不准写法: 七饭、回水、脑斧、灰机
- 禁止: no cap、emoji式网络词(emo)、过长复杂从句、像作文的叙述
- 短促但完整的句子，像刚学说话的孩子，但不能是半句
"""

    stress_instruction = build_stress_instructions(scenario_key, subscene, batch_size) if is_stress else ""
    reading_instruction = build_reading_aloud_instructions(
        scenario_key, subscene, lang_key, length_key, age_tier, batch_size, task_id
    )
    diversity_instruction = build_diversity_instructions(
        scenario_key,
        subscene,
        emotion,
        age_tier,
        lang_key,
        batch_size,
        task_id,
    )
    # Build detailed tag placement guide
    tag_guide_lines = []
    for tag_name, tag_info in TAG_DEFINITIONS.items():
        tag_guide_lines.append(
            f"{tag_name}: {tag_info['description']}\n"
            f"  位置规则: {tag_info['placement']}\n"
            f"  好例子: {tag_info['examples'][0]}"
        )
    tag_guide_text = "\n\n".join(tag_guide_lines)

    # Language-specific instruction injection
    lang_override = ""
    if lang_key == "pure_en":
        lang_override = """
[CRITICAL LANGUAGE RULE: All output must be NATURAL ENGLISH child speech. No Chinese characters allowed.
Use authentic child English: "wanna", "gonna", "very very", child grammar errors like "tooked", "drawed".
Include complete false starts and self-repairs: "Can we go now, wait, after class?" and "I-I-I want it, no, I don't want it now".
DO NOT use: um, uh, ugh, soooo, sooo, super duper, like as filler, you know. These fillers are often NOT in ASR vocab.
]"""
    elif lang_key == "en_mostly":
        lang_override = """
[CRITICAL LANGUAGE RULE: Mostly English child speech. MUST include at least 1 Chinese word (妈妈, 爸爸, 抱抱, 谢谢, etc.).
Use authentic child English: "wanna", "gonna", "very very", child grammar errors like "tooked", "drawed".
Include complete false starts and self-repairs: "Can we go now, wait, after class?" and "I-I-I want it, no, I don't want it now".
DO NOT use: um, uh, ugh, soooo, sooo, super duper, like as filler, you know. These fillers are often NOT in ASR vocab.
]"""
    elif lang_key == "frequent_mix":
        lang_override = """
[CRITICAL LANGUAGE RULE: Natural bilingual code-switching between English and Chinese.
要求: 中文和英文都必须有显著占比（各至少30%）。不能一整句只有一种语言。
必须在句子中间切换，不要前半句全中文后半句全英文。
好例子: "我觉得这个idea超棒的，totally agree", "You know what? 我今天发现一件超有趣的事", "It's unfair不公平".
坏例子: "我觉得这个idea超棒的"（只有1个英文词，不算frequent_mix）
]"""

    # Length emphasis note
    length_strict = ""
    if length_key == "ultra_short":
        length_strict = """
⚠️ 长度约束 ⚠️
当前任务: ultra_short（极短）。去掉标签后，中文最多5字/英文最多3词。
好例子: "怕怕[dissatisfaction-hnn]", "要要要！[surprise-ah]", "No way![laughter]"
"""
    elif length_key == "short":
        length_strict = """
⚠️ 长度约束 ⚠️
当前任务: short（短句）。去掉标签后，中文最多8字/英文最多6词。
标签按极低概率注入（约15%概率），大部分句子不需要标签。
好例子: "那个，我要抱抱[laughter]", "不不不，才不是[dissatisfaction-hnn]", "妈妈我要吃糖"（无标签）
"""
    elif length_key == "medium":
        length_strict = """
⚠️ 长度约束 ⚠️
当前任务: medium（中等）。去掉标签后，中文最多15字/英文最多12词。
标签按极低概率注入（约20%概率），大部分句子不需要标签。
"""
    elif length_key == "long":
        length_strict = """
⚠️ 长度约束 ⚠️
当前任务: long（较长）。去掉标签后，中文最多25字/英文最多20词。
标签按极低概率注入（约20%概率），大部分句子不需要标签。
"""
    elif length_key == "very_long":
        length_strict = """
⚠️ 长度约束 ⚠️
当前任务: very_long（长句）。去掉标签后，中文最多40字/英文最多35词。
标签按低概率注入（约25%概率），约3/4句子无标签，有标签的通常只放1个。
"""

    prompt = f"""你是一位专业的儿童语音数据采集专家。请生成 {batch_size} 条听起来 EXACTLY 像真实儿童自然口语的文本。{lang_override}

{length_strict}

这些文本将用于训练 OmniVoice TTS 模型。OmniVoice 支持在文本中插入非语言标签（如 [laughter]）来生成对应的副语言声音。

=== 儿童画像 ===
年龄: {age["name"]}
场景: {scenario["name"]} — {subscene}
情绪: {emotion}（标签密度: {emotion_profile["tag_density"]}，位置倾向: {emotion_profile["position_bias"]}）
长度: {length_spec["cn"] if "cn" in lang_key else length_spec["en"]}
语言: {lang_spec}
{stress_instruction}
{reading_instruction}
{diversity_instruction}
{suppression_hint}
=== 年龄段语言特征 ===
{age_features_text}
{mix_section}
{preschool_note}
=== 口语自然度规则 (极其重要，必须严格遵守) ===

**强制要求：每条文本必须至少包含以下1项口语特征（没有算失败）。推荐优先使用填充词，最简单且最自然：**
- 【首选】填充词（中文）: 嗯、啊、那个、就是、呃
  好例子: "嗯，我不想去嘛"、"那个，我要抱抱"
  【禁止】不要使用英文填充词 um, uh, like, you know, soooo 等——这些词通常不在ASR识别词表中
- 自我修复: "我要去，不对，我不想去了"、"我-我-我想要那个"、"Can we go now, wait, after class?"
- 句子重叠/重复: "那个那个"、"就是就是"、"very very"、"不不不"
- 完整改口: "然后那个，我想起来是数学书，不是语文书"
- 短口语句: "我其实想先读这页"、"但是那个题我还没写完"

**WER/ASR 对齐要求（极其重要）**
- 每条 text 必须是完整可朗读文本，不能是半句、残句、说到一半。
- 禁止用省略号 "..." 或 "…" 表示没说完；需要停顿时用逗号。
- 结尾必须自然收束，不能以逗号、顿号、破折号、省略号结尾。
- 自我修复、口吃、重复都可以有，但修复后必须把最终意思说完整。

**儿童特有的语音模式（越多越好）：**
- 拖长音: "好——的——"、"知——道——了——"（中文拖长音）；英文可用 "veeeery", "sooo" 但不要过度使用
- 突然的 shout: 句子中间突然大声喊一个词
- 模仿大人但用词不当
- 用身体动作描述代替语言（如"这样这样"）
- 不会用复杂连词，只用"然后"、"还有"、"但是"

**句式与主谓宾（与逐条计划一致，必须落实）**
- 每条按「句式/主语/宾语/动词」计划写，batch 内换陈述/疑问/感叹/祈使/把字/被字/转折/因果等
- 主语换：我、我们、你、名字、事物、时间、省略；不要10条都以「我」开头
- 动词换：动作、心理、能愿、趋向、感官、互动；不要10条都是「想/要/是」
- 宾语换：具体物、人、处所、小句、无宾语、双宾；两条之间宾语类型也要变

**中文特有（中文文本必须至少包含1项）：**
- 叠词: 吃饭饭、喝水水、睡觉觉、抱抱、亲亲
- 三连重复: 要要要、不不不、好好好、抱抱抱
- 语气词: 嘛、呀、啦、呢、吧、哦、哼
- 童言童语: "七饭"(吃饭)、"脑斧"(老虎)

**英文特有（英文文本必须至少包含1项）：**
- 简单语法错误: "tooked", "drawed", "falled", "more better"
- "gonna", "wanna", "super", "very very"
- 【禁止】不要使用: "um", "uh", "ugh", "soooo", "like"(作填充词), "you know" — 这些词通常不在ASR识别词表中

**通用禁忌**
- NEVER 用括号解释（如"我(小明)"）
- **禁止在 text 里写拼音、注音、声调符号**：如 māma、zhōng1、nǐ hǎo、ㄅㄆㄇ、/ma/、IPA；不要写「读作…」「拼音是…」
- **禁止把标点当名字念出来**：不要写「省略号」「逗号」「句号」「问号」等代替 … ， 。 ？ ； 应用正常中文/英文标点或直接口语，不要用符号名凑句子
- **禁止仅用于注音的符号**：…（省略号字符）、·•※◆●□、单独声调符号 ˉˊˇˋ；数学/几何题里 ∠△° 等可保留
- 场景是「拼音/注音」时：正文只朗读汉字或英文词，卡顿用口语描述，不要把课本上的拼音行写进 text
- 年龄适当的内容，不要涉及抽象概念、政治、暴力、成人内容
- 不要像书面作文，但必须是完整可朗读的一句话或一小段

=== 标签使用详细指南 ===

**情绪标签策略**（当前情绪: {emotion}）:
- 主要标签: {primary}
- 次要标签: {secondary}
- 每条文本标签数: {density_min}-{density_max} 个
- 位置倾向: {emotion_profile["position_bias"]}
  - start_weighted: 标签倾向于放在文本前半部分
  - end_weighted: 标签倾向于放在文本后半部分或句尾
  - scattered: 标签分散在文本各处
  - mixed: 无明显倾向

**标签位置规则**（极其重要，必须遵守）:

{tag_guide_text}

**标签组合规则**:
- [laughter] 和 [sigh] 通常不会同时出现（矛盾情绪）
- 同类型标签不要连续出现: "[surprise-ah][surprise-wa]" 不好
- 标签可以组合但要自然: "[surprise-ah]有虫子！[surprise-wa]好大一只"
- 短文本（ultra_short/short）最多1个标签
- 中/长文本最多2-3个标签

**标签密度控制（概率注入，不强制）**:
- ultra_short: 0-1个标签，约10%概率注入
- short: 0-1个标签，约15%概率注入
- medium: 0-1个标签，约20%概率注入
- long: 0-1个标签，约20%概率注入
- very_long: 0-2个标签，约25%概率注入

**标签注入原则（重要）**:
- 绝大多数句子（约75-85%）不需要任何标签，保持自然口语即可
- 仅在情绪特别强烈时才考虑加入1个标签，极少情况才用2个
- 优先生成不带标签的纯文本，标签只是偶尔点缀
- 幻想(fantasy)场景：建议约15%包含惊叹类标签 [surprise-wa] 或 [laughter]
- 情绪爆发(emotions)场景：建议约20%包含情绪标签

=== 场景典型标签参考 ===
本场景典型标签: {', '.join(scenario.get('typical_tags', []))}

=== 输出格式 ===

只返回 JSON 数组。每个元素必须有这些字段:
- "text": 带自然非语言标签的语音文本（用于 TTS 克隆）
- 说明: 系统会自动从 text 派生 "text_tn"（去掉标签并做 ASR/WER 归一化），无需在 JSON 里输出 text_tn
- "length_type": "{length_key}"
- "lang_type": "{lang_key}"
- "scenario": "{scenario_key}"
- "subscene": "{subscene}"
- "emotion": "{emotion}"
- "age_tier": "{age_tier}"
- "language": 如果中文为主设为 "zh"，英文为主设为 "en"

好例子（大部分无标签，少数带标签点缀）:
[
  {{"text": "妈妈你看，那只小狗在追自己的尾巴耶", "length_type": "short", "lang_type": "pure_cn", "scenario": "daily_family", "subscene": "要零食要玩具", "emotion": "happy", "age_tier": "preschool", "language": "zh"}},
  {{"text": "又要背课文了，不对不对，我还没带书", "length_type": "medium", "lang_type": "pure_cn", "scenario": "school", "subscene": "背课文卡顿", "emotion": "worried", "age_tier": "early_elem", "language": "zh"}},
  {{"text": "你怎么突然出来了，吓死我了！", "length_type": "short", "lang_type": "pure_cn", "scenario": "emotions", "subscene": "惊讶大叫", "emotion": "shocked", "age_tier": "preschool", "language": "zh"}},
  {{"text": "嗯，那个，我要吃饭饭[laughter]", "length_type": "short", "lang_type": "pure_cn", "scenario": "daily_family", "subscene": "吃饭挑食讨价还价", "emotion": "happy", "age_tier": "preschool", "language": "zh"}},
  {{"text": "I-I-I want it, no, I don't want it now", "length_type": "short", "lang_type": "pure_en", "scenario": "daily_family", "subscene": "被批评顶嘴", "emotion": "frustrated", "age_tier": "early_elem", "language": "en"}},
  {{"text": "Um, I think 我觉得这个idea还不错[surprise-oh]", "length_type": "medium", "lang_type": "frequent_mix", "scenario": "learning", "subscene": "科学实验好奇", "emotion": "curious", "age_tier": "upper_elem", "language": "zh"}}
]

生成恰好 {batch_size} 条。不要在 JSON 前后加任何文字。
"""
    return prompt


_READING_ALOUD_SCENARIOS = frozenset({
    "primary_chinese",
    "primary_english",
    "primary_math",
    "poetry_classics",
})
_READING_SUBSCENE_HINTS = (
    "读", "朗读", "跟读", "指读", "默读", "念", "诵读", "背诗", "背课文",
    "课文", "题干", "题目", "单词表", "Dialog", "Read", "read", "Word list", "拔高",
)

_READING_DIFFICULTY_BY_SUBJECT = {
    "primary_chinese": {
        "ratio_note": "本 batch 至少 60% 条为「拔高」难度（接近初一至初二语文课本/试卷），其余可为小学高年级课文",
        "advanced_topics": (
            "文言文短篇(如论语、陋室铭、岳阳楼记一句)、七律/词牌、说明文(科普/社会现象)、"
            "议论文句群、长篇阅读理解题干(含“根据上文”“主旨”“修辞”)"
        ),
        "cn_examples": (
            "拔高-文言: \"庆历四年春，滕子京谪守巴陵郡[sigh] 不对，谪守巴陵郡，越明年政通人和[question-oh]\"",
            "拔高-阅读: \"阅读材料：随着城市化进程加快……问：本文主要运用了什么说明方法？[question-ah] 列数字？还是打比方？\"",
            "拔高-古诗: \"先天下之忧而忧，后天下之乐而乐[surprise-oh] 乐而乐，后面怎么背的来着\"",
            "拔高-议论: \"我认为网络让生活更方便，但是[sigh] 但是也会让人上瘾[question-ei]\"",
            "跟读: \"老师，我再读一遍，是故人之国，故人的国[question-ah] 不对，故国\"",
            "读题: \"根据上文，主人公第一次见爷爷时的心情是——A紧张 B开心 C害怕[sigh] 我选B？\"",
            "基础: \"第12课 草船借箭 周瑜说，诸葛亮说，三天造十万支箭[question-ei]\"",
        ),
        "bad_examples": (
            "\"小蝌蚪找妈妈春天到了\"（过于低年级）",
            "\"我不想读课文\"（没有朗读正文）",
        ),
    },
    "primary_english": {
        "ratio_note": "本 batch 至少 60% 条为「拔高」难度（接近初一至初二英语课文/听力/阅读），避免只有 color/cat/apple",
        "advanced_topics": (
            "120-180词段落、完形/阅读长题干、含定语从句/被动/完成时的对话、"
            "科普或历史类英语短文、演讲/辩论节选"
        ),
        "cn_examples": (),
        "en_examples": (
            "Advanced-passage: \"Although the experiment was conducted twice, the results were not consistent until the temperature was lowered.[question-oh]\"",
            "Advanced-follow: \"Read after me: The boy who won the competition said that he had been training for three years.[sigh] for three years\"",
            "Advanced-QA: \"Passage: Climate change affects crop yields. Question: What is the main idea of paragraph 2?[question-ah]\"",
            "Dialog: \"A: Where is the science lab? B: It's on the second floor, next to the library.[question-oh] second floor\"",
            "Cloze: \"Tom was so tired that he ___ asleep during the movie.[sigh] fell? fall? fallen?\"",
            "Basic: \"Page 5. How many apples? There are eight apples.[laughter]\"",
        ),
        "bad_examples": (
            "\"Apple, banana, cat\"（单词表过简单）",
            "\"I like red\"（无课文/题干正文）",
        ),
    },
    "primary_math": {
        "ratio_note": "本 batch 至少 60% 条为「拔高」难度（接近初一至初二数学题题干），避免只有 3+5 口算",
        "advanced_topics": (
            "一元一次方程应用题、二元一次方程组、一次函数与图像、勾股定理、相似三角形、"
            "不等式、概率统计、百分数复杂应用、圆与扇形"
        ),
        "cn_examples": (
            "拔高-几何: \"已知Rt△ABC中，∠C=90°，AC=6，BC=8，求AB的长[sigh] AB……是10吗？用勾股定理[question-oh]\"",
            "拔高-行程: \"甲、乙两人同时从A、B两地相向而行，甲速60km/h，乙速40km/h，2小时后仍相距20km，求AB距离[question-ah]\"",
            "拔高-函数: \"一次函数y=2x-3的图像经过点P(2,k)，求k的值[question-ei] k等于……1？\"",
            "拔高-统计: \"根据条形图，三班植树棵数是中位数的多少倍？[question-oh] 倍？\"",
            "读选项: \"下列计算正确的是 A 2a+3a=5a B 3a·2a=6a² C ……[sigh] 我念太快了\"",
            "口算念题: \"口算，24×15，嗯，24乘10是240，再加……[question-ei]\"",
            "基础: \"小明有5个苹果，妈妈又给他3个，一共几个？[question-ah]\"",
        ),
        "bad_examples": (
            "\"3加5等于几\"（过于简单）",
            "\"这道题好难\"（没有读出题干）",
        ),
    },
}

_READING_DIVERSITY_AXES = {
    "primary_chinese": {
        "modes": [
            "跟读老师", "指读课文", "朗读段落", "默读出声", "读阅读题干",
            "读选项ABCD", "读注释词语", "跟读点读笔", "读黑板句子", "过关背诵",
            "分角色朗读", "轮读下一段", "课本有拼音但只念汉字正文", "读看图写话提示",
            "读默写纸", "读作文评语", "读词典释义",
        ],
        "materials": [
            "文言文短篇", "七律绝句", "词牌短调", "说明文科普", "议论文片段",
            "记叙文节选", "成语典故", "名著节选", "新闻短评", "同步练习阅读",
            "古诗背诵", "课后思考题", "对联上下联", "寓言故事", "书信格式",
            "演讲稿节选", "剧本对话", "广告语分析",
        ],
        "settings": [
            "早读课", "语文课堂", "在家写作业", "小组朗读", "录音作业",
            "黑板跟读", "同桌互查", "过关检查",
        ],
        "stumbles": [
            "漏字", "重复上一词", "同音字读错", "断句错误", "文言字卡壳",
            "读太快含糊", "读到一半问老师", "自我纠正", "读串行", "跳过硬词",
        ],
        "openings": [
            "直接起句背", "先报篇名作者", "跟读老师说第几段", "先读注释再读正文",
            "指读第一行", "从‘阅读材料’念起", "先读题干再读材料",
        ],
    },
    "primary_english": {
        "modes": [
            "Repeat after me", "指读绘本", "逐句跟读", "读Dialog", "读阅读短文",
            "读完形段落", "读题干", "读选项", "跟读听力", "读板书",
            "Shadow reading", "Choral reading", "读字幕台词", "读歌曲歌词",
            "读邮件格式", "读菜单说明",
        ],
        "materials": [
            "课本Dialog", "120词短文", "完形段落", "阅读理解", "听力脚本",
            "科普短文", "演讲节选", "辩论观点", "图表说明", "历史阅读",
            "Word list句", "考试Part阅读", "广告文案", "旅游指南",
            "日记体短文", "童话改编", "采访实录",
        ],
        "settings": [
            "英语课堂", "外教示范", "听力跟读", "考试朗读", "绘本时间",
            "角色扮演", "小组对话", "在家点读",
        ],
        "stumbles": [
            "漏冠词", "复数忘读s", "连读错误", "长句断不开", "生词卡住",
            "时态念错", "重音不对", "读一半改口", "跟读落后", "数字日期读错",
        ],
        "openings": [
            "Read after me 起句", "先读标题", "从 Dialog 第一句", "先读 Question 1",
            "指读绘本页码", "先读选项再读短文", "跟读外教一句",
        ],
    },
    "primary_math": {
        "modes": [
            "念完整题干", "念条件再念设问", "念算式", "念选项", "指读关键词",
            "边写边念", "念图形已知", "念表格数据", "同桌互读题目",
        ],
        "materials": [
            "行程问题", "工程问题", "勾股几何", "一次函数", "方程组",
            "概率统计", "图表题", "百分数应用", "口算式", "竖式步骤",
            "填空选择", "相似三角形", "圆与扇形", "浓度配比", "分段计费",
            "坐标描点", "不等式方案", "数列规律", "质数分解",
        ],
        "settings": [
            "课堂讲题", "作业辅导", "口算比赛", "黑板读题", "错题订正",
            "小组讨论", "自己验算前重读",
        ],
        "stumbles": [
            "漏单位", "漏‘一共’", "把加号读成减号", "倍比念反", "数字口误",
            "几何条件漏半句", "念到一半推翻", "人物名读混", "分数念错",
        ],
        "openings": [
            "从‘已知’念起", "先读设问再读条件", "指读图形标注", "先读表格标题",
            "边写边念算式", "同桌互读第一行", "先读题号",
        ],
    },
    "poetry_classics": {
        "modes": ["跟读", "背诵朗读", "读注释", "读题目", "飞花令接诗", "接龙背诗"],
        "materials": ["五言律诗", "七言律诗", "宋词小令", "乐府诗", "古文名句", "课标必背"],
        "settings": ["早读", "语文课堂", "背诵过关", "比赛展示", "线上打卡"],
        "stumbles": ["上下句背反", "字音读错", "节奏断了", "诗人记混", "韵脚念错"],
        "openings": ["直接起句背", "先报诗名", "跟读老师", "读到一半停"],
    },
}

# Concrete topic seeds — force each row to use a different passage / problem stem.
_READING_TOPIC_SEEDS: Dict[str, List[str]] = {
    "primary_chinese": [
        "《岳阳楼记》‘庆历四年春’开篇", "《陋室铭》‘山不在高’", "《论语》‘学而时习之’",
        "《出师表》‘先帝创业未半’", "《桃花源记》‘晋太元中’", "《醉翁亭记》‘环滁皆山’",
        "《三峡》‘自三峡七百里’", "《马说》‘世有伯乐’", "《爱莲说》‘水陆草木之花’",
        "《天净沙·秋思》", "《水调歌头》‘明月几时有’", "《念奴娇·赤壁怀古》选句",
        "《声声慢》选句", "《如梦令·昨夜雨疏风骤》", "《七律·长征》选句",
        "说明文：城市垃圾分类", "说明文：湿地保护", "说明文：蜜蜂酿蜜过程",
        "议论文：阅读的意义", "议论文：诚信的重要性",
        "记叙文：运动会接力掉棒", "新闻稿：科技展进校园",
        "阅读理解：敦煌莫高窟", "阅读理解：青铜面具出土",
        "《从百草园到三味书屋》节选", "《背影》‘我最不能忘记的是他的背影’",
        "《藤野先生》开头", "《春》‘盼望着盼望着’",
        "同步练习：环境描写段落", "课后题：判断拟人还是比喻",
        "成语：刻舟求剑典故", "名著：《西游记》三打白骨精节选",
        "词牌：《清平乐·村居》", "默写：《观沧海》水何澹澹",
        "跟读：同步练习《赵州桥》说明段", "读选项：下列没有语病的一项",
        "《木兰诗》‘唧唧复唧唧’", "《塞翁失马》寓意段", "《记承天寺夜游》",
        "说明文：蜜蜂社会分工", "议论文：网络利弊", "记叙：第一次独自回家",
        "《阿长与山海经》节选", "《济南的冬天》开头", "《观潮》浪潮段",
    ],
    "primary_english": [
        "Dialog: school library rules", "Dialog: lost and found office",
        "Dialog: booking a dentist", "Dialog: sports day signup",
        "Passage: penguin migration", "Passage: Mars rover mission",
        "Passage: coral reef bleaching", "Passage: ancient Silk Road",
        "Cloze: past perfect vacation", "Cloze: passive voice in lab report",
        "Cloze: relative clause about the boy who saved a cat",
        "Reading: climate and crop yields", "Reading: history of printing",
        "Reading: how vaccines work", "Reading: Olympic host cities",
        "Debate: phones in school", "Interview: young inventor",
        "Chart: bar chart of after-school clubs", "Recipe: banana pancake steps",
        "Story: camping in the rain", "News: community clean-up day",
        "Letter: apology for being late", "Listening: airport gate change",
        "Speech: protect endangered pandas", "Biography: Marie Curie discovery",
        "Dialog: winter coat size exchange", "Dialog: group project roles",
        "Part reading: the water cycle", "Word list: environment vocabulary in sentences",
        "Repeat: complex sentence with although and until",
        "Read options: best title for the passage",
        "Dialog: canteen allergy notice", "Passage: volcano eruption safety",
        "Cloze: comparative and superlative cities", "Reading: bee pollination",
        "Debate: homework on weekends", "Letter: invite to birthday party",
        "Listening: school bell schedule", "Speech: reduce plastic bags",
    ],
    "primary_math": [
        "行程：甲乙两地相向而行", "行程：环形跑道同向追及", "行程：顺水逆水划船",
        "工程：甲乙合作修水渠", "工程：注水排水交替", "工程：三人合作完成",
        "勾股：Rt△求斜边", "勾股：梯子靠墙高度", "勾股：坐标系两点距离",
        "一次函数：求k与图像", "一次函数：两车出发时间", "一次函数：利润与销量",
        "二元一次：鸡兔同笼", "二元一次：成人儿童票价", "二元一次：数字对调",
        "不等式：文具采购预算", "不等式：最大载客人数",
        "概率：摸球不放回", "概率：转盘指针区域",
        "统计：条形图读倍数", "统计：扇形图百分比", "统计：折线图趋势",
        "百分数：打折促销", "百分数：增长率比较", "百分数：浓度配制",
        "几何：平行线拐角", "几何：相似三角形测高", "几何：全等证明已知",
        "圆：已知半径求周长", "圆：扇形面积", "圆：圆环宽度",
        "口算：24×15竖式思路", "竖式：除法验算有余数",
        "应用：浓度糖水混合", "应用：分段计费水电", "应用：植树间隔数",
        "坐标：描点连线", "方程：年龄差问题",
        "工程：交替工作休息", "行程：追及相遇", "统计：中位数与众数",
        "几何：仰角测楼高", "概率：两次抛硬币", "百分数：利息计算",
        "不等式：租车方案比较", "因式分解：提公因式", "分式：化简求值",
    ],
    "poetry_classics": [
        "《静夜思》", "《春晓》", "《登鹳雀楼》", "《望庐山瀑布》",
        "《江雪》", "《枫桥夜泊》", "《泊秦淮》", "《赤壁怀古》选句",
        "《声声慢》选句", "《如梦令》", "《己亥杂诗》", "《过零丁洋》",
        "《悯农》", "《咏鹅》", "《江南》", "《鹿柴》",
        "《送元二使安西》", "《凉州词》", "《早发白帝城》", "《游子吟》",
        "《四时田园杂兴》", "《题西林壁》", "《观书有感》", "《冬夜读书示子聿》",
        "飞花令：含‘月’", "飞花令：含‘花’", "接龙：下句‘举头望明月’",
        "读注释：‘阡陌’释义", "读题目：诗人表达什么情感",
    ],
}

_CHILD_NAME_POOL = [
    "晓彤", "子涵", "浩宇", "梓轩", "雨桐", "俊杰", "欣怡", "博文",
    "语彤", "子墨", "思远", "若曦", "嘉豪", "诗涵", "宇航", "悦然",
    "小北", "阿杰", "乐乐", "童童", "磊磊", "芳芳", "阿凯", "苗苗",
    "一诺", "子睿", "沐晨", "诗琪", "俊熙", "梓涵", "雨泽", "思琪",
    "奕辰", "佳怡", "泽宇", "欣妍", "皓轩", "梦瑶", "子轩", "语嫣",
    "阿朵", "小虎", "豆豆", "团团", "果果", "壮壮", "妮妮", "阳阳",
]

_READING_OPENING_HINTS = [
    "直接从正文第一个字读起", "先小声说‘我开始了’", "先读标题或篇名",
    "跟读老师说‘我读第二段’", "先读页码或题号", "先读‘阅读材料’",
    "先读‘Question’或‘第几题’", "先指读再念", "先读括号里的条件",
    "从‘已知’开始念", "从‘Read after me’开始", "先读人名再读正文",
    "先读作者朝代", "先读图注", "先读小标题", "先读脚注",
]

_READING_NUMERIC_HINTS = [
    "题中数字用37和58", "题中数字用2024年3月15日", "题中数字用3:5比例",
    "题中数字用12.5cm和8cm", "题中数字用第3题第2问", "题中数字用80%和120元",
    "题中数字用半径6cm", "题中数字用速度45km/h", "题中数字用样本200人",
    "题中数字用页码P47", "题中数字用海拔8848", "题中数字用分数3/7",
]

_CREATIVE_TWIST_POOL = [
    "刚发现误会了规则", "其实想说的是另一件事", "被大人打断后重说",
    "突然想起更紧急的事", "假装不在乎但语气出卖", "把责任推给弟弟/同桌",
    "用夸张比喻解释", "编了一个听起来真的理由", "刚赢/刚输的情绪残留",
    "发现秘密被说漏嘴", "东西掉了/坏了触发情绪", "时间来不及了着急",
    "记错名字或记错课", "把两个事件顺序说反", "听到隔壁声音分心",
    "本来要撒娇突然改严肃", "本来要哭突然笑场", "用反问收尾",
]

_PLACE_ANCHOR_POOL = [
    "客厅沙发", "厨房餐桌", "学校走廊", "操场篮球架下", "科学实验室",
    "图书馆角落", "公交车上", "小区滑梯旁", "医院候诊区", "外婆家阳台",
    "早读教室", "美术教室", "音乐教室", "宿舍上下铺", "雨天校门口",
    "超市收银台", "动物园熊猫馆", "夏令营帐篷", "游泳池边", "地铁车厢",
]

_PROP_ANCHOR_POOL = [
    "橡皮屑、卷笔刀", "贴纸册、闪光笔", "篮球、护膝", "魔方、计时器",
    "跳绳、计步器", "便当盒、保温杯", "雨衣、雨靴", "风筝线、线轴",
    "乐高标准件、说明书", "显微镜、载玻片", "口风琴、谱架",
    "公交卡、零钱", "手表、闹钟", "雨伞、伞骨断了", "快递盒、泡沫粒",
    "冰淇淋、化了的滴落", "奖状、折痕", "秘密纸条、胶条",
    "点读笔、耳机", "错题本、红笔", "乐高小人、缺件",
]

_GENERIC_TOPIC_FALLBACK = [
    "换具体物品名", "换说话对象", "换时间（刚发生/昨天/马上）",
    "换地点细节", "换孩子的小目标", "换阻碍原因",
    "换一句完全不同的开头", "换情绪触发点",
]

_AUDIENCE_POOL = [
    "对妈妈说", "对爸爸说", "对奶奶撒娇", "对老师说", "对同桌说",
    "对全班同学说", "对假想玩偶说", "对自己嘀咕", "对弟弟/妹妹吼",
    "对邻居阿姨", "对教练/体育老师", "对医生", "对电话那头的外婆",
    "对小组长", "对课代表", "对来家访的客人",
]

_TIME_ANCHOR_POOL = [
    "刚发生30秒内", "昨天放学后", "今天早上出门前", "午饭前",
    "午睡刚醒", "晚饭桌上", "睡前五分钟", "周末上午",
    "下雨被困室内", "考试刚结束", "春游大巴上", "寒假第一天",
    "暑假游泳后", "台风天停课", "运动会颁奖刚结束",
]

_SENSORY_DETAIL_POOL = [
    "带一种声音(砰/嘀/哗)", "带一种气味(饭香/油漆/雨土)",
    "带触觉(烫/凉/扎手)", "带颜色细节", "带温度冷热",
    "带口感(甜/苦/辣)", "带光线(刺眼/昏暗)", "带震动/发麻感",
]

_BATCH_CREATIVE_CHALLENGE_POOL = [
    "至少1条几乎不用标签、靠标点和语气词表现情绪",
    "至少1条用反问句收尾", "至少1条先否认再肯定",
    "至少1条提到具体数字或时间", "至少1条提到天气或季节",
    "至少1条是对话里只说一句但信息完整", "至少1条用拟声词开头",
    "至少1条模仿大人说话但露馅", "至少1条提到动画片/游戏名(虚构也可)",
    "至少2条说话对象不同", "至少2条地点完全不同",
    "至少1条英文夹杂(若语言允许)", "至少1条极短像脱口而出",
    "至少1条较长带小转折", "禁止10条全是抱怨口吻",
    "至少3种不同句式(陈述/疑问/感叹/把字/被字等)",
    "至少2条主语不是「我」", "至少2条核心动词不是想/要/是",
    "至少1条无宾语不及物", "至少1条双宾或兼语",
    "至少1条转折或因果复句", "至少1条祈使或反问",
]

_ANTI_OPENING_TEMPLATE_POOL = [
    "妈妈我想", "我觉得", "嗯那个", "今天老师", "阅读材料",
    "老师说要", "爸爸说", "不对不对", "为什么为什么",
    "One day", "I think", "Read after me", "已知Rt",
    "庆历四年", "床前明月光", "小明有", "Tom has",
    "先帝", "子曰", "Question 1", "根据以上材料",
]

_READING_REGISTER_HINTS = [
    "文言语气但读错断句", "白话说明文口吻", "新闻播报腔",
    "对话体口语", "议论文说理腔", "诗歌韵律感",
    "演讲稿排比", "剧本台词", "广告语押韵",
]

_READING_ERA_HINTS = [
    "先秦诸子语感", "唐宋诗词", "明清小说节选", "近现代白话",
    "当代说明科普", "外国故事译介腔", "同步教材课文",
    "考试模拟题风格", "课外读物摘抄",
]

_ADULT_NAME_POOL = [
    "王老师", "李老师", "张老师", "陈老师", "刘老师",
    "赵老师", "周老师", "外教Mr. Brown", "外教Ms. Green",
    "妈妈", "爸爸", "奶奶", "爷爷", "教练叔叔",
]

_SCENARIO_TOPIC_SEEDS: Dict[str, List[str]] = {
    "daily_family": [
        "挑食：只肯吃面不吃菜", "赖床：闹钟响三遍", "洗澡：泡泡弄一地",
        "抢遥控器看动画片", "藏起来的零食被发现", "被说顶嘴后委屈",
        "睡前非要再听一个故事", "发烧量体温不配合", "出游忘带水壶",
        "帮妈妈摆碗筷邀功", "妹妹先拿了玩具", "爸爸答应的奖励没兑现",
        "奶奶做的菜太烫", "雨天不肯穿雨衣", "宠物把拖鞋叼走",
        "空调太冷要盖毯", "拍照不肯笑", "零花钱想买卡包",
        "洗碗打碎盘子", "亲戚小孩来访抢房间", "疫苗打针害怕",
    ],
    "school": [
        "听错题把加号当减号", "红领巾忘带", "体育课假装肚子疼",
        "食堂今天的菜有苦瓜", "放学冲太快撞同学", "作业本落在同桌桌上",
        "被老师点名回答走神", "传纸条被没收", "考试草稿纸不够",
        "值日扫地偷懒", "跳绳比赛数错", "美术课颜料弄手上",
        "科学课观察记录漏写", "小组展示忘词", "课间操站错队",
        "订正作业红笔没水", "眼保健操偷睁眼", "社团课换教室迷路",
        "暴雨停课网课登录失败", "校服拉链卡住", "饮水机排队溅湿鞋",
    ],
    "classroom_interaction": [
        "举手后突然忘答案", "同桌小声提醒被听见", "小组讨论抢话",
        "黑板讲题指错行", "老师追问为什么选B", "课代表催交作业",
        "课堂小游戏输了不服", "读音被纠正‘载’读zài", "互评时说对方字丑",
        "下课铃响还在答", "投影看不清眯眼", "换座位第一天不习惯",
    ],
    "emotions": [
        "积木塔塌了", "贴纸被撕坏", "画被弟弟涂了",
        "糖掉地上脏了", "奖状折了角", "游戏存档没了",
        "被误会先动手", "秘密被说出来", "排队被插队",
        "礼物期待落空", "表演紧张忘动作", "下雨取消春游",
    ],
    "social": [
        "新朋友不敢打招呼", "卡片交换不公平", "秘密约定被破",
        "攀比谁的手表贵", "小团体不让加入", "道歉只说一半",
        "借橡皮不还", "生日没邀请我", "游戏规则临时改",
        "谁当队长争论", "分享零食只给好朋友",
    ],
    "play": [
        "游戏连输三局", "沙堡被踩", "捉迷藏躲衣柜太热",
        "卡牌规则吵起来", "新玩具电池没电", "角色扮演入戏太深",
        "球出界判犯规", "乐高缺关键件", "桌游骰子怀疑作弊",
        "户外探险怕虫子", "假想怪兽太吓人",
    ],
    "learning": [
        "乘法口诀7×8卡壳", "英语th发音练不好", "背课文漏一句",
        "实验步骤跳了一步", "历史年代记混", "天文问黑洞是什么",
        "攀比谁先做完作业", "假装懂被追问露馅", "错题本同一题又错",
        "ü音和u音搞混但正文只写汉字", "作文开头写‘有一天’",
    ],
    "primary_science": [
        "绿豆发芽第几天记录", "磁铁不吸铝箔", "水蒸发杯子变轻",
        "影子中午变短", "月相上弦月辨认", "昆虫足数观察",
        "音叉振动摸不到", "灯泡不亮检查电路", "天气图符号看错",
        "浮沉盐度实验", "食物链草兔狼顺序", "呼吸心跳运动后加快",
    ],
    "morality_life": [
        "排队插队被提醒", "打碎东西要不要说", "扶摔倒同学",
        "红绿灯还剩3秒冲不冲", "水龙头没关紧", "垃圾分类电池放哪",
        "班干部竞选发言", "国旗下讲话忘词", "借书超期还",
        "值日分工不均", "校园欺凌旁观", "劳动课缝扣子手扎",
    ],
    "arts_music": [
        "合唱高音唱破", "节奏拍手错拍", "认不出小号声音",
        "水彩调不出绿色", "剪纸对称剪反", "泥塑鼻子掉了",
        "美术展作品被碰歪", "画国旗星星数量", "给画起名太土",
    ],
    "pe_health": [
        "跳绳双摇失败", "接力棒掉地上", "篮球传球砸脸",
        "立定跳远踩线", "做操手臂幅度被笑", "运动后喘不上气",
        "扭伤脚单脚跳", "比赛输不服规则", "健康课刷牙示范",
    ],
    "info_labor": [
        "打字CapsLock没关", "文件存桌面找不到", "机器人小车轮子卡线",
        "流程图菱形画成方形", "种绿豆忘浇水", "纸桥放砝码塌了",
        "值日拖把太湿", "缝扣子针扎手", "做早餐煎蛋粘锅",
    ],
    "knowledge_chat": [
        "恐龙霸王龙和迅猛龙区别", "木星大红斑是什么", "为什么冬天呼白气",
        "高铁和火车谁快", "春节为什么要贴春联", "诸葛亮草船借箭",
        "向日葵跟着太阳转吗", "微波炉不能放金属", "地铁换乘怎么走",
        "地震要躲哪里", "企鹅为什么不怕冷",
        "火山为什么会喷发", "彩虹几种颜色", "蝙蝠是不是鸟",
        "长城有多长", "月球有没有水", "为什么海水是咸的",
        "红绿灯三个颜色含义", "电梯为什么会动",
    ],
    "fantasy": [
        "变身超级英雄救猫", "魔法咒语让作业消失失败", "火箭去火星忘带氧气",
        "恐龙当宠物要喂什么", "童话改编：灰姑娘穿运动鞋",
        "超能力：听懂植物说话", "梦境：在学校飞起来", "假想朋友叫星星",
    ],
    "body_senses": [
        "饿得肚子咕咕叫", "吃太撑打嗝", "发烧额头烫",
        "膝盖擦破渗血", "犯困眼皮打架", "痒痒忍不住笑",
        "闻到垃圾桶臭味", "鞭炮巨响捂耳朵", "晕车想开窗",
    ],
    "requests": [
        "连问五个为什么", "反复求再玩10分钟", "讨价还价多一块糖",
        "撒娇抱腿", "威胁不吃饭", "紧急找厕所",
        "礼貌说请和谢谢", "假装可怜眨眼睛", "得寸进尺要第二个",
    ],
    "stress_numbers": [
        "报家里座机区号不同", "报门牌带字母", "念农历和公历日期",
        "倒数从20到1", "念带小数的价格", "念电话号码后四位不同",
    ],
    "stress_tonguetwister": [
        "四是四十是十", "吃葡萄不吐葡萄皮", "She sells seashells变体",
        "中英混合：四是four", "四是四快速三遍",
    ],
    "stress_emotion_shift": [
        "开心→发现弄错→生气", "害怕→原来是惊喜→笑",
        "无聊→突然宣布春游→兴奋", "生气→被哄→撒娇",
    ],
    "stress_repetition": [
        "紧张时我我我", "兴奋时好好好好", "撒娇时嘛嘛嘛",
        "解释题时那那那个", "快速连珠炮不停顿",
    ],
    "stress_whisper_shout": [
        "悄悄说秘密代号", "走廊远处喊妈妈", "压低声音威胁不许说",
        "突然惊吓啊一声", "耳语传话传歪了",
    ],
}


def _pick_shuffled_pool(
    pool: List[str],
    batch_size: int,
    seed_text: str,
    seed_suffix: str = "pool",
) -> List[str]:
    """Shuffle pool without replacement when long enough; else bag-resample."""
    if not pool:
        return [""] * batch_size
    if len(pool) >= batch_size:
        rng = random.Random(hashlib.md5(f"{seed_suffix}|{seed_text}".encode()).hexdigest())
        bag = list(pool)
        rng.shuffle(bag)
        return bag[:batch_size]
    rng = random.Random(hashlib.md5(seed_text.encode()).hexdigest())
    return _draw_diverse_axis_list(pool, batch_size, rng)


def build_batch_forbidden_openings(task_id: Optional[int], batch_size: int) -> str:
    """Per-batch list of overused sentence skeletons to avoid."""
    seed = f"forbid_open|task={task_id}|n={batch_size}"
    n = min(8, batch_size + 2)
    picks = _pick_shuffled_pool(_ANTI_OPENING_TEMPLATE_POOL, n, seed, "anti_open")
    return (
        "本 batch 禁止作为主句开头或骨架（可用其它说法替代）: "
        + "、".join(picks)
    )


def build_global_batch_diversity_manifest(
    scenario_key: str,
    batch_size: int,
    task_id: Optional[int] = None,
) -> str:
    """Batch-level creative challenges applied across rows."""
    seed = f"manifest|{scenario_key}|task={task_id}|n={batch_size}"
    rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
    n_challenges = min(5, max(3, batch_size // 2))
    challenges = _pick_shuffled_pool(
        _BATCH_CREATIVE_CHALLENGE_POOL, n_challenges, seed, "challenges"
    )
    lines = "\n".join(f"- {c}" for c in challenges)
    return f"""
=== 本 batch 创意挑战（分散到不同条目，必须落实） ===
{lines}
"""


def build_reading_diversity_instructions(
    scenario_key: str,
    subscene: str,
    batch_size: int,
    task_id: Optional[int] = None,
) -> str:
    """Per-row diversity plan for reading / follow-read / read-aloud tasks."""
    axes = _READING_DIVERSITY_AXES.get(scenario_key)
    if not axes:
        return ""

    seed_text = f"reading|{scenario_key}|{subscene}|task={task_id}"
    rng = random.Random(hashlib.md5(seed_text.encode()).hexdigest())

    modes = _draw_diverse_axis_list(axes["modes"], batch_size, rng)
    materials = _draw_diverse_axis_list(axes["materials"], batch_size, rng)
    settings = _draw_diverse_axis_list(axes["settings"], batch_size, rng)
    stumbles = _draw_diverse_axis_list(axes["stumbles"], batch_size, rng)
    openings = _draw_diverse_axis_list(
        axes.get("openings") or _READING_OPENING_HINTS, batch_size, rng
    )
    names = _pick_shuffled_pool(_CHILD_NAME_POOL, batch_size, seed_text, "reading_names")
    numeric_hints = _pick_shuffled_pool(
        _READING_NUMERIC_HINTS, batch_size, seed_text, "reading_nums"
    )
    twists = _draw_diverse_axis_list(_CREATIVE_TWIST_POOL, batch_size, rng)
    registers = _draw_diverse_axis_list(_READING_REGISTER_HINTS, batch_size, rng)
    eras = _draw_diverse_axis_list(_READING_ERA_HINTS, batch_size, rng)
    adults = _pick_shuffled_pool(_ADULT_NAME_POOL, batch_size, seed_text, "reading_adults")
    mat_syntax = _pick_shuffled_pool(
        _READING_MATERIAL_SENTENCE_MIX, batch_size, seed_text, "reading_mat_syn"
    )
    syn_sent, syn_subj, syn_obj, syn_verb, syn_conn = draw_syntactic_axes(
        "cn_mostly" if scenario_key == "primary_chinese" else (
            "pure_en" if scenario_key == "primary_english" else "pure_cn"
        ),
        batch_size,
        f"{seed_text}|reading_syntax",
    )

    topic_pool = _READING_TOPIC_SEEDS.get(scenario_key, [])
    topic_seeds = _pick_shuffled_pool(
        topic_pool or [subscene], batch_size, seed_text, "reading_topic"
    )

    rows = []
    for i in range(batch_size):
        prev_note = ""
        if i > 0:
            prev_note = (
                f"; 与第{i}条在篇目/开头/数字/人名/语体/句式至少5项不同"
            )
        rows.append(
            f"{i + 1}. 【必用主题】{topic_seeds[i]}; 朗读方式={modes[i]}; "
            f"材料类型={materials[i]}; 场景={settings[i]}; 卡顿={stumbles[i]}; "
            f"开头方式={openings[i]}; {numeric_hints[i]}; 语体={registers[i]}; "
            f"时代/文体感={eras[i]}; 情节感={twists[i]}; "
            f"材料句式={mat_syntax[i]}; 句式={syn_sent[i]}; 主语={syn_subj[i]}; "
            f"宾语={syn_obj[i]}; 动词={syn_verb[i]}; 连接={syn_conn[i]}; "
            f"朗读者={names[i]}; 文中大人={adults[i]}(禁小明/小红/小刚){prev_note}; "
            f"子场景={subscene}"
        )

    min_modes = min(5, batch_size)
    min_topics = min(8, batch_size)
    min_materials = min(6, batch_size)
    forbidden = build_batch_forbidden_openings(task_id, batch_size)
    syntax_rules = build_syntactic_diversity_rules(
        "pure_en" if scenario_key == "primary_english" else "pure_cn",
        batch_size,
        is_reading=True,
    )

    return f"""
=== 跟读/读题超高多样性（必须遵守） ===
{syntax_rules}
这 {batch_size} 条必须是不同的课文片段、不同的题目、不同的朗读方式，禁止换汤不换药。
- 至少 {min_modes} 种「朗读方式」、{min_topics} 个不同「必用主题」、{min_materials} 种「材料类型」
- 任意两条不得相同：篇名/诗人/Dialog模板/应用题类型/开头前6字/朗读者名/语体
- {forbidden}
- 每条必须把「必用主题」写成可朗读的正文片段（≥2句或≥1长句），不能只写标题
- 朗读正文禁止附带拼音、注音符号、…/· 等注音标点；只写要念出来的汉字/英文词（数学∠△°除外）
- 数字/单位/地名/人名每条必须不同；跟读、读题、朗读、背诵、默写要交替出现

逐条朗读差异计划（逐条执行，禁止忽略必用主题）：
{chr(10).join(rows)}
"""


def build_reading_aloud_instructions(
    scenario_key: str,
    subscene: str,
    lang_key: str,
    length_key: str,
    age_tier: str,
    batch_size: int,
    task_id: Optional[int] = None,
) -> str:
    """Extra guidance for textbook / exam-item reading aloud."""
    is_reading = scenario_key in _READING_ALOUD_SCENARIOS or any(
        hint in subscene for hint in _READING_SUBSCENE_HINTS
    )
    if not is_reading:
        return ""

    is_advanced_subscene = "拔高" in subscene
    subject_cfg = _READING_DIFFICULTY_BY_SUBJECT.get(scenario_key)

    if age_tier == "preschool":
        age_reading_note = (
            "当前为幼儿园段：即使在本场景，也不要使用初一初二难度；"
            "仅朗读极短儿歌、三字句、简单古诗两句。"
        )
        advanced_quota = "本 batch 不要生成拔高/初一初二难度文本。"
    elif age_tier == "early_elem":
        age_reading_note = (
            "当前为小学低年级：拔高难度最多占 30%，其余为小学中高年级朗读；"
            "拔高条也要读得磕磕绊绊，不要像老师播音。"
        )
        advanced_quota = f"本 batch 约 {max(1, batch_size // 3)} 条可为接近初一难度，其余为小学中高年级。"
    else:
        age_reading_note = (
            "当前为小学高年级：允许接近初一、初二的课文/试题难度，"
            "但必须保持儿童朗读感（卡顿、改口、问老师、读错术语）。"
        )
        advanced_quota = subject_cfg["ratio_note"] if subject_cfg else (
            f"本 batch 至少 {max(1, (batch_size * 3) // 5)} 条为接近初一至初二难度的朗读/读题。"
        )

    length_note = ""
    if length_key in ("long", "very_long"):
        length_note = "当前长度偏长：务必输出完整长段课文或整道大题题干，不要只读一句标题。"
    elif length_key in ("short", "ultra_short") and scenario_key in _READING_DIFFICULTY_BY_SUBJECT:
        length_note = (
            "当前长度虽短，也要嵌入较难词句/公式/文言字；"
            "优先读拔高题中的一个关键长分句，不要只读“3+5”。"
        )

    if subject_cfg:
        topics = subject_cfg["advanced_topics"]
        en_pool = subject_cfg.get("en_examples") or ()
        cn_pool = subject_cfg.get("cn_examples") or ()
        if scenario_key == "primary_english":
            examples = "\n".join(f"- {e}" for e in en_pool)
            lang_note = "English lesson/exam wording at junior-high lower-bound difficulty."
        elif lang_key == "pure_en":
            ex = en_pool or cn_pool
            examples = "\n".join(f"- {e}" for e in ex)
            lang_note = (
                "English child speech; read lesson/exam content in English where natural."
            )
        elif lang_key in ("en_mostly", "frequent_mix") and scenario_key == "primary_english":
            examples = "\n".join(f"- {e}" for e in en_pool[:3])
            lang_note = "Mix junior-high English reading with brief Chinese interjections."
        else:
            examples = "\n".join(f"- {e}" for e in cn_pool)
            lang_note = "Chinese lesson/exam wording at junior-high lower-bound difficulty (初一至初二)."
        bad_examples = "\n".join(f"- {e}" for e in subject_cfg["bad_examples"])
    else:
        topics = "古诗、课文、阅读题干"
        examples = "- \"床前明月光，疑是地上霜[surprise-oh] 举头望明月\""
        bad_examples = "- \"我不想读课文\""
        lang_note = "Include concrete text being read aloud."

    advanced_hint = ""
    if is_advanced_subscene:
        advanced_hint = "本子场景为【拔高】：必须按初一至初二难度出题面/课文，禁止降回小学一二年级内容。"

    reading_diversity = ""
    if scenario_key in _READING_DIVERSITY_AXES:
        reading_diversity = build_reading_diversity_instructions(
            scenario_key, subscene, batch_size, task_id=task_id
        )

    # Shuffle example pool so batches don't always see the same 4 lines
    if subject_cfg:
        if scenario_key == "primary_english":
            ex_pool = list(en_pool or cn_pool)
        elif lang_key == "pure_en":
            ex_pool = list(en_pool or cn_pool)
        else:
            ex_pool = list(cn_pool)
        if len(ex_pool) > 3:
            ex_rng = random.Random(f"{subscene}|{length_key}|{batch_size}")
            ex_rng.shuffle(ex_pool)
            examples = "\n".join(f"- {e}" for e in ex_pool[: min(6, len(ex_pool))])

    return f"""
=== 朗读课文 / 读题要求（本场景重点，必须遵守） ===
{lang_note}
{age_reading_note}
{advanced_quota}
{advanced_hint}
{length_note}
{reading_diversity}
- 拔高难度知识范围: {topics}
- 每条必须是在「读」正文（课文/古诗/英文段落/数学题题干），不要只描述“我要读书了”。
- 允许：漏字、读错音、术语念错、读一半改口、重复上一句、问老师怎么念。
- 好例子（本 batch 可参考不同风格，勿照抄同一条）:
{examples}
- 坏例子:
{bad_examples}
"""


def build_stress_instructions(scenario_key: str, subscene: str, batch_size: int = 10) -> str:
    if scenario_key == "stress_numbers":
        return f"""
=== 压力测试：数字序列 ===
孩子正在: {subscene}。这个 batch 的 {batch_size} 条文本必须覆盖至少 3 种不同的数字/时间类型，禁止全部同一种。

可用类型（每个类型最多出现 3 条）：
- 电话号码: "我的电话是15987654321", "Call me at 555-0199"
- 门牌号/楼层: "我家住302室", "We live on the 15th floor"
- 日期时间: "今天是6月1号", "My birthday is July 4th"
- 数数/序列: "我会数1、2、3、4、5", "Count with me: ten, twenty, thirty"
- 数学算式: "3加5等于8", "Two plus two is four"
- 价格金额: "这块糖5块钱", "This toy costs 12 dollars and 99 cents"

要求:
- 不要全部用 138 开头，电话号码要换不同的号段（159、186、177 等）
- 数字要自然嵌入孩子的对话，不要干巴巴只念数字
- 中英文都混合出现
"""
    elif scenario_key == "stress_tonguetwister":
        return f"""
=== 压力测试：绕口令 ===
生成适合孩子年龄的绕口令。孩子正在: {subscene}
例子: "四是四，十是十", "She sells sea shells by the sea shore"
注意：要真的有难度，但不要太长
"""
    elif scenario_key == "stress_emotion_shift":
        return f"""
=== 压力测试：情绪突变 ===
文本中要展示 CLEAR 的情绪转变。转变: {subscene}
用标签标记转变点。好例子:
"耶我考了100分[laughter]，等等，这是小明的卷子[surprise-ah][sigh]"
"[sigh]又要上课了，哇！今天有春游[surprise-wa][laughter]"
"""
    elif scenario_key == "stress_repetition":
        return f"""
=== 压力测试：重复与口吃 ===
孩子正在: {subscene}
包含自然的重复: "我我我想先说完"、"那那那个题我会做"
注意：真实孩子的口吃不是每句话都结巴，只在紧张/兴奋时
"""
    elif scenario_key == "stress_whisper_shout":
        return f"""
=== 压力测试：耳语与大喊 ===
孩子正在: {subscene}
混合悄悄话和突然大喊。用标签和标点表示音量变化
"""
    return ""


def _draw_diverse_axis_list(options: List[str], batch_size: int, rng: random.Random) -> List[str]:
    """Sample axis values per row; reshuffle bag when exhausted (better than i % len)."""
    if not options:
        return [""] * batch_size
    bag = list(options)
    rng.shuffle(bag)
    picked: List[str] = []
    for _ in range(batch_size):
        if not bag:
            bag = list(options)
            rng.shuffle(bag)
        picked.append(bag.pop())
    for i in range(1, len(picked)):
        if picked[i] == picked[i - 1] and len(options) > 1:
            swap_pool = [x for x in options if x != picked[i]]
            picked[i] = rng.choice(swap_pool)
    return picked


def build_diversity_instructions(
    scenario_key: str,
    subscene: str,
    emotion: str,
    age_tier: str,
    lang_key: str,
    batch_size: int,
    task_id: Optional[int] = None,
) -> str:
    """Force variety inside a single batch, where most repetition is introduced."""
    micro_contexts = list(MICRO_CONTEXTS.get(scenario_key, []))
    if scenario_key in _READING_DIVERSITY_AXES:
        micro_contexts = micro_contexts + [
            "本篇课文换标题/作者/朝代",
            "本题换数字人物地名",
            "换朗读方式：跟读/指读/读题/朗读",
            "换场景：课堂/在家/小组/考试",
        ]
    child_profiles = list(CHILD_PROFILES.get(age_tier, []))
    speech_errors = list(SPEECH_ERROR_PATTERNS.get(lang_key, []))

    seed_text = f"{scenario_key}|{subscene}|{emotion}|{age_tier}|{lang_key}|task={task_id}"
    rng = random.Random(hashlib.md5(seed_text.encode()).hexdigest())

    openings = _draw_diverse_axis_list(list(OPENING_STYLES), batch_size, rng)
    focuses = _draw_diverse_axis_list(list(FOCUS_ANGLES), batch_size, rng)
    patterns = _draw_diverse_axis_list(list(SPEECH_PATTERNS), batch_size, rng)
    micros = _draw_diverse_axis_list(micro_contexts or [subscene], batch_size, rng)
    profiles = _draw_diverse_axis_list(child_profiles or [age_tier], batch_size, rng)
    errors = _draw_diverse_axis_list(
        speech_errors or ["自然口语，允许儿童不完美但句意完整"],
        batch_size,
        rng,
    )
    dialogues = _draw_diverse_axis_list(list(DIALOGUE_STATES), batch_size, rng)
    child_names = _pick_shuffled_pool(_CHILD_NAME_POOL, batch_size, seed_text, "child_names")
    places = _pick_shuffled_pool(_PLACE_ANCHOR_POOL, batch_size, seed_text, "places")
    props = _pick_shuffled_pool(_PROP_ANCHOR_POOL, batch_size, seed_text, "props")
    twists = _draw_diverse_axis_list(_CREATIVE_TWIST_POOL, batch_size, rng)
    audiences = _pick_shuffled_pool(_AUDIENCE_POOL, batch_size, seed_text, "audiences")
    times = _pick_shuffled_pool(_TIME_ANCHOR_POOL, batch_size, seed_text, "times")
    sensories = _draw_diverse_axis_list(_SENSORY_DETAIL_POOL, batch_size, rng)
    adults = _pick_shuffled_pool(_ADULT_NAME_POOL, batch_size, seed_text, "adults")
    syn_sent, syn_subj, syn_obj, syn_verb, syn_conn = draw_syntactic_axes(
        lang_key, batch_size, seed_text
    )

    is_reading = scenario_key in _READING_DIVERSITY_AXES
    topic_pool = _SCENARIO_TOPIC_SEEDS.get(scenario_key, _GENERIC_TOPIC_FALLBACK)
    if is_reading:
        topic_seeds = [""] * batch_size
    else:
        topic_seeds = _pick_shuffled_pool(
            topic_pool, batch_size, seed_text, f"scenario_topic|{scenario_key}"
        )

    preschool_axis = ""
    if age_tier == "preschool":
        preschool_axis = (
            "本 batch 为幼儿园段：每条必须含叠词/三连重复/第三人称自称/发音不准写法至少一项；"
            "禁止 no cap、emo 等青少年网络用语。"
        )

    min_openings = min(8, batch_size)
    min_focus = min(8, batch_size)
    min_topics = min(8, batch_size) if not is_reading else 0
    forbidden = build_batch_forbidden_openings(task_id, batch_size)
    manifest = build_global_batch_diversity_manifest(scenario_key, batch_size, task_id)

    rows = []
    for i in range(batch_size):
        topic_part = ""
        if topic_seeds[i]:
            topic_part = f"必用切入={topic_seeds[i]}; "
        prev_note = ""
        if i > 0:
            prev_note = (
                f"; 与第{i}条在开头/主人公/物品/句式/主语/动词至少4项不同"
            )
        rows.append(
            f"{i + 1}. {topic_part}"
            f"句式={syn_sent[i]}; 主语={syn_subj[i]}; 宾语={syn_obj[i]}; "
            f"动词={syn_verb[i]}; 连接={syn_conn[i]}; "
            f"开头方式={openings[i]}; "
            f"关注点={focuses[i]}; "
            f"微场景={micros[i]}; "
            f"地点={places[i]}; "
            f"物品={props[i]}; "
            f"时间={times[i]}; "
            f"说话对象={audiences[i]}; "
            f"感官细节={sensories[i]}; "
            f"情节转折={twists[i]}; "
            f"儿童画像={profiles[i]}; "
            f"错误/不完美={errors[i]}; "
            f"对话状态={dialogues[i]}; "
            f"口语模式={patterns[i]}; "
            f"主人公={child_names[i]}; 在场大人={adults[i]}"
            f"(禁小明/小红/小刚){prev_note}"
        )

    topic_rule = ""
    if not is_reading and min_topics:
        topic_rule = (
            f"- 至少 {min_topics} 条落实不同「必用切入」，写入正文而非只写关键词\n"
        )

    syntax_rules = build_syntactic_diversity_rules(lang_key, batch_size, is_reading=is_reading)

    return f"""
=== 批内超高多样性要求（必须遵守） ===
{manifest}
{syntax_rules}
{preschool_axis}
这 {batch_size} 条不能只是同一意思换说法。任意两条不得共享：开头前4字 + 主人公 + 核心物品 + 句式 + 主语类型。
- 至少 {min_openings} 种开头方式、{min_focus} 种关注点；主人公名 batch 内不得重复
{topic_rule}- {forbidden}
- 禁止超过 2 条同一叙事模板；不要每条都像作文背景
- 每条完整可朗读；年龄匹配的不完美感即可

逐条差异计划（逐条执行）：
{chr(10).join(rows)}
"""


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str = "",
    api_key: Optional[str] = None,
    base_url: str = "",
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    max_tokens: int = 8192,
    temperature: float = 0.85,
) -> List[Dict]:
    import time
    import urllib.error
    import urllib.request

    resolved_api_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    resolved_model = model or os.environ.get("LLM_MODEL", "")
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL", "")

    def _extract_json(raw_text: str) -> List[Dict]:
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
            text = text[start : end + 1]
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # Fallback: regex extract
        objects = []
        pattern = r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"(\w+)"\s*:\s*"([^"]*)")*\s*\}'
        for m in re.finditer(pattern, text, re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if "text" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                pass
        return objects

    def _call_openai_compatible() -> str:
        endpoint = resolved_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resolved_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {body}") from e

        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    def _call_anthropic_compatible() -> str:
        import anthropic

        client = anthropic.Anthropic(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_headers={"User-Agent": "Claude-Code/0.1.0"},
        )
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
        return ""

    use_anthropic_protocol = "/anthropic" in resolved_base_url.rstrip("/")
    last_error = None
    for attempt in range(max_retries):
        try:
            content = _call_anthropic_compatible() if use_anthropic_protocol else _call_openai_compatible()
            if not content:
                return []
            return _extract_json(content)
        except Exception as e:
            last_error = e
            wait = retry_base_delay * (2 ** attempt)
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait:.1f}s...")
            time.sleep(wait)

    print(f"API failed after {max_retries} attempts: {last_error}")
    return []


# ---------------------------------------------------------------------------
# Task Scheduler
# ---------------------------------------------------------------------------

def _weighted_choice(options: Dict[str, float], rng: random.Random) -> str:
    """Exact weighted sampling using random.choices."""
    keys = list(options.keys())
    weights = list(options.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def apply_config_from_env(config: GenConfig) -> GenConfig:
    """Load .env and override GenConfig from environment variables (optional)."""
    load_env_file()
    if os.environ.get("LLM_MODEL"):
        config.model = os.environ["LLM_MODEL"]
    if os.environ.get("LLM_API_KEY"):
        config.api_key = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_BASE_URL"):
        config.base_url = os.environ["LLM_BASE_URL"]
    if os.environ.get("GEN_MODEL"):
        config.model = os.environ["GEN_MODEL"]
    if os.environ.get("GEN_SEED"):
        config.seed = int(os.environ["GEN_SEED"])
    if os.environ.get("GEN_OUTPUT_DIR"):
        config.output_dir = os.environ["GEN_OUTPUT_DIR"]
    if os.environ.get("GEN_SEMANTIC_DEDUP_THRESHOLD"):
        config.semantic_dedup_threshold = float(os.environ["GEN_SEMANTIC_DEDUP_THRESHOLD"])
    if os.environ.get("GEN_TOTAL_TARGET"):
        config.total_target = int(os.environ["GEN_TOTAL_TARGET"])
    if os.environ.get("GEN_BATCH_SIZE"):
        config.batch_size = int(os.environ["GEN_BATCH_SIZE"])
    if os.environ.get("GEN_MAX_WORKERS"):
        config.max_workers = int(os.environ["GEN_MAX_WORKERS"])
    return config


def _pick_subscene(
    scenario_key: str,
    scenario: Dict,
    rng: random.Random,
    reading_subscene_pools: Optional[Dict[str, List[str]]] = None,
    reading_subscene_idx: Optional[Dict[str, int]] = None,
) -> str:
    """Rotate reading subscenes for even coverage; mix 拔高/基础 pools."""
    subscenes = scenario["subscenes"]
    if scenario_key in _READING_DIFFICULTY_BY_SUBJECT and reading_subscene_pools and reading_subscene_idx is not None:
        pool = reading_subscene_pools.get(scenario_key) or subscenes
        idx = reading_subscene_idx.get(scenario_key, 0)
        subscene = pool[idx % len(pool)]
        reading_subscene_idx[scenario_key] = idx + 1
        return subscene

    if scenario_key in _READING_DIFFICULTY_BY_SUBJECT:
        advanced = [s for s in subscenes if s.startswith("拔高")]
        basic = [s for s in subscenes if not s.startswith("拔高")]
        if advanced and basic:
            pool = advanced if rng.random() < 0.55 else basic
            return rng.choice(pool)
        if advanced:
            return rng.choice(advanced)
    return rng.choice(subscenes)


def _init_reading_subscene_pools(rng: random.Random) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Shuffle per-subject subscene order once per task-list generation."""
    pools: Dict[str, List[str]] = {}
    idx: Dict[str, int] = {}
    for key in _READING_DIVERSITY_AXES:
        if key not in SCENARIOS:
            continue
        pool = list(SCENARIOS[key]["subscenes"])
        rng.shuffle(pool)
        pools[key] = pool
        idx[key] = 0
    return pools, idx


def generate_task_list(config: GenConfig) -> List[Dict]:
    rng = random.Random(config.seed)
    tasks = []
    total_batches = config.total_target // config.batch_size

    regular_scenarios = {k: v for k, v in SCENARIOS.items() if not v.get("is_stress_test", False)}
    regular_keys = list(regular_scenarios.keys())

    stress_scenarios = {k: v for k, v in SCENARIOS.items() if v.get("is_stress_test", False)}
    stress_keys = list(stress_scenarios.keys())

    num_stress = int(total_batches * config.stress_test_ratio)

    # Track subscene usage for stress tests to ensure variety
    stress_subscene_idx = {k: 0 for k in stress_keys}
    reading_subscene_pools, reading_subscene_idx = _init_reading_subscene_pools(rng)

    for i in range(total_batches):
        if i < num_stress and stress_keys:
            scenario_key = rng.choice(stress_keys)
        else:
            scenario_weights = {
                key: config.scenario_distribution.get(key, 1.0)
                for key in regular_keys
            }
            scenario_key = _weighted_choice(scenario_weights, rng)

        scenario = SCENARIOS[scenario_key]
        # For stress tests, cycle through subscenes to avoid repetition
        if scenario.get("is_stress_test", False):
            subscenes = scenario["subscenes"]
            idx = stress_subscene_idx[scenario_key] % len(subscenes)
            subscene = subscenes[idx]
            stress_subscene_idx[scenario_key] += 1
        else:
            subscene = _pick_subscene(
                scenario_key,
                scenario,
                rng,
                reading_subscene_pools,
                reading_subscene_idx,
            )

        # Emotion weighted by scenario, fallback to uniform if missing
        emotion_weights = scenario.get("typical_emotions")
        if emotion_weights:
            emotion = _weighted_choice(emotion_weights, rng)
        else:
            emotion = rng.choice(EMOTIONS)

        tasks.append({
            "task_id": i,
            "scenario_key": scenario_key,
            "subscene": subscene,
            "length_key": _weighted_choice(config.length_distribution, rng),
            "lang_key": _weighted_choice(config.lang_mix_distribution, rng),
            "emotion": emotion,
            "age_tier": _weighted_choice(config.age_distribution, rng),
        })

    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["task_id"] = i

    return tasks


def worker(task: Dict, config: GenConfig) -> List[Dict]:
    prompt = build_prompt(
        task["scenario_key"],
        task["subscene"],
        task["length_key"],
        task["lang_key"],
        task["emotion"],
        task["age_tier"],
        config.batch_size,
        task.get("suppression_hint", ""),
        task.get("task_id"),
    )
    temp_rng = random.Random(
        hashlib.md5(f"temp|{task.get('task_id')}|{task.get('scenario_key')}".encode()).hexdigest()
    )
    batch_temperature = min(1.0, max(0.76, config.temperature + temp_rng.uniform(-0.08, 0.15)))

    results = call_llm(
        prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        max_tokens=config.max_tokens,
        temperature=batch_temperature,
    )
    for item in results:
        item["task_id"] = task["task_id"]
        for k, v in task.items():
            if k == "suppression_hint":
                continue
            if k not in item:
                item[k] = v
    if config.generate_text_tn:
        try:
            from .text_tn import attach_text_tn_batch
        except ImportError:
            from text_tn import attach_text_tn_batch

        attach_text_tn_batch(results)
    return results


# ---------------------------------------------------------------------------
# Post-processing v3: Tag Validation & Correction
# ---------------------------------------------------------------------------

def _count_tags(text: str) -> Tuple[int, List[str]]:
    pattern = re.compile(rf"\[({VALID_TAG_NAMES})\]")
    tags = pattern.findall(text)
    return len(tags), tags


def _validate_tag_position(text: str) -> List[str]:
    """Check if tag positions are reasonable."""
    issues = []
    tag_pattern = re.compile(rf"(\[({VALID_TAG_NAMES})\])")

    # Tags should not be at the very beginning unless they are sigh/question/surprise-ah
    allowed_start_tags = {"sigh", "question-ah", "question-oh", "question-ei",
                          "question-yi", "surprise-ah", "surprise-oh", "surprise-wa"}

    for match in tag_pattern.finditer(text):
        tag_full = match.group(1)
        tag_name = match.group(2)
        start_pos = match.start()

        # Check if tag is at the very start
        if start_pos == 0 and tag_name not in allowed_start_tags:
            issues.append(f"{tag_full} should not be at sentence start")

        # laughter should not be at start
        if tag_name == "laughter" and start_pos < 3:
            issues.append(f"{tag_full} should not be near start")

    return issues


def _validate_tag_combinations(text: str) -> List[str]:
    """Check for contradictory tag combinations."""
    issues = []
    _, tags = _count_tags(text)

    tag_set = set(tags)

    # Contradictory combinations
    if "laughter" in tag_set and "sigh" in tag_set:
        issues.append("laughter and sigh contradict")

    # Too many similar tags
    surprise_tags = [t for t in tags if t.startswith("surprise-")]
    if len(set(surprise_tags)) >= 3:
        issues.append(f"too many surprise variants: {surprise_tags}")

    question_tags = [t for t in tags if t.startswith("question-")]
    if len(set(question_tags)) >= 3:
        issues.append(f"too many question variants: {question_tags}")

    return issues


def _auto_correct_tags(text: str, _emotion: str = "") -> str:
    """Auto-correct common tag issues."""

    # Fix tag spacing: remove space before tag, ensure space after
    tag_pattern = re.compile(rf"\s*(\[({VALID_TAG_NAMES})\])\s*")

    def fix_spacing(m):
        tag = m.group(1)
        # Check if next char is punctuation
        next_char = m.string[m.end():m.end()+1] if m.end() < len(m.string) else ""
        if next_char and next_char in "，。？！,.?!":
            return tag
        return tag + " "

    text = tag_pattern.sub(fix_spacing, text)

    # Fix multiple consecutive spaces
    text = re.sub(r"  +", " ", text)

    return text.strip()


def _semantic_similarity(a: str, b: str) -> float:
    """Lightweight semantic similarity using char/word overlap."""
    # Normalize
    a_norm = re.sub(r"[\s\[\]，。！？,.?!]", "", a.lower())
    b_norm = re.sub(r"[\s\[\]，。！？,.?!]", "", b.lower())
    if not a_norm or not b_norm:
        return 0.0

    # For Chinese-heavy text, use character bigrams; for English, use words
    def _tokens(s: str):
        # Mix of char bigrams and words
        chars = [s[i:i+2] for i in range(len(s)-1)]
        words = re.findall(r"[a-zA-Z]+", s)
        return set(chars + words)

    ta = _tokens(a_norm)
    tb = _tokens(b_norm)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def semantic_deduplicate(texts: List[Dict], threshold: float = 0.88) -> List[Dict]:
    """Remove semantically near-duplicate texts within the same generation context."""
    unique: List[Dict] = []
    context_buckets: Dict[Tuple[str, str, str, str, str, str], List[Dict]] = {}
    for item in texts:
        text = item.get("text", "").strip()
        if not text:
            continue
        key = _duplicate_context_key(item)
        bucket = context_buckets.get(key, [])
        is_dup = False
        for existing in bucket:
            sim = _semantic_similarity(text, existing.get("text", ""))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(item)
            bucket.append(item)
            context_buckets[key] = bucket
    return unique


def _truncate_to_length(text: str, length_type: str) -> str:
    """Intelligently truncate text to fit declared length."""
    import re

    # Extract tags to preserve them
    tag_pattern = re.compile(r'(\[[^\]]+\])')
    parts = []
    last_end = 0
    tags_positions = []
    for m in tag_pattern.finditer(text):
        if m.start() > last_end:
            parts.append(text[last_end:m.start()])
        tags_positions.append((m.start(), m.end(), m.group(1)))
        last_end = m.end()
    if last_end < len(text):
        parts.append(text[last_end:])

    clean_text = ''.join(parts).strip()

    # Count
    cjk_chars = len(re.findall(r'[一-鿿]', clean_text))
    en_words = len(re.findall(r'[a-zA-Z]+', clean_text))
    total = cjk_chars + en_words

    limits = {
        'ultra_short': 5,
        'short': 8,
        'medium': 15,
        'long': 25,
        'very_long': 40,
    }
    max_len = limits.get(length_type, 999)

    if total <= max_len:
        return text

    # Need to truncate.
    if cjk_chars > 0 and en_words == 0:
        # Pure Chinese: truncate by character count
        result = clean_text[:max_len]
        result = result.rstrip('，,、 ')
        if not result.endswith(('...', '…', '！', '。', '?', '!', '～', '~')):
            result = result + '...'
    elif en_words > 0 and cjk_chars == 0:
        # Pure English: truncate by word count
        max_words = 3 if length_type == 'ultra_short' else (6 if length_type == 'short' else max_len)
        # Split preserving words
        tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[^a-zA-Z\s]+", clean_text)
        kept = tokens[:max_words]
        result = ' '.join(kept)
        result = result.rstrip(' ,')
        if not result.endswith(('...', '…', '!', '?', '.')):
            result = result + '...'
    else:
        # Mixed: truncate by character count
        result = clean_text[:max_len]
        result = result.rstrip('，,、 ')
        if not result.endswith(('...', '…', '！', '。', '?', '!', '～', '~')):
            result = result + '...'

    # Re-insert up to 2 tags at the end
    if tags_positions:
        kept_tags = [tag for _, _, tag in tags_positions[:2]]
        result = result + ''.join(kept_tags)

    return result


_LENGTH_BOUNDS = {
    "ultra_short": (1, 5),
    "short": (3, 10),
    "medium": (8, 18),
    "long": (14, 30),
    "very_long": (22, 50),
}


def _count_speech_units(text: str) -> int:
    clean = re.sub(r"\[[^\]]+\]", "", text).strip()
    cjk_chars = len(re.findall(r"[一-鿿]", clean))
    en_words = len(re.findall(r"[a-zA-Z]+", clean))
    return cjk_chars + en_words


def _validate_length(text: str, length_type: str) -> bool:
    """Rough length validation."""
    total = _count_speech_units(text)
    lo, hi = _LENGTH_BOUNDS.get(length_type, (0, 999))
    return lo <= total <= hi


def _is_reading_item(item: Dict) -> bool:
    sk = item.get("scenario") or item.get("scenario_key") or ""
    sub = item.get("subscene") or ""
    return sk in _READING_ALOUD_SCENARIOS or any(h in sub for h in _READING_SUBSCENE_HINTS)


def _min_required_tags_for_item(item: Dict) -> int:
    """Reading/long passages often omit tags; do not over-reject raw LLM batches."""
    length_type = item.get("length_type", "medium")
    base = _min_required_tags(length_type)
    if _is_reading_item(item):
        if length_type in ("long", "very_long", "medium"):
            return 0
        return min(base, 1)
    return base


def _max_validation_issues_for_item(item: Dict) -> int:
    return 6 if _is_reading_item(item) else 5


def _is_severe_length_mismatch(
    text: str,
    length_type: str,
    item: Optional[Dict] = None,
) -> bool:
    """Reject only extreme length violations (LLM approximate control)."""
    total = _count_speech_units(text)
    lo, hi = _LENGTH_BOUNDS.get(length_type, (0, 999))
    if lo <= 0:
        return False
    is_reading = item is not None and _is_reading_item(item)
    lo_ratio = 0.25 if is_reading else 0.35
    if is_reading and length_type in ("medium", "long", "very_long"):
        hi_mult, hi_extra = 4.5, 14
    else:
        hi_mult, hi_extra = 3.0, 10
    if total < max(1, int(lo * lo_ratio)):
        return True
    if total > int(hi * hi_mult) + hi_extra:
        return True
    return False


def diagnose_quality_rejections(texts: List[Dict]) -> Counter:
    """Count first-failure reason per item (for tuning quality_filter)."""
    reasons: Counter = Counter()
    for item in texts:
        text = item.get("text", "").strip()
        if not text:
            reasons["empty"] += 1
            continue
        lower = text.lower()
        bad_markers = [
            "股票", "投资", "政治", "战争", "sex", "kill", "die", "porn",
            "no cap", "cap,", " emo", "emo了", "绝绝子", "yyds",
            # English fillers not in ASR vocab
            " um", " uh", "ugh", "soooo", "sooo ", " ummm", " uhh", " erm",
        ]
        if any(m in lower for m in bad_markers):
            reasons["bad_marker"] += 1
            continue
        if item.get("lang_type") not in {
            "pure_cn", "pure_en", "cn_mostly", "en_mostly", "frequent_mix",
        }:
            reasons["lang_type"] += 1
            continue
        if _find_invalid_tags(text):
            reasons["invalid_tags"] += 1
            continue
        if not _is_complete_for_asr(text):
            reasons["incomplete_asr"] += 1
            continue
        if _contains_pinyin_or_phonetic_markup(text):
            reasons["pinyin_phonetic"] += 1
            continue
        tag_count, tags = _count_tags(text)
        if tag_count > 4:
            reasons["too_many_tags"] += 1
            continue
        if any(tags.count(tag) > 2 for tag in set(tags)):
            reasons["repeat_tag"] += 1
            continue
        length_type = item.get("length_type", "medium")
        if _is_severe_length_mismatch(text, length_type, item):
            reasons["severe_length"] += 1
            continue
        pos_issues = _validate_tag_position(text)
        combo_issues = _validate_tag_combinations(text)
        text_corr = _auto_correct_tags(text, item.get("emotion", ""))
        tag_count, _ = _count_tags(text_corr)
        if tag_count < _min_required_tags_for_item(item):
            reasons["too_few_tags"] += 1
            continue
        if item.get("age_tier") == "preschool" and not _has_preschool_markers(text_corr):
            reasons["preschool_markers"] += 1
            continue
        probe = dict(item)
        probe["text"] = text_corr
        try:
            from .text_tn import attach_text_tn
        except ImportError:
            from text_tn import attach_text_tn
        attach_text_tn(probe)
        if not probe.get("text_tn"):
            reasons["no_text_tn"] += 1
            continue
        if len(pos_issues) + len(combo_issues) > _max_validation_issues_for_item(item):
            reasons["validation_issues"] += 1
            continue
        reasons["pass"] += 1
    return reasons


def deduplicate(texts: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    unique = []
    for item in texts:
        text = item.get("text", "").strip()
        if not text:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        normalized = re.sub(r"[\s，。？！,.?!\[\]]", "", text).lower()
        h2 = hashlib.md5(normalized.encode()).hexdigest()
        if h2 in seen_hashes:
            continue
        seen_hashes.add(h2)
        unique.append(item)
    return unique


def _normalize_for_duplicate_check(text: str) -> str:
    """Normalize text for incremental duplicate checks."""
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\d+", "<num>", text.lower())
    text = re.sub(r"[\s，。？！,.?!、；;：:\"“”‘’'\-—…~～()（）\[\]]+", "", text)
    return text.strip()


def _duplicate_context_key(item: Dict) -> Tuple[str, str, str, str, str, str]:
    """Group near-duplicate checks by the generation context."""
    return (
        item.get("scenario") or item.get("scenario_key") or "",
        item.get("subscene") or "",
        item.get("emotion") or "",
        item.get("age_tier") or "",
        item.get("length_type") or item.get("length_key") or "",
        item.get("lang_type") or item.get("lang_key") or "",
    )


def _duplicate_shingles(normalized: str) -> set:
    if len(normalized) <= 6:
        return set(normalized)
    return {normalized[i:i + 3] for i in range(len(normalized) - 2)}


def build_duplicate_index(texts: List[Dict]) -> Tuple[set, Dict[Tuple[str, str, str, str, str, str], List[Tuple[str, set]]]]:
    seen_normalized = set()
    context_index = {}
    for item in texts:
        normalized = _normalize_for_duplicate_check(item.get("text", ""))
        if not normalized:
            continue
        seen_normalized.add(normalized)
        key = _duplicate_context_key(item)
        context_index.setdefault(key, []).append((normalized, _duplicate_shingles(normalized)))
    return seen_normalized, context_index


def filter_incremental_duplicates(
    texts: List[Dict],
    seen_normalized: set,
    context_index: Dict[Tuple[str, str, str, str, str, str], List[Tuple[str, set]]],
    same_context_threshold: float = 0.52,
) -> Tuple[List[Dict], int]:
    """Filter new results before they are added to the checkpoint."""
    unique = []
    skipped = 0

    for item in texts:
        normalized = _normalize_for_duplicate_check(item.get("text", ""))
        if not normalized:
            unique.append(item)
            continue

        if normalized in seen_normalized:
            skipped += 1
            continue

        key = _duplicate_context_key(item)
        shingles = _duplicate_shingles(normalized)
        is_duplicate = False
        for existing_norm, existing_shingles in context_index.get(key, []):
            max_len = max(len(normalized), len(existing_norm), 1)
            if abs(len(normalized) - len(existing_norm)) / max_len > 0.55:
                continue

            if shingles and existing_shingles:
                overlap = len(shingles & existing_shingles) / max(len(shingles | existing_shingles), 1)
                if overlap < 0.12:
                    continue

            score = difflib.SequenceMatcher(None, normalized, existing_norm).ratio()
            if score >= same_context_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            skipped += 1
            continue

        seen_normalized.add(normalized)
        context_index.setdefault(key, []).append((normalized, shingles))
        unique.append(item)

    return unique, skipped


def build_frequency_suppression_hint(
    texts: List[Dict],
    window_size: int = 800,
) -> str:
    """Discourage overused recent patterns without banning natural child speech globally."""
    if len(texts) < 40:
        return ""

    recent = texts[-window_size:]
    opening_counter = Counter()
    tag_counter = Counter()
    filler_counter = Counter()
    object_counter = Counter()

    fillers = [
        "嗯", "那个", "就是", "啊", "呃", "哎呀", "为什么",
        "wait", "okay", "no no", "very very",
    ]
    common_objects = [
        "妈妈", "爸爸", "故事", "phone", "number", "玩具", "作业",
        "magic", "spell", "不公平", "回家", "睡觉", "小熊",
        "古诗", "唐诗", "诗人", "拼音", "生字", "造句", "作文",
        "单词", "English", "math", "加法", "乘法", "口诀", "恐龙",
        "老师", "同桌", "黑板", "实验", "磁铁", "影子", "垃圾分类",
        "跳绳", "篮球", "唱歌", "画画", "键盘", "机器人", "值日",
        "小明", "小红", "小刚", "阅读材料", "庆历四年", "Read after me",
        "已知rt", "Tom has", "how many", "相向而行", "岳阳楼",
        "我想", "我要", "我不", "能不能", "为什么", "然后就是",
        "i want", "i don't", "can we", "mom can",
    ]
    reading_phrase_counter = Counter()

    for item in recent:
        text = item.get("text", "")
        normalized = _normalize_for_duplicate_check(text)
        if len(normalized) >= 3:
            opening_counter[normalized[:4]] += 1
        for tag in re.findall(r"\[([^\]]+)\]", text):
            tag_counter[tag] += 1
        lower = text.lower()
        for filler in fillers:
            if filler.lower() in lower:
                filler_counter[filler] += 1
        for obj in common_objects:
            if obj.lower() in lower:
                object_counter[obj] += 1
        for phrase in (
            "阅读材料", "read after me", "question:", "已知rt",
            "庆历四年", "床前明月光", "how many apples", "相向而行",
            "庆历四年春", "学而时习之", "repeat after me", "小明有",
            "妈妈你看", "我觉得", "嗯那个", "不对不对",
            "我想", "我要", "我不", "能不能", "i want", "i don't",
            "因为所以", "但是如果", "先帝", "子曰",
        ):
            if phrase in lower:
                reading_phrase_counter[phrase] += 1

    def hot_terms(counter: Counter, min_count: int, limit: int) -> List[str]:
        return [term for term, count in counter.most_common(limit) if count >= min_count]

    hot_openings = hot_terms(opening_counter, max(4, len(recent) // 80), 8)
    hot_tags = hot_terms(tag_counter, max(8, len(recent) // 30), 5)
    hot_fillers = hot_terms(filler_counter, max(10, len(recent) // 25), 8)
    hot_objects = hot_terms(object_counter, max(8, len(recent) // 35), 8)

    hot_reading_phrases = hot_terms(reading_phrase_counter, max(3, len(recent) // 60), 6)

    if not any((hot_openings, hot_tags, hot_fillers, hot_objects, hot_reading_phrases)):
        return ""

    sections = []
    if hot_openings:
        sections.append(f"- 最近常见开头片段: {', '.join(hot_openings)}。本 batch 尽量换开头，不要照搬这些开头。")
    if hot_tags:
        sections.append(f"- 最近高频标签: {', '.join(f'[{t}]' for t in hot_tags)}。除非情绪强匹配，否则优先换其它允许标签或减少标签数。")
    if hot_fillers:
        sections.append(f"- 最近高频口头禅: {', '.join(hot_fillers)}。本 batch 最多少量使用，换成别的自然口语。")
    if hot_objects:
        sections.append(f"- 最近高频主题词: {', '.join(hot_objects)}。优先换具体物品、人物或冲突点。")
    if hot_reading_phrases:
        sections.append(
            f"- 最近高频朗读/读题句式: {', '.join(hot_reading_phrases)}。"
            "本 batch 换不同篇目、不同题型、不同开头，不要重复同一阅读材料或同一应用题模板。"
        )

    return f"""
=== 最近高频模式抑制 ===
下面这些模式最近出现偏多，不是绝对禁止，但本 batch 要主动避开或少用：
{chr(10).join(sections)}
"""


def _find_invalid_tags(text: str) -> List[str]:
    """Find any bracketed tags that are NOT in TAG_DEFINITIONS."""
    # Match anything in brackets
    all_bracketed = re.findall(r"\[([^\]]+)\]", text)
    valid_names = set(tag.strip("[]") for tag in TAG_DEFINITIONS.keys())
    invalid = [t for t in all_bracketed if t not in valid_names]
    return invalid


def _is_complete_for_asr(text: str) -> bool:
    """Reject half-utterances that are hard to use for WER filtering."""
    clean = re.sub(r"\[[^\]]+\]", "", text).strip()
    if not clean:
        return False
    # Allow ellipsis as natural child speech pause marker
    if re.search(r"[,，、;；:：—-]\s*$", clean):
        return False
    lower = clean.lower().rstrip()
    lower = re.sub(r"[。！？!?\"\'\"')\]）]+$", "", lower).strip()
    # Ultra-relaxed for child speech: only reject truly incomplete utterances
    unfinished_tail = (
        "还有", "可是", "因为",
        "如果", "要是", "我其实想", "我本来想",
        "等一下", "等下", "你知道", "怎么说",
        "wait", "because", "and then",
        "if", "when", "then", "you know", "i was gonna", "i'm gonna", "or maybe", "actually",
    )
    if any(lower.endswith(t) for t in unfinished_tail):
        return False
    unfinished_patterns = (
        r"(可是|如果|要是)[嘛呀啊呢吧哦]*$",
        r"(我其实想|我本来想|我只是想|我还以为)[，,\s]*(那个|就是|要|去|说|问)?$",
        r"\b(i|we|you)\s+(was gonna|were gonna|am gonna|or maybe)\s*$",
        r"\b(because|if|when|then|you know)\s*$",
    )
    if any(re.search(pattern, lower) for pattern in unfinished_patterns):
        return False
    if re.search(r"(算了不说了|不说了|说不出来|never mind|whatever)$", lower):
        return False
    return True


_LENGTH_MIN_TAGS = {
    "ultra_short": 0,
    "short": 0,
    "medium": 0,
    "long": 0,
    "very_long": 0,
}

_PRESCHOOL_MARKERS_RE = re.compile(
    r"(饭饭|水水|觉觉|手手|脚脚|抱抱抱|走走走|要要要|不不不|好好好|怕怕怕|"
    r"宝宝要|宝宝不|宝宝想|明明不|明明要|朵朵|七饭|回水|脑斧|灰机|"
    r"吃饭饭|喝水水|睡觉觉|洗手手|亲亲|抱抱[^抱]|走走[^走]|"
    r"呜呜|哼哼|痛痛|痒痒|怕怕|乖乖)"
)
_PRESCHOOL_PARTICLE_RE = re.compile(
    r"(嘛[！。]?$|呀[！。]?$|啦[！。]?$|呢[！。]?$|哦[！。]?$|"
    r"不要嘛|好不好呀|行不行呀|人家|妈咪|爸爸抱)"
)

# Pinyin / zhuyin / IPA / "read this punctuation aloud" patterns — not for TTS text.
_PINYIN_TONE_VOWEL_RE = re.compile(
    r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüÜɑḿń]"
)
_BOPOMOFO_RE = re.compile(r"[ㄅ-ㄩ]")
_PINYIN_NUM_TONE_RE = re.compile(
    r"(?<![a-zA-Z])[a-z]{1,6}[1-5](?![a-zA-Z0-9])",
    re.IGNORECASE,
)
_PINYIN_SLASH_IPA_RE = re.compile(r"/[a-zA-Zəɪʊʌæɔɑɒθðʃʒŋ]+/")
_PINYIN_LABEL_RE = re.compile(
    r"(读作|读音[是为]|拼音[是为：:]|注音[是为]|谐音[是为]|国际音标)"
)
_META_PUNCT_SPOKEN_RE = re.compile(
    r"(省略号|顿号|逗号|句号|问号|叹号|感叹号|分号|冒号|破折号|引号|书名号|"
    r"括号|斜杠|反斜杠|星号|井号|at\s*sign|ellipsis)"
)
_PHONETIC_ONLY_PUNCT_RE = re.compile(r"[·•※◆●□◇▲▼]")
# Standalone bopomofo tone marks (often printed above zhuyin).
_ZHUYIN_TONE_MARK_RE = re.compile(r"[ˊˇˋ˙ˉˊˇˋ]")


def _contains_pinyin_or_phonetic_markup(text: str) -> bool:
    """True if text embeds pinyin/zhuyin/IPA or asks to speak punctuation names."""
    clean = re.sub(r"\[[^\]]+\]", "", text)
    if not clean.strip():
        return False
    if _PINYIN_TONE_VOWEL_RE.search(clean):
        return True
    if _BOPOMOFO_RE.search(clean):
        return True
    if _PINYIN_NUM_TONE_RE.search(clean):
        return True
    if _PINYIN_SLASH_IPA_RE.search(clean):
        return True
    if _PINYIN_LABEL_RE.search(clean):
        return True
    if _META_PUNCT_SPOKEN_RE.search(clean):
        return True
    if _PHONETIC_ONLY_PUNCT_RE.search(clean):
        return True
    if _ZHUYIN_TONE_MARK_RE.search(clean):
        return True
    return False


def _min_required_tags(length_type: str) -> int:
    return _LENGTH_MIN_TAGS.get(length_type, 1)


def _has_preschool_markers(text: str) -> bool:
    clean = re.sub(r"\[[^\]]+\]", "", text)
    if _PRESCHOOL_MARKERS_RE.search(clean):
        return True
    return bool(_PRESCHOOL_PARTICLE_RE.search(clean))


def quality_filter(
    texts: List[Dict],
    reject_severe_length_mismatch: bool = True,
) -> List[Dict]:
    filtered = []
    for item in texts:
        text = item.get("text", "").strip()
        if not text:
            continue
        if len(text) < 1:
            continue

        lower = text.lower()
        bad_markers = [
            "股票", "投资", "政治", "战争", "sex", "kill", "die", "porn",
            "no cap", "cap,", " emo", "emo了", "绝绝子", "yyds",
            # English fillers not in ASR vocab
            " um", " uh", "ugh", "soooo", "sooo ", " ummm", " uhh", " erm",
        ]
        if any(m in lower for m in bad_markers):
            continue

        # Validate lang_type
        valid_lang_types = {"pure_cn", "pure_en", "cn_mostly", "en_mostly", "frequent_mix"}
        if item.get("lang_type") not in valid_lang_types:
            continue

        # Reject texts with invalid (non-existent) tags
        invalid_tags = _find_invalid_tags(text)
        if invalid_tags:
            continue

        # WER/ASR filtering needs complete, readable reference text.
        if not _is_complete_for_asr(text):
            continue

        if _contains_pinyin_or_phonetic_markup(text):
            continue

        # Tag count validation
        tag_count, tags = _count_tags(text)
        if tag_count > 4:
            continue  # Too many tags

        # Check for same tag repeated too many times
        if any(tags.count(tag) > 2 for tag in set(tags)):
            continue

        # Length validation
        length_type = item.get("length_type", "medium")
        if not _validate_length(text, length_type):
            item["_length_mismatch"] = True
        if reject_severe_length_mismatch and _is_severe_length_mismatch(
            text, length_type, item
        ):
            continue

        # Run validation
        pos_issues = _validate_tag_position(text)
        combo_issues = _validate_tag_combinations(text)

        # Auto-correct
        text = _auto_correct_tags(text, item.get("emotion", ""))
        item["text"] = text
        tag_count, tags = _count_tags(text)

        if tag_count < _min_required_tags_for_item(item):
            continue

        if item.get("age_tier") == "preschool" and not _has_preschool_markers(text):
            continue

        try:
            from .text_tn import attach_text_tn
        except ImportError:
            from text_tn import attach_text_tn

        attach_text_tn(item)
        if not item.get("text_tn"):
            continue

        if len(pos_issues) + len(combo_issues) > _max_validation_issues_for_item(item):
            continue

        # Store validation info for analysis
        item["_tag_count"] = tag_count
        item["_tags"] = tags
        item["_validation_issues"] = pos_issues + combo_issues

        filtered.append(item)
    return filtered


def _age_tier_targets(config: GenConfig, total: int) -> Dict[str, int]:
    """Largest-remainder allocation for age tiers."""
    tiers = list(config.age_distribution.keys())
    weights = [config.age_distribution[t] for t in tiers]
    weight_sum = sum(weights) or 1.0
    raw = [total * w / weight_sum for w in weights]
    counts = {t: int(r) for t, r in zip(tiers, raw)}
    remainder = total - sum(counts.values())
    if remainder > 0:
        fractions = sorted(
            ((raw[i] - int(raw[i]), tiers[i]) for i in range(len(tiers))),
            reverse=True,
        )
        for _, tier in fractions:
            if remainder <= 0:
                break
            counts[tier] += 1
            remainder -= 1
    return counts


def _accept_refill_by_age_quota(
    results: List[Dict],
    texts: List[Dict],
    config: GenConfig,
    target: int,
) -> List[Dict]:
    """Prefer candidates that fill under-represented age tiers."""
    if not results:
        return []
    remaining = target - len(texts)
    if remaining <= 0:
        return []

    tier_targets = _age_tier_targets(config, target)
    tier_counts = Counter(item.get("age_tier") for item in texts)
    accepted: List[Dict] = []
    pool = list(results)

    while pool and len(accepted) < remaining:
        def priority(idx: int) -> Tuple[int, int]:
            tier = pool[idx].get("age_tier", "")
            shortfall = tier_targets.get(tier, 0) - tier_counts.get(tier, 0)
            return (shortfall, -idx)

        best_idx = max(range(len(pool)), key=priority)
        item = pool.pop(best_idx)
        accepted.append(item)
        tier = item.get("age_tier")
        if tier:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return accepted


def refill_to_target(
    texts: List[Dict],
    config: GenConfig,
    checkpoint_path: Optional[str] = None,
    future_timeout: int = 180,
    max_rounds: int = 8,
) -> List[Dict]:
    """Generate extra batches after filtering so the saved dataset reaches total_target."""
    target = config.total_target
    if len(texts) >= target:
        return texts[:target]

    print(f"Refill needed: {len(texts)}/{target} texts after filtering")
    seen_normalized, duplicate_context_index = build_duplicate_index(texts)

    def next_task_id_start() -> int:
        task_ids = []
        for item in texts:
            try:
                task_ids.append(int(item.get("task_id", -1)))
            except (TypeError, ValueError):
                continue
        return max(task_ids, default=-1) + 1

    next_task_id = next_task_id_start()
    stalled_rounds = 0

    for round_idx in range(max_rounds):
        missing = target - len(texts)
        if missing <= 0:
            break

        batches_needed = (missing + config.batch_size - 1) // config.batch_size
        batches_to_request = max(config.max_workers, int(batches_needed * 1.3) + 1)
        refill_config = replace(
            config,
            total_target=batches_to_request * config.batch_size,
            seed=config.seed + 1_000_003 + round_idx * 7_919 + len(texts),
        )
        refill_tasks = generate_task_list(refill_config)
        for offset, task in enumerate(refill_tasks):
            task["task_id"] = next_task_id + offset
        next_task_id += len(refill_tasks)

        added_this_round = 0
        skipped_duplicates = 0
        failed = 0
        completed = 0
        print(
            f"Refill round {round_idx + 1}: missing={missing}, "
            f"requesting {len(refill_tasks)} batches"
        )

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_task = {}
            next_task_idx = 0

            def submit_next_task():
                nonlocal next_task_idx
                if next_task_idx >= len(refill_tasks) or len(texts) >= target:
                    return
                task_to_submit = dict(refill_tasks[next_task_idx])
                task_to_submit["suppression_hint"] = build_frequency_suppression_hint(
                    texts,
                    window_size=config.suppression_window_size,
                )
                future = executor.submit(worker, task_to_submit, config)
                future_to_task[future] = task_to_submit
                next_task_idx += 1

            for _ in range(min(config.max_workers, len(refill_tasks))):
                submit_next_task()

            while future_to_task and len(texts) < target:
                for future in as_completed(list(future_to_task), timeout=None):
                    task = future_to_task.pop(future)
                    break

                try:
                    results = future.result(timeout=future_timeout)
                    if results:
                        results = quality_filter(
                            results,
                            reject_severe_length_mismatch=config.reject_severe_length_mismatch,
                        )
                    if results:
                        results, skipped = filter_incremental_duplicates(
                            results,
                            seen_normalized,
                            duplicate_context_index,
                            same_context_threshold=config.same_context_dup_threshold,
                        )
                        skipped_duplicates += skipped
                        if results:
                            accepted = _accept_refill_by_age_quota(
                                results, texts, config, target
                            )
                            texts.extend(accepted)
                            added_this_round += len(accepted)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    print(f"Refill task {task['task_id']} failed: {exc}")
                    failed += 1

                done = completed + failed
                if done % 10 == 0 or len(texts) >= target:
                    print(
                        f"Refill progress: total={len(texts)}/{target}, "
                        f"completed={completed}, failed={failed}, "
                        f"skipped_duplicates={skipped_duplicates}"
                    )
                    if checkpoint_path:
                        save_checkpoint(texts, checkpoint_path)

                submit_next_task()

        if checkpoint_path:
            save_checkpoint(texts, checkpoint_path)
        print(
            f"Refill round {round_idx + 1} added {added_this_round} texts; "
            f"total={len(texts)}/{target}"
        )

        if added_this_round == 0:
            stalled_rounds += 1
            if stalled_rounds >= 2:
                print("Refill stopped after two rounds with no accepted texts.")
                break
        else:
            stalled_rounds = 0

    if len(texts) < target:
        print(f"WARNING: refill ended below target: {len(texts)}/{target}")
    return texts[:target]


def analyze_tags(texts: List[Dict]) -> Dict:
    tag_pattern = re.compile(rf"\[({VALID_TAG_NAMES})\]")
    stats = {
        "total_texts": len(texts),
        "texts_with_tags": 0,
        "tag_distribution": {},
        "avg_tags_per_text": 0,
        "by_length": {},
        "validation_issues": 0,
    }

    total_tags = 0
    for item in texts:
        text = item.get("text", "")
        tags = tag_pattern.findall(text)
        length = item.get("length_type", "unknown")

        if tags:
            stats["texts_with_tags"] += 1
            total_tags += len(tags)
            for tag in tags:
                stats["tag_distribution"][tag] = stats["tag_distribution"].get(tag, 0) + 1

        # Count by length
        if length not in stats["by_length"]:
            stats["by_length"][length] = {"count": 0, "with_tags": 0, "avg_tags": 0}
        stats["by_length"][length]["count"] += 1
        stats["by_length"][length]["with_tags"] += 1 if tags else 0
        stats["by_length"][length]["avg_tags"] += len(tags)

        if item.get("_validation_issues"):
            stats["validation_issues"] += len(item["_validation_issues"])

    # Normalize by-length stats
    for length, data in stats["by_length"].items():
        if data["count"] > 0:
            data["avg_tags"] = data["avg_tags"] / data["count"]
            data["tag_rate"] = data["with_tags"] / data["count"]

    if texts:
        stats["avg_tags_per_text"] = total_tags / len(texts)
        stats["tag_coverage"] = stats["texts_with_tags"] / len(texts)

    return stats


def analyze_emotion_tag_alignment(texts: List[Dict]) -> Dict:
    """Check if tags align with declared emotion."""
    tag_pattern = re.compile(rf"\[({VALID_TAG_NAMES})\]")
    alignment = {}

    for item in texts:
        emotion = item.get("emotion", "unknown")
        text = item.get("text", "")
        tags = tag_pattern.findall(text)

        if emotion not in alignment:
            alignment[emotion] = {"count": 0, "expected_tags": set(), "actual_tags": [], "match_rate": 0}

        profile = EMOTION_PROFILES.get(emotion, {})
        expected = set(profile.get("primary_tags", []) + profile.get("secondary_tags", []))
        expected = {t.strip("[]") for t in expected}

        alignment[emotion]["count"] += 1
        alignment[emotion]["expected_tags"] = expected
        alignment[emotion]["actual_tags"].extend(tags)

    # Calculate match rate
    for emotion, data in alignment.items():
        if data["actual_tags"]:
            matched = sum(1 for t in data["actual_tags"] if t in data["expected_tags"])
            data["match_rate"] = matched / len(data["actual_tags"])

    return alignment


def analyze_language(texts: List[Dict]) -> Dict:
    stats = {"by_declared": {}, "by_actual": {}}
    for item in texts:
        lang_type = item.get("lang_type", "unknown")
        language = item.get("language", "unknown")
        stats["by_declared"][lang_type] = stats["by_declared"].get(lang_type, 0) + 1
        stats["by_actual"][language] = stats["by_actual"].get(language, 0) + 1
    return stats


# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def save_checkpoint(texts: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in texts:
            # Strip internal fields for output
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def load_checkpoint(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    texts.append(json.loads(line))
                except:
                    pass
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = apply_config_from_env(GenConfig())
    os.makedirs(config.output_dir, exist_ok=True)

    api_key = (
        config.api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print("=" * 60)
        print("ERROR: API key is not set.")
        print(f"Copy {_ENV_FILE.name}.example to {_ENV_FILE.name} and set LLM_API_KEY.")
        print("=" * 60)
        sys.exit(1)

    output_jsonl = os.path.join(config.output_dir, "llm_children_v3.jsonl")
    checkpoint_path = os.path.join(config.output_dir, ".checkpoint_v3.jsonl")

    all_texts = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(all_texts)} existing texts from checkpoint")
    seen_normalized, duplicate_context_index = build_duplicate_index(all_texts)

    tasks = generate_task_list(config)
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks} (batch_size={config.batch_size}, target={config.total_target})")
    print(f"Model: {config.model}")
    print(f"Same-context duplicate threshold: {config.same_context_dup_threshold}")

    completed_task_ids = {t.get("task_id", -1) for t in all_texts}
    pending_tasks = [t for t in tasks if t["task_id"] not in completed_task_ids]
    print(f"Pending tasks: {len(pending_tasks)}")

    if not pending_tasks:
        print("All tasks completed!")
    else:
        completed = 0
        failed = 0
        skipped_duplicates = 0

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_task = {}
            next_task_idx = 0

            def submit_next_task():
                nonlocal next_task_idx
                if next_task_idx >= len(pending_tasks):
                    return
                task_to_submit = dict(pending_tasks[next_task_idx])
                task_to_submit["suppression_hint"] = build_frequency_suppression_hint(
                    all_texts,
                    window_size=config.suppression_window_size,
                )
                future = executor.submit(worker, task_to_submit, config)
                future_to_task[future] = task_to_submit
                next_task_idx += 1

            for _ in range(min(config.max_workers, len(pending_tasks))):
                submit_next_task()

            while future_to_task:
                for future in as_completed(list(future_to_task), timeout=None):
                    task = future_to_task.pop(future)
                    break
                try:
                    results = future.result(timeout=120)
                    if results:
                        results, skipped = filter_incremental_duplicates(
                            results,
                            seen_normalized,
                            duplicate_context_index,
                            same_context_threshold=config.same_context_dup_threshold,
                        )
                        skipped_duplicates += skipped
                        all_texts.extend(results)
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Task {task['task_id']} failed: {e}")
                    failed += 1

                if (completed + failed) % 10 == 0:
                    print(
                        f"Progress: {completed} succeeded, {failed} failed, "
                        f"total={len(all_texts)}, skipped_duplicates={skipped_duplicates}"
                    )
                    save_checkpoint(all_texts, checkpoint_path)

                submit_next_task()

        save_checkpoint(all_texts, checkpoint_path)
        if skipped_duplicates:
            print(f"Skipped {skipped_duplicates} near-duplicate texts before checkpointing")

    # Post-processing
    print(f"\nPost-processing {len(all_texts)} texts...")
    all_texts = deduplicate(all_texts)
    print(f"After exact dedup: {len(all_texts)}")
    all_texts = semantic_deduplicate(all_texts, threshold=config.semantic_dedup_threshold)
    print(f"After semantic dedup: {len(all_texts)}")
    all_texts = quality_filter(
        all_texts,
        reject_severe_length_mismatch=config.reject_severe_length_mismatch,
    )
    print(f"After quality filter: {len(all_texts)}")
    all_texts = refill_to_target(all_texts, config, checkpoint_path=checkpoint_path, future_timeout=120)
    print(f"After refill: {len(all_texts)}")

    # Do not truncate by default: truncation creates half-utterances that hurt WER filtering.
    if config.truncate_overlength:
        truncated_count = 0
        for item in all_texts:
            lt = item.get("length_type", "medium")
            if lt in ("ultra_short", "short", "medium", "long"):
                original = item["text"]
                truncated = _truncate_to_length(original, lt)
                if truncated != original:
                    item["text"] = truncated
                    truncated_count += 1
        if truncated_count:
            print(f"Truncated {truncated_count} over-length texts")

    length_mismatches = sum(1 for t in all_texts if t.get("_length_mismatch"))
    if length_mismatches:
        print(f"Length mismatches (warned, not filtered): {length_mismatches}")

    save_checkpoint(all_texts, output_jsonl)

    # Statistics
    from collections import Counter

    print("\n=== Final Statistics ===")
    print(f"Total texts: {len(all_texts)}")
    print(f"Length: {dict(Counter(t.get('length_type') for t in all_texts))}")
    print(f"Language: {dict(Counter(t.get('lang_type') for t in all_texts))}")
    print(f"Age: {dict(Counter(t.get('age_tier') for t in all_texts))}")
    print(f"Scenario: {dict(Counter(t.get('scenario') for t in all_texts))}")

    # Tag analysis
    tag_stats = analyze_tags(all_texts)
    print(f"\nTag Statistics:")
    print(f"  Texts with tags: {tag_stats['texts_with_tags']} ({tag_stats.get('tag_coverage', 0)*100:.1f}%)")
    print(f"  Avg tags per text: {tag_stats['avg_tags_per_text']:.2f}")
    print(f"  Tag distribution: {tag_stats['tag_distribution']}")
    print(f"  Validation issues: {tag_stats['validation_issues']}")

    # By length
    print(f"\nTag by Length:")
    for length, data in sorted(tag_stats["by_length"].items()):
        print(f"  {length}: {data['tag_rate']*100:.0f}% have tags, avg {data['avg_tags']:.1f} tags")

    # Emotion-tag alignment
    alignment = analyze_emotion_tag_alignment(all_texts)
    print(f"\nEmotion-Tag Alignment:")
    for emotion, data in sorted(alignment.items(), key=lambda x: -x[1]["match_rate"]):
        print(f"  {emotion}: match_rate={data['match_rate']:.1%} (n={data['count']})")

    # Language
    lang_stats = analyze_language(all_texts)
    print(f"\nLanguage Distribution:")
    print(f"  Declared: {lang_stats['by_declared']}")
    print(f"  Actual (OmniVoice): {lang_stats['by_actual']}")

    print(f"\nSaved to: {output_jsonl}")


if __name__ == "__main__":
    main()
