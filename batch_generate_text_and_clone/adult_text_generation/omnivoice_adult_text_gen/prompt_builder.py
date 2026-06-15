"""Compact adult prompt builder with diversity axes (OmniVoice [tag] format)."""

import hashlib
import random
from typing import Optional

from .scenarios import (
    EMOTIONS,
    LANG_SPECS,
    LENGTH_SPECS,
    SCENARIO_EMOTION_MAP,
    SCENARIOS,
)
from .tags import EMOTION_PROFILES, TAG_DEFINITIONS, TAG_DENSITY_MAP

_PERSONAS = [
    "急性子打工人(说话快、直接)",
    "温柔体贴(语气软、带关心)",
    "挑剔顾客(话里带刺)",
    "疲惫上班族(语气低沉)",
    "话痨朋友(絮絮叨叨)",
    "严肃老板(简短有力)",
    "害羞内向(欲言又止)",
    "自信达人(语气上扬)",
    "焦虑患者(反复确认)",
    "佛系青年(无所谓态度)",
    "专业主播(节奏稳定)",
    "八卦同事(神秘兮兮)",
]

_PLACES = [
    "家里", "公司", "咖啡店", "地铁", "公园", "超市", "餐厅",
    "健身房", "图书馆", "商场", "路上", "车里", "机场", "酒店",
]

_TIMES = [
    "早上刚醒", "中午吃饭", "下午茶", "下班路上", "深夜加班",
    "周末清晨", "周一早晨", "假期第一天", "下雨天", "晴天傍晚",
]

_REGISTERS = [
    "随意口语(好友聊天)",
    "半正式(同事/客户)",
    "正式(演讲/汇报)",
    "急切(催促/紧急)",
    "亲昵(家人/恋人)",
    "幽默调侃",
    "严肃警告",
]

_TOPIC_SEEDS = {
    "daily_chat": [
        "丢了东西/找到了", "收到意外礼物", "被放鸽子", "做饭翻车",
        "偶遇老同学", "手机坏了", "快递到了", "睡过头", "小区装修噪音",
    ],
    "business": [
        "项目延期", "客户改需求", "预算被砍", "升职/被表扬",
        "方案被否", "临时会议", "同事离职", "数据不对", "季度汇报",
    ],
    "education": [
        "突然顿悟", "背不下来", "实验失败", "在线课程卡顿",
        "分享学习笔记", "考试前夜", "技能上手",
    ],
    "emotional": [
        "久别重逢", "误会解开", "深夜倾诉", "道歉和好",
        "感动落泪", "遗憾错过",
    ],
    "entertainment": [
        "游戏翻车", "段子接龙", "模仿名人", "看剧吐槽",
        "直播互动", "综艺reaction",
    ],
    "narration": [
        "城市夜景", "历史事件", "产品介绍", "旅行见闻",
        "人物传记", "科技趋势",
    ],
    "social_media": [
        "开箱测评", "探店vlog", "健身打卡", "穿搭分享",
        "数码评测", "旅行攻略",
    ],
    "service": [
        "改签机票", "投诉处理", "账单疑问", "预约确认",
        "售后退换", "业务办理",
    ],
    "creative_writing": [
        "散文片段", "小说独白", "诗歌朗诵", "书信朗读",
    ],
    "asr_stress": [
        "一串订单号", "快速报数字", "中英切换", "长句连读",
    ],
}


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def _emotion_profile(emotion: str) -> dict:
    mapped = SCENARIO_EMOTION_MAP.get(emotion, "happy")
    return EMOTION_PROFILES.get(mapped, EMOTION_PROFILES["happy"])


def build_compact_prompt(
    scenario_key: str,
    subscene: str,
    length_key: str,
    lang_key: str,
    emotion: str,
    batch_size: int,
    suppression_hint: str = "",
    task_id: Optional[int] = None,
) -> str:
    scenario = SCENARIOS[scenario_key]
    length_spec = LENGTH_SPECS[length_key]
    lang_spec = LANG_SPECS[lang_key]
    profile = _emotion_profile(emotion)
    density_min, density_max = TAG_DENSITY_MAP[profile["tag_density"]]
    primary = ", ".join(profile["primary_tags"])
    secondary = ", ".join(profile["secondary_tags"]) if profile["secondary_tags"] else "无"

    seed = int(hashlib.md5(f"adult|{task_id}|{scenario_key}|{emotion}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    persona = _pick(rng, _PERSONAS)
    place = _pick(rng, _PLACES)
    time_ctx = _pick(rng, _TIMES)
    register = _pick(rng, _REGISTERS)
    topics = _TOPIC_SEEDS.get(scenario_key, ["日常小事"])
    topic = _pick(rng, topics)

    tag_lines = []
    for tag_name, info in TAG_DEFINITIONS.items():
        tag_lines.append(
            f"{tag_name}: {info['description']} | 位置: {info['placement']} | 例: {info['examples'][0]}"
        )
    tag_guide = "\n".join(tag_lines)

    lang_rule = ""
    if lang_key == "pure_en":
        lang_rule = "【语言】必须纯英文，禁止汉字。"
    elif lang_key == "pure_cn":
        lang_rule = "【语言】必须纯中文，禁止拉丁字母。"
    elif lang_key == "frequent_mix":
        lang_rule = "【语言】中英显著混用，各至少30%，句子中间自然切换。"

    return f"""你是专业的成人自然口语语料采集专家。请生成 {batch_size} 条听起来像真实成年人（18-60岁）自然说出口的文本。

这些文本将用于 OmniVoice 语音克隆/TTS。可在文本中插入非语言标签（如 [laughter]、[sigh]）控制副语言。

=== 任务参数 ===
场景: {scenario["name"]} — {subscene}
情绪: {emotion}（OmniVoice标签倾向: {primary}；次要: {secondary}）
长度: {length_spec["cn"] if "cn" in lang_key else length_spec["en"]}
语言: {lang_spec}
{lang_rule}

=== 多样性轴（本 batch 必须体现） ===
人设: {persona}
地点: {place}
时间: {time_ctx}
语体: {register}
话题种子: {topic}
要求: batch 内换开头、换主语、换句式，不要10条都像同一个人在说同一件事。

{suppression_hint}

=== 成人口语规则 ===
- 像真实成人说话：可有填充词（嗯、那个、就是）、自我修正、口语省略，但必须说完整。
- 禁止半句、残句；不要以逗号/省略号/and/but/因为/而且 结尾。
- 禁止儿童叠词（吃饭饭、宝宝要）、幼稚口吻、作文腔。
- 禁止: um, uh, ugh, soooo, no cap, yyds, 绝绝子 等 ASR 不友好网络梗。
- 叙述/旁白/朗读类可更书面，但仍需完整可朗读。

=== 标签规则 ===
- 标签格式仅允许: {", ".join(TAG_DEFINITIONS.keys())}
- 建议标签数: {density_min}-{density_max}，最多4个；约30%句子可无标签。
- 标签需符合场景情绪，位置合理；[laughter] 不在句首。

=== 标签参考 ===
{tag_guide}

=== 输出格式 ===
返回 JSON 数组，每项字段:
- text: 带标签的完整口语文本
- length_type: "{length_key}"
- lang_type: "{lang_key}"
- scenario: "{scenario_key}"
- subscene: "{subscene}"
- emotion: "{emotion}"
- language: 中文为主填 "zh"，英文为主填 "en"

只输出 JSON 数组，不要解释。"""
