import os
from dataclasses import dataclass, field, replace
from typing import Dict

from .env import load_env_file

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "batch_generated_text")
DEFAULT_LLM_MODEL = "qwen3.6-27b"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"


@dataclass
class GenConfig:
    total_target: int = 100000
    batch_size: int = 10
    max_workers: int = 10
    generate_text_tn: bool = True
    oversample_ratio: float = 1.20
    output_dir: str = _DEFAULT_OUTPUT_DIR
    seed: int = 42

    same_context_dup_threshold: float = 0.52
    semantic_dedup_threshold: float = 0.88
    reject_severe_length_mismatch: bool = True
    suppression_window_size: int = 500
    max_tags_per_text: int = 4
    max_same_tag_repeat: int = 2

    length_distribution: Dict[str, float] = field(default_factory=lambda: {
        "ultra_short": 0.12,
        "short": 0.28,
        "medium": 0.32,
        "long": 0.20,
        "very_long": 0.08,
    })

    lang_mix_distribution: Dict[str, float] = field(default_factory=lambda: {
        "pure_cn": 0.40,
        "pure_en": 0.30,
        "cn_mostly": 0.15,
        "en_mostly": 0.08,
        "frequent_mix": 0.07,
    })

    scenario_distribution: Dict[str, float] = field(default_factory=lambda: {
        "daily_chat": 2.0,
        "business": 1.2,
        "education": 1.0,
        "emotional": 1.2,
        "entertainment": 1.5,
        "narration": 1.0,
        "social_media": 1.5,
        "service": 0.8,
        "creative_writing": 0.8,
        "asr_stress": 0.8,
    })

    stress_test_ratio: float = 0.10

    model: str = DEFAULT_LLM_MODEL
    api_key: str | None = None
    base_url: str = DEFAULT_LLM_BASE_URL
    max_retries: int = 3
    retry_base_delay: float = 1.0
    max_tokens: int = 8192
    temperature: float = 0.85
    truncate_overlength: bool = False


def apply_config_from_env(config: GenConfig) -> GenConfig:
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
    if os.environ.get("GEN_OVERSAMPLE_RATIO"):
        config.oversample_ratio = max(1.0, float(os.environ["GEN_OVERSAMPLE_RATIO"]))
    if not config.model:
        config.model = DEFAULT_LLM_MODEL
    if not config.base_url:
        config.base_url = DEFAULT_LLM_BASE_URL
    return config
