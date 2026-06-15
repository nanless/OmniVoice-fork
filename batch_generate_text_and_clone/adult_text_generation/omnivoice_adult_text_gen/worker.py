"""Worker: build prompt, call LLM, attach metadata and text_tn."""

import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List

from .config import GenConfig
from .llm_client import call_llm
from .prompt_builder import build_compact_prompt

_CHILD_GEN = Path(__file__).resolve().parents[2] / "text_generation"
if str(_CHILD_GEN) not in sys.path:
    sys.path.insert(0, str(_CHILD_GEN))


def worker(task: Dict, config: GenConfig) -> List[Dict]:
    prompt = build_compact_prompt(
        scenario_key=task["scenario_key"],
        subscene=task["subscene"],
        length_key=task["length_key"],
        lang_key=task["lang_key"],
        emotion=task["emotion"],
        batch_size=config.batch_size,
        suppression_hint=task.get("suppression_hint", ""),
        task_id=task.get("task_id"),
    )
    temp_rng = random.Random(
        hashlib.md5(f"temp|{task.get('task_id')}|{task.get('scenario_key')}".encode()).hexdigest()
    )
    temperature = min(1.0, max(0.70, config.temperature + temp_rng.uniform(-0.10, 0.12)))

    results = call_llm(
        prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        max_tokens=config.max_tokens,
        temperature=temperature,
    )
    if not results:
        return []

    for item in results:
        item["task_id"] = task["task_id"]
        item["age_tier"] = "adult"
        item.setdefault("scenario", task["scenario_key"])
        item.setdefault("subscene", task["subscene"])
        item.setdefault("emotion", task["emotion"])
        item.setdefault("length_type", task["length_key"])
        item.setdefault("lang_type", task["lang_key"])
        lt = item.get("lang_type", "")
        item.setdefault("language", "zh" if "cn" in lt or lt == "frequent_mix" else "en")

    if config.generate_text_tn:
        from text_tn import attach_text_tn_batch  # noqa: WPS433

        attach_text_tn_batch(results)
    return results
