"""Task list generation for adult text batches."""

import random
from typing import Dict, List

from .config import GenConfig
from .scenarios import EMOTIONS, SCENARIOS


def _weighted_choice(options: Dict[str, float], rng: random.Random) -> str:
    keys = list(options.keys())
    weights = list(options.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def generate_task_list(config: GenConfig) -> List[Dict]:
    rng = random.Random(config.seed)
    total_batches = max(1, (config.total_target + config.batch_size - 1) // config.batch_size)

    regular = {k: v for k, v in SCENARIOS.items() if not v.get("is_stress_test")}
    stress = {k: v for k, v in SCENARIOS.items() if v.get("is_stress_test")}
    regular_keys = list(regular.keys())
    stress_keys = list(stress.keys())
    num_stress = int(total_batches * config.stress_test_ratio)

    subscene_recent: Dict[str, List[str]] = {k: [] for k in regular_keys}
    emotion_recent: Dict[str, List[str]] = {k: [] for k in regular_keys}
    tasks: List[Dict] = []

    for i in range(total_batches):
        if i < num_stress and stress_keys:
            scenario_key = rng.choice(stress_keys)
        else:
            scenario_key = _weighted_choice(
                {k: config.scenario_distribution.get(k, 1.0) for k in regular_keys},
                rng,
            )

        scenario = SCENARIOS[scenario_key]
        subscenes = scenario["subscenes"]
        recent_sub = subscene_recent[scenario_key]
        if len(recent_sub) >= 3 and len(subscenes) > 1:
            pool = [s for s in subscenes if s not in set(recent_sub[-3:])] or subscenes
        else:
            pool = subscenes
        subscene = rng.choice(pool)
        recent_sub.append(subscene)
        if len(recent_sub) > 10:
            subscene_recent[scenario_key] = recent_sub[-10:]

        emo_weights = scenario.get("typical_emotions")
        if emo_weights:
            emo_list = list(emo_weights.keys())
            emo_w = list(emo_weights.values())
            recent_emo = emotion_recent[scenario_key]
            if len(recent_emo) >= 2 and len(emo_list) > 1:
                filtered = [(e, w) for e, w in zip(emo_list, emo_w) if e not in set(recent_emo[-2:])]
                if not filtered:
                    filtered = list(zip(emo_list, emo_w))
                el, ew = zip(*filtered)
                emotion = rng.choices(list(el), weights=list(ew), k=1)[0]
            else:
                emotion = rng.choices(emo_list, weights=emo_w, k=1)[0]
            recent_emo.append(emotion)
            if len(recent_emo) > 10:
                emotion_recent[scenario_key] = recent_emo[-10:]
        else:
            emotion = rng.choice(EMOTIONS)

        tasks.append({
            "task_id": 0,
            "scenario_key": scenario_key,
            "subscene": subscene,
            "length_key": _weighted_choice(config.length_distribution, rng),
            "lang_key": _weighted_choice(config.lang_mix_distribution, rng),
            "emotion": emotion,
            "age_tier": "adult",
        })

    rng.shuffle(tasks)
    for idx, task in enumerate(tasks):
        task["task_id"] = idx
    return tasks
