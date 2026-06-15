#!/usr/bin/env python3
"""Generate a single batch of adult texts for prompt testing."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omnivoice_adult_text_gen.config import GenConfig, apply_config_from_env
from omnivoice_adult_text_gen.output import save_jsonl
from omnivoice_adult_text_gen.prompt_builder import build_compact_prompt
from omnivoice_adult_text_gen.llm_client import call_llm
from omnivoice_adult_text_gen.quality_filter import quality_filter
from omnivoice_adult_text_gen.scenarios import EMOTIONS, LANG_MIX_SPECS, LENGTH_SPECS, SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="daily_chat")
    parser.add_argument("--subscene", default=None)
    parser.add_argument("--emotion", choices=EMOTIONS, default="happy")
    parser.add_argument("--length", choices=list(LENGTH_SPECS.keys()), default="medium")
    parser.add_argument("--lang", choices=list(LANG_MIX_SPECS.keys()), default="pure_cn")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--output", default="")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        for key, s in SCENARIOS.items():
            print(f"  {key}: {s['name']} — {s['description']}")
        return

    config = apply_config_from_env(GenConfig(batch_size=args.count, temperature=args.temperature))
    scenario = SCENARIOS[args.scenario]
    subscene = args.subscene or scenario["subscenes"][0]

    print(f"Generating {args.count} adult texts...")
    print(f"  {scenario['name']} / {subscene} / {args.emotion} / {args.length} / {args.lang}")

    prompt = build_compact_prompt(
        scenario_key=args.scenario,
        subscene=subscene,
        length_key=args.length,
        lang_key=args.lang,
        emotion=args.emotion,
        batch_size=args.count,
        task_id=0,
    )

    results = call_llm(
        prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    if not results:
        print("No results.")
        return

    for item in results:
        item.setdefault("scenario", args.scenario)
        item.setdefault("subscene", subscene)
        item.setdefault("emotion", args.emotion)
        item.setdefault("length_type", args.length)
        item.setdefault("lang_type", args.lang)
        item.setdefault("age_tier", "adult")
        lt = item.get("lang_type", "")
        item.setdefault("language", "zh" if "cn" in lt else "en")

    filtered = quality_filter(results)
    print(f"Quality pass: {len(filtered)}/{len(results)}")

    for i, item in enumerate(filtered, 1):
        print(f"\n[{i}] {item.get('text', '')}")
        if item.get("text_tn"):
            print(f"    text_tn: {item['text_tn']}")

    if args.output:
        save_jsonl(filtered, args.output)
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + json.dumps(filtered, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
