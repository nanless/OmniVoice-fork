#!/usr/bin/env python3
"""Local HuggingFace model inference (no vLLM needed).

Usage:
    from llm_local import call_llm_local
    results = call_llm_local(prompt, model_name_or_path="Qwen/Qwen3.6-27B")

Requires: pip install transformers torch accelerate
"""

import json
import re
from typing import Dict, List, Optional

_model_cache: dict = {}


def _load_model(model_name_or_path: str):
    if model_name_or_path in _model_cache:
        return _model_cache[model_name_or_path]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    _model_cache[model_name_or_path] = (model, tokenizer)
    return model, tokenizer


def _extract_json(raw_text: str) -> List[Dict]:
    text = raw_text.strip()
    for marker in ("</think>",):
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
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


def call_llm_local(
    prompt: str,
    model_name_or_path: str = "Qwen/Qwen3.6-27B",
    max_new_tokens: int = 8192,
    temperature: float = 0.85,
) -> List[Dict]:
    import torch

    model, tokenizer = _load_model(model_name_or_path)
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _extract_json(raw_text)
