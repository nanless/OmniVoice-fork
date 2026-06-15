"""LLM client wrapper — reuses children generator implementation."""

import sys
from pathlib import Path

_CHILD_GEN = Path(__file__).resolve().parents[2] / "text_generation"
if str(_CHILD_GEN) not in sys.path:
    sys.path.insert(0, str(_CHILD_GEN))

from llm_generate_texts import call_llm  # noqa: E402

__all__ = ["call_llm"]
