"""Re-export OmniVoice non-verbal tag definitions from the children generator."""

import sys
from pathlib import Path

_CHILD_GEN = Path(__file__).resolve().parents[2] / "text_generation"
if str(_CHILD_GEN) not in sys.path:
    sys.path.insert(0, str(_CHILD_GEN))

import llm_generate_texts as _child  # noqa: E402

TAG_DEFINITIONS = _child.TAG_DEFINITIONS
VALID_TAG_NAMES = _child.VALID_TAG_NAMES
EMOTION_PROFILES = _child.EMOTION_PROFILES
TAG_DENSITY_MAP = _child.TAG_DENSITY_MAP

__all__ = [
    "TAG_DEFINITIONS",
    "VALID_TAG_NAMES",
    "EMOTION_PROFILES",
    "TAG_DENSITY_MAP",
]
