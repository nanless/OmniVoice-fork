"""Load .env for adult text generation."""

import os
from pathlib import Path

_ADULT_DIR = Path(__file__).resolve().parents[1]
_CHILD_ENV = Path(__file__).resolve().parents[2] / "text_generation" / ".env"


def load_env_file() -> None:
    for path in (_ADULT_DIR / ".env", _CHILD_ENV):
        if not path.is_file():
            continue
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
