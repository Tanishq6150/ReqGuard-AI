"""Load prompt templates from disk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values: object) -> str:
    template = load_prompt(name)
    for key, value in values.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
    return template
