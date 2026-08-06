"""Persistence helpers for user-editable agent prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


PROMPT_SETTINGS_FILE = Path("prompt_settings.json")


def load_prompt_settings(path: Path = PROMPT_SETTINGS_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def save_prompt_settings(
    prompts: Mapping[str, str], path: Path = PROMPT_SETTINGS_FILE
) -> None:
    path.write_text(
        json.dumps(dict(prompts), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
