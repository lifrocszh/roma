from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv(Path("config/.env"))


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = os.getenv("ROMA_MODEL", os.getenv("openrouter_model")) or "deepseek-v4-flash"


def _resolve_key(*names: str) -> str | None:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None


def build_client() -> OpenAI | None:
    base_url = os.getenv("ROMA_BASE_URL")

    api_key = _resolve_key("ROMA_API_KEY")
    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    openrouter_key = _resolve_key("OPENROUTER_API_KEY", "openrouter_api_key")
    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/roma"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "ROMA"),
            },
        )

    deepseek_key = _resolve_key("DEEPSEEK_API_KEY", "deepseek_api_key")
    if deepseek_key:
        return OpenAI(api_key=deepseek_key, base_url=base_url or "https://api.deepseek.com")

    openai_key = _resolve_key("OPENAI_API_KEY", "openai_api_key")
    if openai_key:
        return OpenAI(api_key=openai_key, base_url=base_url or "https://api.openai.com/v1")

    return None


def get_default_model() -> str:
    print(f"Default model: {_DEFAULT_MODEL}")
    return _DEFAULT_MODEL
