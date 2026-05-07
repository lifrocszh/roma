"""Structured LLM inference with Chain-of-Thought, few-shot support, and JSON output."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv(Path("config/.env"))

def _resolve_key(*names: str) -> str | None:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_client() -> OpenAI | None:
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


_DEFAULT_MODEL = _resolve_key("ROMA_MODEL", "openrouter_model") or "deepseek-v4-flash"
print(f"default model: {_DEFAULT_MODEL}")


def llm_judge(
    *,
    system: str,
    instructions: str,
    context: str = "",
    output_schema: dict[str, str],
    examples: list[tuple[str, str]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    client = _build_client()
    if client is None:
        return None

    chosen_model = model or _DEFAULT_MODEL
    schema_lines = "\n".join(f'  "{k}": "{v}"' for k, v in output_schema.items())

    messages = [
        {
            "role": "system",
            "content": (
                f"{system}\n\n"
                "You MUST respond with valid JSON only, containing exactly these fields:\n"
                f"{schema_lines}\n\n"
                "Think step by step before producing your final answer. "
                "Your reasoning must be placed in the \"reasoning\" field. "
                "Then provide each requested field with its value."
            ),
        },
    ]

    if examples:
        for inp, out in examples:
            messages.append({"role": "user", "content": inp})
            messages.append({"role": "assistant", "content": out})

    if context.strip():
        user_content = f"Task: {instructions}\n\nContext:\n{context}"
    else:
        user_content = f"Task: {instructions}"

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 2048,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            return None
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        import logging
        logging.getLogger("roma.inference").warning(
            "LLM call failed: model=%s", chosen_model, exc_info=True
        )
        return None


def llm_decision(
    *,
    system: str,
    instructions: str,
    context: str = "",
    output_schema: dict[str, str],
    examples: list[tuple[str, str]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    return llm_judge(
        system=system,
        instructions=instructions,
        context=context,
        output_schema=output_schema,
        examples=examples,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


def llm_freeform(
    *,
    system: str,
    instructions: str,
    context: str = "",
    temperature: float = 0.3,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str | None:
    client = _build_client()
    if client is None:
        return None

    chosen_model = model or _DEFAULT_MODEL
    messages = [
        {"role": "system", "content": system},
    ]

    if context.strip():
        user_content = f"Task: {instructions}\n\nContext:\n{context}"
    else:
        user_content = f"Task: {instructions}"
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) and content.strip() else None
    except Exception:
        import logging
        logging.getLogger("roma.inference").warning(
            "LLM freeform call failed: model=%s", chosen_model, exc_info=True
        )
        return None
