"""Shared executor helpers for Stage 2."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.core.models import Task, TaskType
from src.core.registry import ExecutorToolView, ToolError
from src.core.signatures import Executor, ExecutorOutput


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_openai_client() -> OpenAI | None:
    """Build an OpenAI-compatible client from environment variables.

    Resolution order:
    1. ``ROMA_API_KEY`` / ``ROMA_BASE_URL`` — fully custom
    2. ``OPENROUTER_API_KEY`` → base URL ``https://openrouter.ai/api/v1``
    3. ``DEEPSEEK_API_KEY`` → base URL ``https://api.deepseek.com``
    4. ``OPENAI_API_KEY`` → base URL ``https://api.openai.com`` (OpenAI default)
    """
    api_key = os.getenv("ROMA_API_KEY")
    base_url = os.getenv("ROMA_BASE_URL")

    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(
            api_key=openrouter_key,
            base_url=base_url or _OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/roma"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "ROMA"),
            },
        )

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return OpenAI(api_key=deepseek_key, base_url=base_url or "https://api.deepseek.com")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key, base_url=base_url or "https://api.openai.com/v1")

    return None


class BaseHeuristicExecutor(Executor):
    """Common base for deterministic Stage 2 executors."""

    def __init__(self, *, prompt: str, supported_task_types: frozenset[TaskType]) -> None:
        self.prompt = prompt
        self._supported_task_types = supported_task_types
        self._tools: ExecutorToolView | None = None
        self._client = _build_openai_client()

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return self._supported_task_types

    def set_tools(self, tools: ExecutorToolView) -> None:
        self._tools = tools

    def _tool(self, name: str):
        if self._tools is None:
            raise ToolError(f"executor has no tool view bound; requested {name!r}")
        return self._tools.get(name)

    def _base_metadata(self, *, task: Task, strategy: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = {
            "strategy": strategy,
            "task_type": task.task_type.value,
            "used_tools": self._tools.list_names() if self._tools else [],
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _result(self, *, task: Task, result: str, strategy: str, artifacts: list[str] | None = None, extra: dict[str, Any] | None = None) -> ExecutorOutput:
        return ExecutorOutput(
            task_id=task.id,
            result=result,
            artifacts=artifacts or [],
            metadata=self._base_metadata(task=task, strategy=strategy, extra=extra),
        )

    def _llm_text(self, *, system: str, user: str, model: str | None = None, temperature: float = 0.2, max_tokens: int | None = None) -> str | None:
        if self._client is None:
            return None

        chosen_model = model or os.getenv("ROMA_MODEL", "deepseek-v4-flash")
        try:
            response = self._client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            content = choice.message.content if choice and choice.message else None
            return content.strip() if isinstance(content, str) and content.strip() else None
        except Exception as exc:
            import logging
            logging.getLogger("roma.executor").warning(
                "LLM call failed: model=%s error=%s", chosen_model, exc
            )
            return None
