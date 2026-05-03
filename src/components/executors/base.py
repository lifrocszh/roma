"""Shared executor helpers for Stage 2."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.core.models import Task, TaskType
from src.core.registry import ExecutorToolView, ToolError
from src.core.signatures import Executor, ExecutorOutput


class BaseHeuristicExecutor(Executor):
    """Common base for deterministic Stage 2 executors."""

    def __init__(self, *, prompt: str, supported_task_types: frozenset[TaskType]) -> None:
        self.prompt = prompt
        self._supported_task_types = supported_task_types
        self._tools: ExecutorToolView | None = None
        self._client = self._build_client()

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return self._supported_task_types

    def set_tools(self, tools: ExecutorToolView) -> None:
        self._tools = tools

    def _build_client(self) -> OpenAI | None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = os.getenv("ROMA_BASE_URL") or "https://api.deepseek.com"
        return OpenAI(api_key=api_key, base_url=base_url)

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
        except Exception:
            return None
