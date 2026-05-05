"""Shared executor helpers for Stage 2."""

from __future__ import annotations

from typing import Any

from src.core.inference import llm_judge, llm_freeform
from src.core.models import Task, TaskType
from src.core.registry import ExecutorToolView, ToolError
from src.core.signatures import Executor, ExecutorOutput


class BaseHeuristicExecutor(Executor):
    """Common base for Stage 2 executors with LLM-powered reasoning."""

    def __init__(self, *, prompt: str, supported_task_types: frozenset[TaskType], temperature: float = 0.0) -> None:
        self.prompt = prompt
        self._supported_task_types = supported_task_types
        self._tools: ExecutorToolView | None = None
        self._temperature = temperature

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

    def _reason(
        self,
        *,
        instructions: str,
        context: str = "",
        temperature: float | None = None,
    ) -> str | None:
        """Structured reasoning with Chain-of-Thought. Returns the answer text."""
        result = llm_judge(
            system=self.prompt,
            instructions=instructions,
            context=context,
            output_schema={
                "reasoning": "Step-by-step reasoning process",
                "answer": "The final answer or response",
            },
            temperature=temperature if temperature is not None else self._temperature,
        )
        if result and result.get("answer"):
            return result["answer"]
        return None

    def _generate(
        self,
        *,
        instructions: str,
        context: str = "",
        temperature: float | None = None,
    ) -> str | None:
        """Free-form text generation without structured output."""
        return llm_freeform(
            system=self.prompt,
            instructions=instructions,
            context=context,
            temperature=temperature if temperature is not None else self._temperature,
        )
