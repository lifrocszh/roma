"""Think executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_THINK_PROMPT


class ThinkExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_THINK_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.THINK, TaskType.GENERAL}))

    def execute(self, task: Task) -> ExecutorOutput:
        context = task.context_input or "No external context provided."
        llm_result = self._llm_text(
            system=self.prompt,
            user=(
                f"Task goal: {task.goal}\n"
                f"Task type: {task.task_type.value}\n"
                f"Context: {context}\n"
                "Produce a concise, direct answer."
            ),
        )
        result = llm_result or (
            f"Analysis for '{task.goal}': "
            f"Identify the core objective, use the available context, and return a concise conclusion. "
            f"Context considered: {context}"
        )
        return self._result(
            task=task,
            result=result,
            strategy="llm reasoning" if llm_result else "single-pass reasoning fallback",
            extra={"llm_used": llm_result is not None},
        )
