"""Write executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_WRITE_PROMPT


class WriteExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_WRITE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.WRITE}), temperature=0.3)

    def execute(self, task: Task) -> ExecutorOutput:
        context = task.context_input or ""
        result = self._generate(instructions=task.goal, context=context)
        return self._result(
            task=task,
            result=result or f"Written response for '{task.goal}': {context}",
            strategy="llm writing" if result else "fallback",
        )
