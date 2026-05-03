"""Write executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_WRITE_PROMPT


class WriteExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_WRITE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.WRITE}))

    def execute(self, task: Task) -> ExecutorOutput:
        context = task.context_input or ""
        llm_result = self._llm_text(
            system=self.prompt,
            user=(
                f"Task goal: {task.goal}\n"
                f"Task type: {task.task_type.value}\n"
                f"Context: {context}\n"
                "Write the final response."
            ),
            temperature=0.5,
        )
        if llm_result:
            result = llm_result
            strategy = "llm writing"
        elif any(token in task.goal.lower() for token in ["story", "scene", "chapter", "narrative"]):
            result = (
                f"Narrative draft for '{task.goal}': "
                f"Set the scene clearly, maintain continuity, and resolve the requested beat. "
                f"Context: {context}".strip()
            )
            strategy = "creative writing synthesis fallback"
        else:
            result = (
                f"Written response for '{task.goal}': "
                f"Present the answer with a clear structure, useful detail, and concise prose. "
                f"Context: {context}".strip()
            )
            strategy = "structured prose synthesis fallback"
        return self._result(task=task, result=result, strategy=strategy)
