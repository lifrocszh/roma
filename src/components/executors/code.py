"""Code executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_CODE_PROMPT


class CodeExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_CODE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.CODE}), temperature=0.1)

    def execute(self, task: Task) -> ExecutorOutput:
        code = task.metadata.get("code")
        language = task.metadata.get("language", "python")
        if code:
            try:
                sandbox = self._tool("code_sandbox")
                tool_result = sandbox.invoke(language=language, code=code)
                return self._result(
                    task=task,
                    result=f"Sandbox execution result for '{task.goal}': {tool_result.output}",
                    strategy="sandbox execution",
                    extra={"language": language, "tool_ok": tool_result.ok},
                )
            except ToolError:
                pass

        result = self._generate(
            instructions=f"Provide code or a code-oriented response for: {task.goal}",
            context=f"Language: {language}",
        )
        return self._result(
            task=task,
            result=result or f"Code response for '{task.goal}' (language: {language})",
            strategy="llm code generation" if result else "fallback",
            extra={"language": language},
        )
