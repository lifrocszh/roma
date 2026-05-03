"""Code executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_CODE_PROMPT


class CodeExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_CODE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.CODE}))

    def execute(self, task: Task) -> ExecutorOutput:
        code = task.metadata.get("code")
        language = task.metadata.get("language", "python")
        if code:
            try:
                sandbox = self._tool("code_sandbox")
                tool_result = sandbox.invoke(language=language, code=code)
                result = f"Sandbox execution result for '{task.goal}': {tool_result.output}"
                return self._result(
                    task=task,
                    result=result,
                    strategy="sandbox execution",
                    extra={"language": language, "tool_ok": tool_result.ok},
                )
            except ToolError:
                pass

        result = (
            f"Code-oriented response for '{task.goal}': "
            f"no runnable snippet was provided, so return a minimal implementation plan or code sketch."
        )
        return self._result(task=task, result=result, strategy="code planning fallback", extra={"language": language})
