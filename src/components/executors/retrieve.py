"""Retrieve executor."""

from __future__ import annotations

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_RETRIEVE_PROMPT


class RetrieveExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_RETRIEVE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.RETRIEVE}))

    def execute(self, task: Task) -> ExecutorOutput:
        query = task.goal
        try:
            search = self._tool("web_search")
            tool_result = search.invoke(query=query)
            llm_result = self._llm_text(
                system=self.prompt,
                user=(
                    f"Task goal: {task.goal}\n"
                    f"Context: {task.context_input or ''}\n"
                    f"Search results: {tool_result.output}\n"
                    "Summarize the useful evidence."
                ),
                temperature=0.2,
            )
            result = llm_result or f"Evidence gathered for '{query}': {tool_result.output}"
            return self._result(
                task=task,
                result=result,
                strategy="search-backed retrieval with llm synthesis" if llm_result else "search-backed retrieval",
                artifacts=[],
                extra={"tool_ok": tool_result.ok, "query": query, "llm_used": llm_result is not None},
            )
        except ToolError:
            fallback = task.context_input or f"No search tool available. Use task statement directly: {query}"
            llm_result = self._llm_text(
                system=self.prompt,
                user=(
                    f"Task goal: {task.goal}\n"
                    f"Context: {fallback}\n"
                    "Summarize the useful evidence or say what is missing."
                ),
                temperature=0.2,
            )
            return self._result(
                task=task,
                result=llm_result or f"Fallback evidence for '{query}': {fallback}",
                strategy="context fallback retrieval with llm synthesis" if llm_result else "context fallback retrieval",
                extra={"query": query, "llm_used": llm_result is not None},
            )
