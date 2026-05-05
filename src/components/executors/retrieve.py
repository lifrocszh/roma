"""Retrieve executor."""

from __future__ import annotations

import re

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_RETRIEVE_PROMPT


_COMMON_TLDS = r'(?:com|org|net|edu|gov|mil|int|io|ai|app|dev|wiki|co|uk|de|jp|fr|au|ca|cn|in|ru|br|eu)'
_URL_RE = re.compile(
    r'\bhttps?://[^\s<>"\']+' r'|'
    r'\b(?:[a-zA-Z0-9-]+\.)+' + _COMMON_TLDS + r'\b(?:/[^\s<>"\']*)?',
    re.IGNORECASE,
)


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        raw = match.group(0).rstrip(".):;!,-\"'")
        if raw not in seen:
            seen.add(raw)
            urls.append(raw)
    return urls


class RetrieveExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_RETRIEVE_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.RETRIEVE}), temperature=0.1)

    @staticmethod
    def _search_query(goal: str, max_chars: int = 350) -> str:
        if len(goal) <= max_chars:
            return goal
        tail = goal[-max_chars:]
        for delim in ("\nQ: ", "\nPlease answer", "\nOptions"):
            idx = tail.find(delim)
            if idx != -1:
                return tail[idx:]
        return tail

    def execute(self, task: Task) -> ExecutorOutput:
        query = task.goal
        urls = _extract_urls(query)

        try:
            tool = self._tool("web_search")

            if urls:
                tool_result = tool.invoke(action="extract", url=urls[0])
                if tool_result.ok:
                    result = self._reason(
                        instructions=f"Answer based on the extracted page content. Task: {query}",
                        context=tool_result.output,
                    )
                    return self._result(
                        task=task,
                        result=result or f"Extracted content from {urls[0]}:\n{tool_result.output}",
                        strategy="url extraction with llm synthesis" if result else "url extraction",
                        extra={"url": urls[0], "action": "extract"},
                    )

            search_for = self._search_query(query)
            tool_result = tool.invoke(action="search", query=search_for)
            result = self._reason(
                instructions=f"Answer based on search results. Task: {query}",
                context=tool_result.output,
            )
            return self._result(
                task=task,
                result=result or f"Evidence gathered: {tool_result.output}",
                strategy="search-backed retrieval" if result else "search fallback",
                extra={"query": query},
            )

        except ToolError:
            fallback = task.context_input or query
            result = self._reason(
                instructions=f"Answer based on available context. Task: {query}",
                context=fallback,
            )
            return self._result(
                task=task,
                result=result or f"Fallback evidence for '{query}'",
                strategy="context fallback" if result else "fallback",
                extra={"query": query},
            )
