"""Retrieve executor — dispatches to search or extract based on goal content."""

from __future__ import annotations

import re

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_RETRIEVE_PROMPT


# Regex to match http(s):// URLs or bare hostnames with a known TLD (e.g. example.com/path)
# Use single quotes to avoid escaping issues inside the raw string
_COMMON_TLDS = r'(?:com|org|net|edu|gov|mil|int|io|ai|app|dev|wiki|co|uk|de|jp|fr|au|ca|cn|in|ru|br|eu)'
_URL_RE = re.compile(
    r'\bhttps?://[^\s<>"\']+'  r'|'
    r'\b(?:[a-zA-Z0-9-]+\.)+' + _COMMON_TLDS + r'\b(?:/[^\s<>"\']*)?'
    , re.IGNORECASE
)


def _extract_urls(text: str) -> list[str]:
    """Return all URLs found in *text*, deduplicated and in order of appearance."""
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
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.RETRIEVE}))

    @staticmethod
    def _search_query(goal: str, max_chars: int = 350) -> str:
        """Extract a concise search query from the task goal.

        The goal typically includes few-shot examples followed by the actual question.
        We take the last ``max_chars`` characters to get just the question.
        """
        if len(goal) <= max_chars:
            return goal
        # Take the tail end where the actual question lives
        tail = goal[-max_chars:]
        # Try to break at a sentence boundary
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

            # --- If the goal contains a concrete URL, extract its content ---
            if urls:
                tool_result = tool.invoke(action="extract", url=urls[0])
                if tool_result.ok:
                    llm_result = self._llm_text(
                        system=self.prompt,
                        user=(
                            f"Task goal: {task.goal}\n"
                            f"Context: {task.context_input or ''}\n"
                            f"Page content:\n{tool_result.output}\n\n"
                            "Answer the task goal based on the extracted page content above."
                        ),
                        temperature=0.2,
                    )
                    result = llm_result or (
                        f"Extracted content from {urls[0]}:\n{tool_result.output}"
                    )
                    return self._result(
                        task=task,
                        result=result,
                        strategy="url extraction with llm synthesis" if llm_result else "url extraction",
                        artifacts=[],
                        extra={
                            "tool_ok": tool_result.ok,
                            "url": urls[0],
                            "action": "extract",
                            "llm_used": llm_result is not None,
                        },
                    )

            # --- Otherwise, do a full web search ---
            search_for = self._search_query(query)
            tool_result = tool.invoke(action="search", query=search_for)
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
