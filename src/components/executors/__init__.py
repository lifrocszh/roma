"""Unified executor — handles all task types with per-type tool access."""

from __future__ import annotations

import re
from typing import Any

from src.core.inference import llm_judge, llm_freeform
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import (
    EXECUTOR_CODE_PROMPT,
    EXECUTOR_RETRIEVE_PROMPT,
    EXECUTOR_THINK_PROMPT,
    EXECUTOR_WRITE_PROMPT,
)


_PROMPTS = {
    TaskType.THINK: EXECUTOR_THINK_PROMPT,
    TaskType.GENERAL: EXECUTOR_THINK_PROMPT,
    TaskType.WRITE: EXECUTOR_WRITE_PROMPT,
    TaskType.RETRIEVE: EXECUTOR_RETRIEVE_PROMPT,
    TaskType.CODE: EXECUTOR_CODE_PROMPT,
}

_TEMPERATURES = {
    TaskType.THINK: 0.0,
    TaskType.GENERAL: 0.0,
    TaskType.WRITE: 0.3,
    TaskType.RETRIEVE: 0.1,
    TaskType.CODE: 0.0,
}

_TOP_TLDS = r"(?:com|org|net|edu|gov|mil|int|io|ai|app|dev|wiki|co|uk|de|jp|fr|au|ca|cn|in|ru|br|eu)"
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"']+|\b(?:[a-zA-Z0-9-]+\.)+" + _TOP_TLDS + r"\b(?:/[^\s<>\"']*)?", re.IGNORECASE)


class UnifiedExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(TaskType)

    def set_tools(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    def _tool(self, name: str):
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"tool {name!r} not available")

    def execute(self, task: Task) -> ExecutorOutput:
        tt = task.task_type
        prompt = _PROMPTS.get(tt, EXECUTOR_THINK_PROMPT)
        temperature = _TEMPERATURES.get(tt, 0.0)

        if tt in (TaskType.RETRIEVE,):
            return self._execute_retrieve(task, prompt, temperature)
        if tt in (TaskType.CODE,):
            return self._execute_code(task, prompt, temperature)
        if tt in (TaskType.WRITE,):
            return self._execute_write(task, prompt, temperature)
        return self._execute_think(task, prompt, temperature)

    def _execute_think(self, task: Task, prompt: str, temperature: float) -> ExecutorOutput:
        result = llm_judge(
            system=prompt,
            instructions=task.goal,
            context=task.context_input or "",
            output_schema={"reasoning": "Step-by-step reasoning process", "answer": "The final answer or response"},
            temperature=temperature,
        )
        text = str(result["answer"]) if result and result.get("answer") is not None else f"Analysis for '{task.goal}'"
        return ExecutorOutput(task_id=task.id, result=text, metadata={"strategy": "think"})

    def _execute_write(self, task: Task, prompt: str, temperature: float) -> ExecutorOutput:
        result = llm_freeform(system=prompt, instructions=task.goal, context=task.context_input or "", temperature=temperature)
        return ExecutorOutput(task_id=task.id, result=result or f"Written response for '{task.goal}'", metadata={"strategy": "write"})

    def _execute_retrieve(self, task: Task, prompt: str, temperature: float) -> ExecutorOutput:
        query = task.goal
        urls = list(dict.fromkeys(m.group(0).rstrip(".).:;!,-\"'") for m in _URL_RE.finditer(query)))

        try:
            tool = self._tool("web_search")

            if urls:
                tr = tool.invoke(action="extract", url=urls[0])
                if tr.ok:
                    result = llm_judge(
                        system=prompt, instructions=f"Answer based on the page content. Task: {query}", context=tr.output,
                        output_schema={"reasoning": "Analysis", "answer": "The answer"},
                        temperature=temperature,
                    )
                    text = str(result["answer"]) if result and result.get("answer") is not None else tr.output
                    return ExecutorOutput(task_id=task.id, result=text, metadata={"strategy": "retrieve_url"})

            tr = tool.invoke(action="search", query=query[:350])
            result = llm_judge(
                system=prompt, instructions=f"Answer based on search results. Task: {query}", context=tr.output,
                output_schema={"reasoning": "Analysis", "answer": "The answer"},
                temperature=temperature,
            )
            text = str(result["answer"]) if result and result.get("answer") is not None else tr.output
            return ExecutorOutput(task_id=task.id, result=text, metadata={"strategy": "retrieve_search"})

        except ToolError:
            result = llm_judge(
                system=prompt, instructions=f"Answer based on available context. Task: {query}", context=task.context_input or query,
                output_schema={"reasoning": "Analysis", "answer": "The answer"},
                temperature=temperature,
            )
            text = str(result["answer"]) if result and result.get("answer") is not None else f"Fallback for '{query}'"
            return ExecutorOutput(task_id=task.id, result=text, metadata={"strategy": "retrieve_fallback"})

    def _execute_code(self, task: Task, prompt: str, temperature: float) -> ExecutorOutput:
        code = task.metadata.get("code")
        if not code:
            code = llm_freeform(
                system=prompt,
                instructions=f"Write Python code for: {task.goal}",
                context=f"Available data:\n{task.context_input or ''}\n\nOutput ONLY the code. Use print() for the result.",
                temperature=0.0,
            )
            if code:
                cleaned = code.strip()
                for m in ["```python", "```"]:
                    if m in cleaned:
                        parts = cleaned.split(m)
                        cleaned = parts[1] if len(parts) > 1 else parts[0]
                        if m in cleaned:
                            cleaned = cleaned.split(m)[0]
                code = cleaned.strip()

        if code:
            try:
                sandbox = self._tool("code_sandbox")
                is_expr = False
                try:
                    compile(code, "<string>", "eval")
                    is_expr = True
                except SyntaxError:
                    pass
                exec_code = f"import math; print({code})" if is_expr else ("import math\n" + code if "print(" not in code else "import math\n" + code)
                tr = sandbox.invoke(language="python", code=exec_code)
                if tr.ok and tr.output.strip():
                    return ExecutorOutput(task_id=task.id, result=tr.output.strip(), metadata={"strategy": "sandbox", "code": code[:200]})
            except ToolError:
                pass

        return ExecutorOutput(task_id=task.id, result=f"Computation for '{task.goal}': {task.context_input or ''}", metadata={"strategy": "code_fallback"})


__all__ = ["UnifiedExecutor"]
