"""Think executor."""

from __future__ import annotations

import re

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_THINK_PROMPT


_MATH_KEYWORDS = ["calculate", "compute", "evaluate", "what is ", "solve ", "how many ", "how much "]


def _has_math_goal(goal: str) -> bool:
    lower = goal.lower()
    if any(kw in lower for kw in _MATH_KEYWORDS):
        return True
    if re.search(r"\d+", goal):
        return True
    return False


def _extract_expression(goal: str) -> str | None:
    expr = goal.strip()
    for prefix in ["what is ", "calculate ", "compute ", "evaluate ", "solve ", "find ", "simplify ", "what's "]:
        if expr.lower().startswith(prefix):
            expr = expr[len(prefix):]
            break
    expr = expr.strip().rstrip("?.,!;:")
    expr = re.sub(r"\b(to\s+)?the\s+power\s+of\b", " ** ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\btimes\b", " * ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bdivided by\b", " / ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bplus\b", " + ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bminus\b", " - ", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bsquared\b", " ** 2", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bcubed\b", " ** 3", expr, flags=re.IGNORECASE)
    if re.search(r"\d+\s*[+\-*/^][\s\S]{0,100}", expr):
        return expr
    return None


class ThinkExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_THINK_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.THINK, TaskType.GENERAL}), temperature=0.0)

    def execute(self, task: Task) -> ExecutorOutput:
        context = task.context_input or ""
        goal = task.goal

        expr = _extract_expression(goal) if _has_math_goal(goal) else None
        if expr is not None:
            try:
                calc = self._tool("calculator")
                tool_result = calc.invoke(expression=expr)
                if tool_result.ok:
                    answer_text = f"Calculation: {expr} = {tool_result.output}"
                    result = self._reason(
                        instructions=f"{goal}\n\n{answer_text}",
                        context=context,
                    )
                    return self._result(
                        task=task,
                        result=result or f"The answer is {tool_result.output}.",
                        strategy="calculator with llm synthesis" if result else "calculator direct",
                        extra={"used_calculator": True, "expression": expr, "calc_result": tool_result.output},
                    )
            except ToolError:
                pass

        result = self._reason(instructions=goal, context=context)
        return self._result(
            task=task,
            result=result or f"Analysis for '{goal}': {context}",
            strategy="llm reasoning" if result else "fallback",
            extra={"llm_used": result is not None},
        )
