"""Think executor."""

from __future__ import annotations

import re

from src.components.executors.base import BaseHeuristicExecutor
from src.core.models import Task, TaskType
from src.core.registry import ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_THINK_PROMPT


# Keywords that suggest a math computation task
_MATH_KEYWORDS = ["calculate", "compute", "evaluate", "what is ", "solve ", "how many ", "how much "]


def _has_math_goal(goal: str) -> bool:
    """Return ``True`` if *goal* looks like a math computation task."""
    lower = goal.lower()
    if any(kw in lower for kw in _MATH_KEYWORDS):
        return True
    # Any explicit numbers suggest potential calculation
    if re.search(r"\d+", goal):
        return True
    return False


def _extract_expression(goal: str) -> str | None:
    expr = goal.strip()
    # Remove common prefixes
    for prefix in ["what is ", "calculate ", "compute ", "evaluate ", "solve ", "find ", "simplify ", "what's "]:
        if expr.lower().startswith(prefix):
            expr = expr[len(prefix):]
            break
    # Remove trailing punctuation
    expr = expr.strip().rstrip("?.,!;:")

    # Translate natural-language patterns to Python syntax
    expr = re.sub(
        r"\b(to\s+)?the\s+power\s+of\b",
        " ** ",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\btimes\b",
        " * ",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\bdivided by\b",
        " / ",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\bplus\b",
        " + ",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\bminus\b",
        " - ",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\bsquared\b",
        " ** 2",
        expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(
        r"\bcubed\b",
        " ** 3",
        expr,
        flags=re.IGNORECASE,
    )

    # If what remains is a plausible expression, return it
    if re.search(r"\d+\s*[+\-*/^][\s\S]{0,100}", expr):
        return expr
    return None


class ThinkExecutor(BaseHeuristicExecutor):
    def __init__(self, *, prompt: str = EXECUTOR_THINK_PROMPT) -> None:
        super().__init__(prompt=prompt, supported_task_types=frozenset({TaskType.THINK, TaskType.GENERAL}))

    def execute(self, task: Task) -> ExecutorOutput:
        context = task.context_input or "No external context provided."
        goal = task.goal

        # --- If the goal looks like a math problem, try the calculator ---
        expr = _extract_expression(goal) if _has_math_goal(goal) else None
        if expr is not None:
            try:
                calc = self._tool("calculator")
                tool_result = calc.invoke(expression=expr)

                if tool_result.ok:
                    answer = tool_result.output
                    llm_result = self._llm_text(
                        system=self.prompt,
                        user=(
                            f"Task goal: {goal}\n"
                            f"Calculation result: {expr} = {answer}\n"
                            "Present the answer in a clear, conversational way."
                        ),
                        temperature=0.2,
                    )
                    result = llm_result or f"The answer is {answer}."
                    return self._result(
                        task=task,
                        result=result,
                        strategy="calculator with llm synthesis" if llm_result else "calculator direct",
                        extra={"used_calculator": True, "expression": expr, "calc_result": answer, "llm_used": llm_result is not None},
                    )
            except ToolError:
                pass  # calculator not available, fall through to LLM

        llm_result = self._llm_text(
            system=self.prompt,
            user=(
                f"Task goal: {goal}\n"
                f"Task type: {task.task_type.value}\n"
                f"Context: {context}\n"
                "Produce a concise, direct answer."
            ),
        )
        result = llm_result or (
            f"Analysis for '{goal}': "
            f"Identify the core objective, use the available context, and return a concise conclusion. "
            f"Context considered: {context}"
        )
        return self._result(
            task=task,
            result=result,
            strategy="llm reasoning" if llm_result else "single-pass reasoning fallback",
            extra={"llm_used": llm_result is not None},
        )
