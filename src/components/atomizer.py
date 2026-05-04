"""Stage 2 Atomizer implementation."""

from __future__ import annotations

import re

from src.core.models import NodeType, Task, TaskType
from src.core.signatures import Atomizer, AtomizerDecision
from src.prompts.seed_prompts import ATOMIZER_PROMPT


class DefaultAtomizer(Atomizer):
    """Heuristic atomizer aligned with ROMA's atomic/non-atomic split."""

    def __init__(self, *, prompt: str = ATOMIZER_PROMPT) -> None:
        self.prompt = prompt

    def decide(self, task: Task) -> AtomizerDecision:
        if task.metadata.get("force_node_type") == NodeType.EXECUTE.value:
            return AtomizerDecision(
                node_type=NodeType.EXECUTE,
                rationale="planner marked this subtask as an executor-ready leaf",
            )

        goal = task.goal.strip()
        lower_goal = goal.lower()

        reasons: list[str] = []
        plan = False

        if task.dependencies:
            plan = True
            reasons.append("task already carries dependencies")

        if any(token in lower_goal for token in [" and ", " then ", " after ", " before ", "compare ", "versus "]):
            plan = True
            reasons.append("goal implies multi-step sequencing or multiple outputs")

        if any(token in lower_goal for token in ["outline", "plan", "roadmap", "strategy", "workflow"]):
            plan = True
            reasons.append("goal requests planning or structured decomposition")

        if task.task_type == TaskType.WRITE and (
            any(token in lower_goal for token in ["chapter", "story arc", "world-building", "character arc", "multi-part"])
            or len(goal.split()) > 40
        ):
            plan = True
            reasons.append("writing task benefits from foundation, development, and synthesis")

        if task.task_type == TaskType.CODE and any(token in lower_goal for token in ["test", "deploy", "refactor", "build pipeline"]):
            plan = True
            reasons.append("code task implies staged work rather than one atomic action")

        if re.search(r"\b(first|second|third|finally)\b", lower_goal):
            plan = True
            reasons.append("explicit ordered stages detected")

        if task.task_type in (TaskType.RETRIEVE,):
            plan = True
            reasons.append("retrieval tasks benefit from query, gather, and synthesize stages")

        if any(token in lower_goal for token in ["search", " fetch ", "http", "www.", ".com", "url", "look up"]):
            plan = True
            reasons.append("goal requires searching or fetching external data")

        if plan:
            return AtomizerDecision(
                node_type=NodeType.PLAN,
                rationale="; ".join(reasons) if reasons else "task needs decomposition",
            )

        return AtomizerDecision(
            node_type=NodeType.EXECUTE,
            rationale="single deliverable appears solvable by one executor pass",
        )
