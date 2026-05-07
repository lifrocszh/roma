"""Atomizer — decides PLAN vs EXECUTE using the LLM, with depth guard."""

from __future__ import annotations

from src.core.inference import llm_judge
from src.core.models import NodeType, Task, TaskType
from src.core.signatures import AtomizerDecision
from src.prompts.seed_prompts import ATOMIZER_PROMPT


_MAX_PLAN_DEPTH = 2


class DefaultAtomizer:
    def __init__(self, *, prompt: str = ATOMIZER_PROMPT) -> None:
        self.prompt = prompt

    def decide(self, task: Task) -> AtomizerDecision:
        if task.metadata.get("force_node_type") == NodeType.EXECUTE.value:
            return AtomizerDecision(node_type=NodeType.EXECUTE, rationale="marked as leaf by planner")

        if task.task_type in (TaskType.RETRIEVE, TaskType.CODE):
            return AtomizerDecision(node_type=NodeType.EXECUTE, rationale=f"{task.task_type.value} is always single-pass")

        depth = task.metadata.get("_depth", 0)
        if isinstance(depth, int) and depth >= _MAX_PLAN_DEPTH:
            return AtomizerDecision(node_type=NodeType.EXECUTE, rationale=f"max plan depth {_MAX_PLAN_DEPTH} reached")

        result = llm_judge(
            system=self.prompt,
            instructions=task.goal,
            context=task.context_input or "",
            output_schema={
                "reasoning": "Step-by-step analysis of whether this task needs decomposition",
                "node_type": "PLAN if the task requires decomposition, coordination, or multiple steps; EXECUTE if one pass suffices",
                "rationale": "Brief explanation of the decision",
            },
        )

        if result and result.get("node_type") == "PLAN":
            return AtomizerDecision(node_type=NodeType.PLAN, rationale=result.get("rationale", "task needs decomposition"))

        return AtomizerDecision(
            node_type=NodeType.EXECUTE,
            rationale=result.get("rationale", "single pass sufficient") if result else "fallback to execute",
        )
