"""Stage 2 Atomizer implementation - LLM-driven."""

from __future__ import annotations

from src.core.inference import llm_judge
from src.core.models import NodeType, Task, TaskType
from src.core.signatures import Atomizer, AtomizerDecision
from src.prompts.seed_prompts import ATOMIZER_PROMPT


class DefaultAtomizer(Atomizer):
    """LLM-driven atomizer that decides PLAN vs EXECUTE."""

    def __init__(self, *, prompt: str = ATOMIZER_PROMPT) -> None:
        self.prompt = prompt

    def decide(self, task: Task) -> AtomizerDecision:
        if task.metadata.get("force_node_type") == NodeType.EXECUTE.value:
            return AtomizerDecision(
                node_type=NodeType.EXECUTE,
                rationale="planner marked this subtask as executor-ready leaf",
            )

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
            return AtomizerDecision(
                node_type=NodeType.PLAN,
                rationale=result.get("rationale", "task needs decomposition"),
            )

        return AtomizerDecision(
            node_type=NodeType.EXECUTE,
            rationale=result.get("rationale", "single deliverable solvable by one executor pass") if result else "fallback to execute",
        )
