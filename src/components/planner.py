"""Stage 2 Planner implementation - LLM-driven."""

from __future__ import annotations

import json

from src.core.graph import TaskGraph
from src.core.inference import llm_judge
from src.core.models import NodeType, Task, TaskType
from src.core.signatures import Planner, PlannerOutput
from src.prompts.seed_prompts import PLANNER_PROMPT


class DefaultPlanner(Planner):
    """LLM-driven planner that generates a task decomposition."""

    def __init__(self, *, prompt: str = PLANNER_PROMPT) -> None:
        self.prompt = prompt

    def plan(self, task: Task) -> PlannerOutput:
        context = task.context_input or ""

        result = llm_judge(
            system=self.prompt,
            instructions=task.goal,
            context=context,
            output_schema={
                "reasoning": "Step-by-step analysis of how to decompose this task",
                "subtasks": "A JSON array of subtask objects, each with: id, goal, task_type (THINK/ RETRIEVE/ WRITE/ CODE), dependencies (list of subtask ids this depends on)",
                "rationale": "Brief explanation of the decomposition strategy",
            },
        )

        subtasks: list[Task] = []
        if result and result.get("subtasks"):
            try:
                raw = result["subtasks"]
                if isinstance(raw, str):
                    raw = json.loads(raw)
                for i, st in enumerate(raw):
                    st_id = str(st.get("id", f"{task.id}.sub{i}"))
                    st_goal = st.get("goal", task.goal)
                    st_type_str = str(st.get("task_type", "THINK")).upper()
                    try:
                        st_type = TaskType(st_type_str)
                    except ValueError:
                        st_type = TaskType.THINK
                    raw_deps = st.get("dependencies", [])
                    st_deps = [str(d) for d in raw_deps]
                    subtasks.append(Task(
                        id=st_id,
                        goal=st_goal,
                        task_type=st_type,
                        dependencies=st_deps,
                        parent_id=task.id,
                        context_input=context,
                        metadata={"force_node_type": NodeType.EXECUTE.value} if not st_deps else {},
                    ))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        if not subtasks:
            subtasks, rationale = self._fallback_plan(task)
        else:
            rationale = (result.get("rationale") if result else None) or "llm-generated decomposition"

        task_graph = TaskGraph()
        for st in subtasks:
            task_graph.add_task(st)
        return PlannerOutput(subtasks=subtasks, task_graph=task_graph, rationale=rationale)

    def _fallback_plan(self, task: Task) -> tuple[list[Task], str]:
        """Fallback single-task plan when LLM output is unparseable."""
        return [
            Task(
                id=f"{task.id}.solve",
                goal=task.goal,
                task_type=task.task_type,
                parent_id=task.id,
                context_input=task.context_input,
                metadata={"force_node_type": NodeType.EXECUTE.value},
            ),
        ], "fallback: llm output unparseable, executing as single task"
