"""Planner — LLM-driven task decomposition with dependency-safe ordering."""

from __future__ import annotations

import json
import random

from src.core.graph import TaskGraph
from src.core.inference import llm_judge
from src.core.models import NodeType, Task, TaskType
from src.core.signatures import PlannerOutput
from src.prompts.seed_prompts import PLANNER_PROMPT


class DefaultPlanner:
    def __init__(self, *, prompt: str = PLANNER_PROMPT) -> None:
        self.prompt = prompt

    def plan(self, task: Task) -> PlannerOutput:
        result = llm_judge(
            system=self.prompt,
            instructions=task.goal,
            context=task.context_input or "",
            output_schema={
                "reasoning": "Step-by-step analysis of how to decompose this task",
                "subtasks": "JSON array of subtask objects, each with: id (string), goal, task_type (THINK/RETRIEVE/WRITE/CODE), dependencies (list of string ids)",
                "rationale": "Brief explanation of the decomposition strategy",
            },
        )

        subtasks: list[Task] = []
        if result and result.get("subtasks"):
            try:
                raw = result["subtasks"]
                if isinstance(raw, str):
                    raw = json.loads(raw)
                used_ids: set[str] = set()
                for i, st in enumerate(raw):
                    st_id = str(st.get("id", f"{task.id}.sub{i}"))
                    if st_id == task.id:
                        st_id = f"{task.id}.sub{i}"
                    while st_id in used_ids:
                        st_id = f"{task.id}.sub{i}_{random.randint(0, 999)}"
                    used_ids.add(st_id)

                    st_type_str = str(st.get("task_type", "THINK")).upper()
                    try:
                        st_type = TaskType(st_type_str)
                    except ValueError:
                        st_type = TaskType.THINK

                    raw_deps = st.get("dependencies", [])
                    st_deps = [str(d) for d in raw_deps if str(d) != st_id]

                    subtasks.append(Task(
                        id=st_id,
                        goal=str(st.get("goal", task.goal)),
                        task_type=st_type,
                        dependencies=st_deps,
                        parent_id=task.id,
                        context_input=task.context_input or "",
                        metadata={"force_node_type": NodeType.EXECUTE.value} if not st_deps else {},
                    ))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        if not subtasks:
            return PlannerOutput(
                subtasks=[Task(
                    id=f"{task.id}.solve",
                    goal=task.goal,
                    task_type=task.task_type,
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )],
                task_graph=TaskGraph.from_tasks([Task(
                    id=f"{task.id}.solve",
                    goal=task.goal,
                    task_type=task.task_type,
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )]),
                rationale="fallback: executing as single task",
            )

        # Filter deps to only known IDs
        all_ids = {st.id for st in subtasks}
        for st in subtasks:
            st.dependencies = [d for d in st.dependencies if d in all_ids]

        # Topological sort: independent tasks first
        ordered: list[Task] = []
        by_id = {st.id: st for st in subtasks}
        remaining = set(by_id.keys())
        while remaining:
            ready = [sid for sid in remaining if all(d not in remaining for d in by_id[sid].dependencies)]
            if not ready:
                ready = list(remaining)
            for sid in sorted(ready):
                ordered.append(by_id[sid])
                remaining.discard(sid)

        task_graph = TaskGraph()
        for st in ordered:
            task_graph.add_task(st)

        rationale = result.get("rationale", "llm-generated decomposition") if result else "llm-generated decomposition"
        return PlannerOutput(subtasks=ordered, task_graph=task_graph, rationale=rationale)
