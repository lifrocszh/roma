from __future__ import annotations

import json
import logging
import random

from src.core.graph import TaskGraph
from src.core.inference import build_client, get_default_model
from src.core.models import NodeType, Task
from src.core.signatures import PlannerOutput
from src.prompts.seed_prompts import PLANNER_PROMPT


_log = logging.getLogger("roma.planner")


class DefaultPlanner:
    def __init__(self, *, prompt: str = PLANNER_PROMPT, model: str | None = None) -> None:
        self.prompt = prompt
        self._model = model or get_default_model()

    def plan(self, task: Task) -> PlannerOutput:
        client = build_client()
        if client is None:
            return PlannerOutput(
                subtasks=[Task(
                    id=f"{task.id}.solve",
                    goal=task.goal,
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )],
                task_graph=TaskGraph.from_tasks([Task(
                    id=f"{task.id}.solve",
                    goal=task.goal,
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )], external_ids={task.id}),
                rationale="no LLM client available, fallback to single task",
            )

        context = task.context_input or ""
        schema_lines = "\n".join(
            f'  "{k}": "{v}"'
            for k, v in {
                "reasoning": "Step-by-step analysis of how to decompose this task",
                "subtasks": "JSON array of subtask objects, each with: id (string), goal, dependencies (list of string ids)",
                "rationale": "Brief explanation of the decomposition strategy",
            }.items()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}\n\n"
                    "You MUST respond with valid JSON only, containing exactly these fields:\n"
                    f"{schema_lines}\n\n"
                    "Think step by step before producing your final answer."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task.goal}\n\nContext:\n{context}" if context.strip() else f"Task: {task.goal}",
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content) if (content and content.strip()) else None
        except Exception:
            _log.warning("LLM call failed: model=%s", self._model, exc_info=True)
            result = None

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

                    raw_deps = st.get("dependencies", [])
                    st_deps = [str(d) for d in raw_deps if str(d) != st_id]

                    subtasks.append(Task(
                        id=st_id,
                        goal=str(st.get("goal", task.goal)),
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
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )],
                task_graph=TaskGraph.from_tasks([Task(
                    id=f"{task.id}.solve",
                    goal=task.goal,
                    parent_id=task.id,
                    context_input=task.context_input,
                    metadata={"force_node_type": NodeType.EXECUTE.value},
                )], external_ids={task.id}),
                rationale="fallback: executing as single task",
            )

        all_ids = {st.id for st in subtasks}
        for st in subtasks:
            st.dependencies = [d for d in st.dependencies if d in all_ids]

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
        task_graph.validate(external_ids={task.id})

        rationale = result.get("rationale", "llm-generated decomposition") if result else "llm-generated decomposition"
        return PlannerOutput(subtasks=ordered, task_graph=task_graph, rationale=rationale)
