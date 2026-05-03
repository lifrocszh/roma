"""Stage 2 Planner implementation."""

from __future__ import annotations

from src.core.graph import TaskGraph
from src.core.models import NodeType, Task, TaskType
from src.core.signatures import Planner, PlannerOutput
from src.prompts.seed_prompts import PLANNER_PROMPT


class DefaultPlanner(Planner):
    """Heuristic planner that emits a compact MECE subtask DAG."""

    def __init__(self, *, prompt: str = PLANNER_PROMPT) -> None:
        self.prompt = prompt

    def plan(self, task: Task) -> PlannerOutput:
        if task.task_type == TaskType.WRITE:
            subtasks = self._plan_write(task)
            rationale = "structured writing into foundation, development, and synthesis"
        elif task.task_type == TaskType.RETRIEVE:
            subtasks = self._plan_retrieve(task)
            rationale = "split retrieval into query framing, evidence gathering, and consolidation"
        elif task.task_type == TaskType.CODE:
            subtasks = self._plan_code(task)
            rationale = "split code task into understanding, implementation, and verification"
        else:
            subtasks = self._plan_general(task)
            rationale = "split task into evidence, reasoning, and final synthesis"

        task_graph = TaskGraph()
        for subtask in subtasks:
            task_graph.add_task(subtask)
        return PlannerOutput(subtasks=subtasks, task_graph=task_graph, rationale=rationale)

    def _plan_write(self, task: Task) -> list[Task]:
        return [
            Task(
                id=f"{task.id}.foundation",
                goal=f"Establish the core outline, constraints, voice, and key themes for: {task.goal}",
                task_type=TaskType.THINK,
                parent_id=task.id,
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.development",
                goal=f"Draft the main body or scenes for: {task.goal}",
                task_type=TaskType.WRITE,
                parent_id=task.id,
                dependencies=[f"{task.id}.foundation"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.synthesis",
                goal=f"Refine and finalize the response for: {task.goal}",
                task_type=TaskType.WRITE,
                parent_id=task.id,
                dependencies=[f"{task.id}.development"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
        ]

    def _plan_retrieve(self, task: Task) -> list[Task]:
        return [
            Task(
                id=f"{task.id}.queries",
                goal=f"Identify the search angles and evidence targets for: {task.goal}",
                task_type=TaskType.THINK,
                parent_id=task.id,
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.evidence",
                goal=f"Gather relevant evidence for: {task.goal}",
                task_type=TaskType.RETRIEVE,
                parent_id=task.id,
                dependencies=[f"{task.id}.queries"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.summary",
                goal=f"Summarize the evidence for: {task.goal}",
                task_type=TaskType.WRITE,
                parent_id=task.id,
                dependencies=[f"{task.id}.evidence"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
        ]

    def _plan_code(self, task: Task) -> list[Task]:
        return [
            Task(
                id=f"{task.id}.analysis",
                goal=f"Analyze requirements and edge cases for: {task.goal}",
                task_type=TaskType.THINK,
                parent_id=task.id,
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.implementation",
                goal=f"Produce or run the code needed for: {task.goal}",
                task_type=TaskType.CODE,
                parent_id=task.id,
                dependencies=[f"{task.id}.analysis"],
                context_input=task.context_input,
                metadata={**task.metadata, **self._leaf_metadata()},
            ),
            Task(
                id=f"{task.id}.verification",
                goal=f"Verify the result of: {task.goal}",
                task_type=TaskType.THINK,
                parent_id=task.id,
                dependencies=[f"{task.id}.implementation"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
        ]

    def _plan_general(self, task: Task) -> list[Task]:
        return [
            Task(
                id=f"{task.id}.inputs",
                goal=f"Collect and organize the essential inputs for: {task.goal}",
                task_type=TaskType.THINK if task.task_type == TaskType.GENERAL else task.task_type,
                parent_id=task.id,
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.analysis",
                goal=f"Reason through the main problem in: {task.goal}",
                task_type=TaskType.THINK,
                parent_id=task.id,
                dependencies=[f"{task.id}.inputs"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
            Task(
                id=f"{task.id}.answer",
                goal=f"Compose the final answer for: {task.goal}",
                task_type=TaskType.WRITE,
                parent_id=task.id,
                dependencies=[f"{task.id}.analysis"],
                context_input=task.context_input,
                metadata=self._leaf_metadata(),
            ),
        ]

    def _leaf_metadata(self) -> dict[str, str]:
        return {"force_node_type": NodeType.EXECUTE.value}
