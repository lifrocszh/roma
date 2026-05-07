"""Dependency-aware task graph primitives for Stage 0."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.core.models import CONTRACT_VERSION, Task


class TaskGraphError(ValueError):
    """Base exception for task graph validation failures."""


class DuplicateTaskIdError(TaskGraphError):
    """Raised when a graph receives the same task ID twice."""


class UnknownDependencyError(TaskGraphError):
    """Raised when a task depends on a task that does not exist."""


class TaskCycleError(TaskGraphError):
    """Raised when the graph contains a cycle."""


class TaskGraph(BaseModel):
    """A DAG of tasks with validation and dependency-aware traversal helpers."""

    model_config = ConfigDict(validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    tasks: dict[str, Task] = Field(default_factory=dict)

    @classmethod
    def from_tasks(cls, tasks: Iterable[Task], *, external_ids: set[str] | None = None) -> TaskGraph:
        graph = cls()
        for task in tasks:
            graph.add_task(task)
        graph.validate(external_ids=external_ids)
        return graph

    def add_task(self, task: Task) -> None:
        if task.id in self.tasks:
            raise DuplicateTaskIdError(f"duplicate task id: {task.id}")
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task id: {task_id}") from exc

    def validate(self, *, external_ids: set[str] | None = None) -> None:
        external_ids = external_ids or set()
        for task in self.tasks.values():
            for dependency_id in task.dependencies:
                if dependency_id not in self.tasks:
                    raise UnknownDependencyError(
                        f"task {task.id} depends on unknown task {dependency_id}"
                    )

            if task.parent_id and task.parent_id not in self.tasks and task.parent_id not in external_ids:
                raise UnknownDependencyError(
                    f"task {task.id} references unknown parent {task.parent_id}"
                )

            for child_id in task.child_ids:
                if child_id not in self.tasks and child_id not in external_ids:
                    raise UnknownDependencyError(
                        f"task {task.id} references unknown child {child_id}"
                    )

        if self._has_cycle():
            raise TaskCycleError("task graph contains a dependency cycle")

    def topological_order(self, *, external_ids: set[str] | None = None) -> list[Task]:
        self.validate(external_ids=external_ids)
        indegree = {task_id: 0 for task_id in self.tasks}
        dependents: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}

        for task in self.tasks.values():
            for dependency_id in task.dependencies:
                indegree[task.id] += 1
                dependents[dependency_id].append(task.id)

        queue = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        ordered_ids: list[str] = []

        while queue:
            current = queue.popleft()
            ordered_ids.append(current)
            for dependent_id in sorted(dependents[current]):
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(ordered_ids) != len(self.tasks):
            raise TaskCycleError("task graph contains a dependency cycle")

        return [self.tasks[task_id] for task_id in ordered_ids]

    def dependency_batches(self, *, external_ids: set[str] | None = None) -> list[list[Task]]:
        self.validate(external_ids=external_ids)
        indegree = {task_id: 0 for task_id in self.tasks}
        dependents: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}

        for task in self.tasks.values():
            for dependency_id in task.dependencies:
                indegree[task.id] += 1
                dependents[dependency_id].append(task.id)

        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        batches: list[list[Task]] = []
        processed = 0

        while ready:
            current_batch = ready
            batches.append([self.tasks[task_id] for task_id in current_batch])
            processed += len(current_batch)

            next_ready: list[str] = []
            for task_id in current_batch:
                for dependent_id in sorted(dependents[task_id]):
                    indegree[dependent_id] -= 1
                    if indegree[dependent_id] == 0:
                        next_ready.append(dependent_id)
            ready = sorted(next_ready)

        if processed != len(self.tasks):
            raise TaskCycleError("task graph contains a dependency cycle")

        return batches

    def ready_tasks(self, completed_task_ids: set[str], *, external_ids: set[str] | None = None) -> list[Task]:
        self.validate(external_ids=external_ids)
        ready = [
            task
            for task in self.tasks.values()
            if task.id not in completed_task_ids
            and all(dependency_id in completed_task_ids for dependency_id in task.dependencies)
        ]
        return sorted(ready, key=lambda task: task.id)

    def _has_cycle(self) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(task_id: str) -> bool:
            if task_id in visiting:
                return True
            if task_id in visited:
                return False

            visiting.add(task_id)
            for dependency_id in self.tasks[task_id].dependencies:
                if dfs(dependency_id):
                    return True
            visiting.remove(task_id)
            visited.add(task_id)
            return False

        return any(dfs(task_id) for task_id in self.tasks)
