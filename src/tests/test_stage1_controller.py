from __future__ import annotations

import sys
import threading
import time

import pytest

from src.core.controller import RecursionGuardError, RomaController
from src.core.graph import TaskGraph
from src.core.models import NodeType, Task, TaskType
from src.core.registry import CodeSandbox, ComponentRegistry, RuntimeLimits, ToolError
from src.core.signatures import (
    Aggregator,
    AggregatorOutput,
    Atomizer,
    AtomizerDecision,
    Executor,
    ExecutorOutput,
    Planner,
    PlannerOutput,
)


class RoutingAtomizer(Atomizer):
    def decide(self, task: Task) -> AtomizerDecision:
        if task.metadata.get("mode") == "plan":
            return AtomizerDecision(node_type=NodeType.PLAN, rationale="requires decomposition")
        return AtomizerDecision(node_type=NodeType.EXECUTE, rationale="atomic leaf")


class MappingPlanner(Planner):
    def __init__(self, mapping: dict[str, list[Task]]) -> None:
        self.mapping = mapping

    def plan(self, task: Task) -> PlannerOutput:
        subtasks = self.mapping[task.id]
        task_graph = TaskGraph()
        for subtask in subtasks:
            task_graph.add_task(subtask)
        return PlannerOutput(
            subtasks=subtasks,
            task_graph=task_graph,
            rationale=f"planned {len(subtasks)} subtasks",
        )


class RecursivePlanner(Planner):
    def plan(self, task: Task) -> PlannerOutput:
        child = Task(
            id=f"{task.id}-child",
            goal=task.goal,
            task_type=task.task_type,
            parent_id=task.id,
            metadata={"mode": "plan"},
        )
        task_graph = TaskGraph()
        task_graph.add_task(child)
        return PlannerOutput(
            subtasks=[child],
            task_graph=task_graph,
            rationale="expand again",
        )


class RecordingExecutor(Executor):
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds
        self.starts: dict[str, float] = {}
        self.bound_tools: list[str] = []
        self._lock = threading.Lock()

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(TaskType)

    def set_tools(self, tools) -> None:  # noqa: ANN001
        self.bound_tools = tools.list_names()
        self._tools = tools

    def execute(self, task: Task) -> ExecutorOutput:
        with self._lock:
            self.starts[task.id] = time.perf_counter()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return ExecutorOutput(
            task_id=task.id,
            result=f"executed:{task.id}",
            metadata={"thread": threading.current_thread().name},
        )


class JoiningAggregator(Aggregator):
    def aggregate(self, task: Task, child_outputs: list[ExecutorOutput]) -> AggregatorOutput:
        summary = " | ".join(output.result for output in child_outputs)
        return AggregatorOutput(task_id=task.id, summary=summary)


def build_registry(atomizer: Atomizer, planner: Planner, executor: Executor) -> ComponentRegistry:
    registry = ComponentRegistry(
        atomizer=atomizer,
        planner=planner,
        aggregator=JoiningAggregator(),
        limits=RuntimeLimits(
            max_recursion_depth=3,
            max_subtasks_per_plan=8,
            max_total_tasks=32,
            max_expansions_per_goal=2,
            max_parallelism=4,
        ),
    )
    sandbox = CodeSandbox(python_executable=sys.executable)
    registry.register_tool(sandbox)
    registry.register_executor(executor, allowed_tools={"code_sandbox"})
    return registry


def test_controller_executes_atomic_task_without_planning() -> None:
    executor = RecordingExecutor()
    registry = build_registry(RoutingAtomizer(), MappingPlanner(mapping={}), executor)
    controller = RomaController(registry)

    outcome = controller.solve(Task(id="root", goal="answer", task_type=TaskType.THINK))

    assert outcome.output.result == "executed:root"
    assert outcome.trace.node_type == NodeType.EXECUTE
    assert any(event.kind == "executor_completed" for event in outcome.trace.events)


def test_controller_executes_dependency_batches_with_parallelism() -> None:
    executor = RecordingExecutor(delay_seconds=0.15)
    registry = build_registry(
        RoutingAtomizer(),
        MappingPlanner(
            mapping={
                "root": [
                    Task(
                        id="a",
                        goal="A",
                        task_type=TaskType.THINK,
                        parent_id="root",
                    ),
                    Task(
                        id="b",
                        goal="B",
                        task_type=TaskType.THINK,
                        parent_id="root",
                    ),
                    Task(
                        id="c",
                        goal="C",
                        task_type=TaskType.THINK,
                        parent_id="root",
                        dependencies=["a", "b"],
                    ),
                ]
            }
        ),
        executor,
    )
    controller = RomaController(registry)
    root = Task(id="root", goal="root", task_type=TaskType.THINK, metadata={"mode": "plan"})

    started = time.perf_counter()
    outcome = controller.solve(root)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.42
    assert outcome.output.result == "executed:a | executed:b | executed:c"
    batch_events = [event for event in outcome.trace.events if event.kind == "batch_started"]
    assert batch_events[0].payload["task_ids"] == ["a", "b"]
    assert batch_events[1].payload["task_ids"] == ["c"]
    assert abs(executor.starts["a"] - executor.starts["b"]) < 0.08
    assert executor.starts["c"] > executor.starts["a"]


def test_controller_recursion_guard_stops_runaway_planning() -> None:
    executor = RecordingExecutor()
    registry = build_registry(RoutingAtomizer(), RecursivePlanner(), executor)
    controller = RomaController(registry)
    task = Task(id="root", goal="loop", task_type=TaskType.THINK, metadata={"mode": "plan"})

    with pytest.raises(RecursionGuardError):
        controller.solve(task)


def test_registry_binds_only_allowed_tools_to_executor() -> None:
    executor = RecordingExecutor()
    registry = build_registry(RoutingAtomizer(), MappingPlanner(mapping={}), executor)

    assert executor.bound_tools == ["code_sandbox"]
    with pytest.raises(ToolError):
        executor._tools.get("web_search")


def test_code_sandbox_executes_python_snippet() -> None:
    sandbox = CodeSandbox(python_executable=sys.executable)

    result = sandbox.invoke(language="python", code="print('ok')")

    assert result.ok is True
    assert result.output == "ok"
