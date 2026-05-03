from __future__ import annotations

import sys

from src.components import (
    DefaultAggregator,
    DefaultAtomizer,
    DefaultPlanner,
    build_default_registry,
)
from src.components.executors import CodeExecutor, RetrieveExecutor, ThinkExecutor, WriteExecutor
from src.core.controller import RomaController
from src.core.models import NodeType, Task, TaskType
from src.core.registry import CodeSandbox
from src.core.signatures import ExecutorOutput


def test_atomizer_bypasses_planning_for_simple_atomic_task() -> None:
    atomizer = DefaultAtomizer()

    decision = atomizer.decide(Task(id="t1", goal="Summarize this paragraph", task_type=TaskType.WRITE))

    assert decision.node_type == NodeType.EXECUTE
    assert "single deliverable" in decision.rationale


def test_atomizer_marks_complex_writing_task_as_plan() -> None:
    atomizer = DefaultAtomizer()

    decision = atomizer.decide(
        Task(
            id="t2",
            goal="Write a multi-part fantasy story with character arcs and world-building",
            task_type=TaskType.WRITE,
        )
    )

    assert decision.node_type == NodeType.PLAN


def test_planner_returns_valid_writing_subtask_structure() -> None:
    planner = DefaultPlanner()
    task = Task(id="story", goal="Write a fantasy chapter", task_type=TaskType.WRITE)

    plan = planner.plan(task)

    assert [subtask.id for subtask in plan.subtasks] == [
        "story.foundation",
        "story.development",
        "story.synthesis",
    ]
    plan.task_graph.validate(external_ids={"story"})
    assert plan.subtasks[1].dependencies == ["story.foundation"]
    assert plan.subtasks[2].dependencies == ["story.development"]


def test_specialized_executors_return_schema_conformant_outputs() -> None:
    think = ThinkExecutor()
    write = WriteExecutor()
    retrieve = RetrieveExecutor()
    code = CodeExecutor()
    code.set_tools(type("ToolView", (), {"get": lambda self, name: CodeSandbox(python_executable=sys.executable), "list_names": lambda self: ["code_sandbox"]})())

    think_output = think.execute(Task(id="a", goal="Reason about the tradeoff", task_type=TaskType.THINK))
    write_output = write.execute(Task(id="b", goal="Draft a release note", task_type=TaskType.WRITE))
    retrieve_output = retrieve.execute(Task(id="c", goal="Find background facts", task_type=TaskType.RETRIEVE, context_input="local facts"))
    code_output = code.execute(
        Task(
            id="d",
            goal="Run a snippet",
            task_type=TaskType.CODE,
            metadata={"code": "print('hello')", "language": "python"},
        )
    )

    assert isinstance(think_output, ExecutorOutput)
    assert "Analysis for" in think_output.result
    assert "Written response" in write_output.result
    assert "Fallback evidence" in retrieve_output.result
    assert "Sandbox execution result" in code_output.result


def test_aggregator_compresses_duplicate_child_outputs() -> None:
    aggregator = DefaultAggregator()
    task = Task(id="agg", goal="merge", task_type=TaskType.THINK)

    output = aggregator.aggregate(
        task,
        [
            ExecutorOutput(task_id="a", result="same"),
            ExecutorOutput(task_id="b", result="same"),
            ExecutorOutput(task_id="c", result="different"),
        ],
    )

    assert output.summary == "same\n\ndifferent"
    assert output.metadata["dropped_duplicates"] == 1


def test_stage2_components_enable_end_to_end_agent_workflow() -> None:
    registry = build_default_registry(python_executable=sys.executable)
    controller = RomaController(registry)
    task = Task(
        id="root",
        goal="Write a multi-part project update then refine the final message",
        task_type=TaskType.WRITE,
        context_input="Status: tests pass. Risk: deployment timing.",
    )

    outcome = controller.solve(task)

    assert outcome.task.result is not None
    assert "Narrative draft" in outcome.task.result or "Written response" in outcome.task.result
    assert outcome.trace.node_type == NodeType.PLAN
    assert len(outcome.trace.child_traces) == 3
