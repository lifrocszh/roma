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


def test_atomizer_returns_valid_decision() -> None:
    atomizer = DefaultAtomizer()

    decision = atomizer.decide(
        Task(
            id="t2",
            goal="Write a multi-part fantasy story with character arcs and world-building",
            task_type=TaskType.WRITE,
        )
    )

    assert decision.node_type in (NodeType.PLAN, NodeType.EXECUTE)
    assert len(decision.rationale) > 0


def test_atomizer_respects_force_node_type() -> None:
    atomizer = DefaultAtomizer()

    decision = atomizer.decide(
        Task(
            id="t3",
            goal="Any task",
            task_type=TaskType.GENERAL,
            metadata={"force_node_type": NodeType.EXECUTE.value},
        )
    )

    assert decision.node_type == NodeType.EXECUTE
    assert "planner marked" in decision.rationale


def test_planner_falls_back_when_llm_unavailable() -> None:
    planner = DefaultPlanner()
    task = Task(id="single", goal="Simple task", task_type=TaskType.GENERAL)

    plan = planner.plan(task)

    # Expect either LLM-generated subtasks or fallback single task
    assert len(plan.subtasks) >= 1
    plan.task_graph.validate(external_ids={"single"})


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
    assert isinstance(write_output, ExecutorOutput)
    assert isinstance(retrieve_output, ExecutorOutput)
    assert isinstance(code_output, ExecutorOutput)

    assert len(think_output.result) > 0
    assert len(write_output.result) > 0
    assert len(retrieve_output.result) > 0
    assert "hello" in code_output.result or "Sandbox" in code_output.result


def test_aggregator_passthrough_single_output() -> None:
    aggregator = DefaultAggregator()
    task = Task(id="agg", goal="merge", task_type=TaskType.THINK)

    output = aggregator.aggregate(
        task,
        [ExecutorOutput(task_id="a", result="single result")],
    )

    assert output.summary == "single result"


def test_aggregator_handles_empty_outputs() -> None:
    aggregator = DefaultAggregator()
    task = Task(id="agg", goal="merge", task_type=TaskType.THINK)

    output = aggregator.aggregate(task, [])

    assert output.summary == "No outputs to aggregate."
    assert output.metadata["child_count"] == 0


def test_aggregator_handles_multiple_outputs() -> None:
    aggregator = DefaultAggregator()
    task = Task(id="agg", goal="merge", task_type=TaskType.THINK)

    output = aggregator.aggregate(
        task,
        [
            ExecutorOutput(task_id="a", result="first result"),
            ExecutorOutput(task_id="b", result="second result"),
        ],
    )

    assert len(output.summary) > 0
    assert output.metadata["child_count"] == 2


def test_stage2_components_enable_end_to_end_agent_workflow() -> None:
    registry = build_default_registry(python_executable=sys.executable)
    controller = RomaController(registry)
    task = Task(
        id="root",
        goal="What is the capital of France?",
        task_type=TaskType.GENERAL,
    )

    outcome = controller.solve(task)

    assert outcome.task.result is not None
    assert len(outcome.task.result) > 0
