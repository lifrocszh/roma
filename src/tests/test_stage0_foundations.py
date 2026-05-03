from __future__ import annotations

import json

import pytest

from src.core.artifact_store import ArtifactStore
from src.core.graph import DuplicateTaskIdError, TaskCycleError, TaskGraph, UnknownDependencyError
from src.core.models import ArtifactHandle, ExecutionTrace, NodeType, Task, TaskType


def test_task_graph_validates_and_serializes() -> None:
    root = Task(id="root", goal="Solve the root task", task_type=TaskType.THINK, child_ids=["child"])
    child = Task(
        id="child",
        goal="Gather evidence",
        task_type=TaskType.RETRIEVE,
        dependencies=["root"],
        parent_id="root",
    )

    graph = TaskGraph.from_tasks([root, child])

    assert [task.id for task in graph.topological_order()] == ["root", "child"]
    assert [[task.id for task in batch] for batch in graph.dependency_batches()] == [["root"], ["child"]]
    assert json.loads(graph.model_dump_json())["tasks"]["root"]["goal"] == "Solve the root task"


def test_graph_rejects_unknown_dependencies() -> None:
    graph = TaskGraph()
    graph.add_task(Task(id="child", goal="missing dep", dependencies=["ghost"]))

    with pytest.raises(UnknownDependencyError):
        graph.validate()


def test_graph_rejects_duplicate_ids() -> None:
    graph = TaskGraph()
    graph.add_task(Task(id="task-1", goal="first"))

    with pytest.raises(DuplicateTaskIdError):
        graph.add_task(Task(id="task-1", goal="second"))


def test_graph_rejects_cycles() -> None:
    graph = TaskGraph.from_tasks(
        [
            Task(id="a", goal="A", dependencies=["b"]),
            Task(id="b", goal="B"),
        ]
    )

    graph.tasks["b"] = Task(id="b", goal="B", dependencies=["a"])

    with pytest.raises(TaskCycleError):
        graph.validate()


def test_artifact_store_preserves_identity_across_task_and_trace(tmp_path) -> None:
    store = ArtifactStore(base_path=tmp_path)
    handle = store.put("artifact://root/summary", {"text": "concise result"}, artifact_type="json")

    task = Task(
        id="root",
        goal="Return a summary",
        artifacts=[handle],
        result="artifact://root/summary",
    )
    trace = ExecutionTrace(task_id="root", goal=task.goal, node_type=NodeType.EXECUTE)
    trace.append_event("artifact_created", {"key": handle.key})

    persisted_store = ArtifactStore(base_path=tmp_path)
    persisted_record = persisted_store.get(handle.key)

    assert task.artifacts == [ArtifactHandle(key="artifact://root/summary", artifact_type="json")]
    assert trace.events[0].payload["key"] == handle.key
    assert persisted_record.handle.key == handle.key
    assert persisted_record.value == {"text": "concise result"}


def test_trace_requires_parent_child_identity_match() -> None:
    root = ExecutionTrace(task_id="root", goal="Root goal")
    child = ExecutionTrace(task_id="child", goal="Child goal", parent_task_id="other-root")

    with pytest.raises(ValueError):
        root.append_child(child)
