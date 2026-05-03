"""Core data models for Stage 0 of the ROMA reimplementation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONTRACT_VERSION = "0.1"


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for serialized records."""
    return datetime.now(UTC)


class TaskType(str, Enum):
    """High-level task categories used for routing later stages."""

    GENERAL = "GENERAL"
    RETRIEVE = "RETRIEVE"
    THINK = "THINK"
    WRITE = "WRITE"
    CODE = "CODE"


class NodeType(str, Enum):
    """Recursive routing decision for a task node."""

    PLAN = "PLAN"
    EXECUTE = "EXECUTE"


class ArtifactHandle(BaseModel):
    """A stable reference to an artifact stored outside prompts."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    artifact_type: str = Field(default="text", min_length=1)


class Task(BaseModel):
    """Canonical task record shared across the controller and components."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    task_type: TaskType = TaskType.GENERAL
    dependencies: list[str] = Field(default_factory=list)
    context_input: str | None = None
    result: str | None = None
    artifacts: list[ArtifactHandle] = Field(default_factory=list)
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dependencies", "child_ids")
    @classmethod
    def ensure_unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate identifiers are not allowed")
        return values

    @model_validator(mode="after")
    def validate_lineage(self) -> Task:
        if self.id in self.dependencies:
            raise ValueError("task cannot depend on itself")
        if self.id in self.child_ids:
            raise ValueError("task cannot list itself as a child")
        if self.parent_id and self.parent_id == self.id:
            raise ValueError("task cannot be its own parent")
        return self


class TraceEvent(BaseModel):
    """An immutable decision or output captured during execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime = Field(default_factory=utc_now)
    kind: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Hierarchical execution trace that mirrors the recursive solve tree."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    task_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    node_type: NodeType | None = None
    task_type: TaskType | None = None
    parent_task_id: str | None = None
    input_context: str | None = None
    output_summary: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    child_traces: list[ExecutionTrace] = Field(default_factory=list)

    def append_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Append a trace event without mutating prior records."""
        self.events.append(TraceEvent(kind=kind, payload=payload or {}))

    def append_child(self, child: ExecutionTrace) -> None:
        """Append a child trace while enforcing tree identity."""
        if child.parent_task_id != self.task_id:
            raise ValueError("child trace parent_task_id must match current task_id")
        self.child_traces.append(child)


class PlanSubtask(BaseModel):
    """A minimal planner-facing subtask record used before execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    task_type: TaskType = TaskType.GENERAL
    dependencies: list[str] = Field(default_factory=list)
    context_input: str | None = None
    artifacts: list[ArtifactHandle] = Field(default_factory=list)

    @field_validator("dependencies")
    @classmethod
    def ensure_unique_dependencies(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate dependency identifiers are not allowed")
        return values
