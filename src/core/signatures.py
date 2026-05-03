"""Stable role contracts for the ROMA Stage 0 foundation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph import TaskGraph
from src.core.models import CONTRACT_VERSION, NodeType, Task, TaskType


class AtomizerDecision(BaseModel):
    """Atomizer output contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    node_type: NodeType
    rationale: str = Field(min_length=1)


class PlannerOutput(BaseModel):
    """Planner output contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    subtasks: list[Task] = Field(default_factory=list)
    task_graph: TaskGraph
    rationale: str = Field(min_length=1)


class ExecutorOutput(BaseModel):
    """Executor output contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    task_id: str = Field(min_length=1)
    result: str = Field(min_length=1)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregatorOutput(BaseModel):
    """Aggregator output contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: str = Field(default=CONTRACT_VERSION)
    task_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Atomizer(ABC):
    """Determines whether a task should be planned or executed."""

    @abstractmethod
    def decide(self, task: Task) -> AtomizerDecision:
        raise NotImplementedError


class Planner(ABC):
    """Decomposes a task into a validated dependency-aware task graph."""

    @abstractmethod
    def plan(self, task: Task) -> PlannerOutput:
        raise NotImplementedError


class Executor(ABC):
    """Executes an atomic task of one or more supported types."""

    @property
    @abstractmethod
    def supported_task_types(self) -> frozenset[TaskType]:
        raise NotImplementedError

    @abstractmethod
    def execute(self, task: Task) -> ExecutorOutput:
        raise NotImplementedError


class Aggregator(ABC):
    """Merges child outputs into a parent-level result."""

    @abstractmethod
    def aggregate(self, task: Task, child_outputs: list[ExecutorOutput]) -> AggregatorOutput:
        raise NotImplementedError
