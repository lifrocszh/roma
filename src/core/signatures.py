from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph import TaskGraph
from src.core.models import CONTRACT_VERSION, NodeType, Task


class AtomizerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = Field(default=CONTRACT_VERSION)
    node_type: NodeType
    rationale: str = Field(min_length=1)
    granted_tools: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = Field(default=CONTRACT_VERSION)
    subtasks: list[Task] = Field(default_factory=list)
    task_graph: TaskGraph
    rationale: str = Field(min_length=1)


class ExecutorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = Field(default=CONTRACT_VERSION)
    task_id: str = Field(min_length=1)
    result: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: str = Field(default=CONTRACT_VERSION)
    task_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
