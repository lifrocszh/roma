"""Stage 4 configuration schema for ROMA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.registry import RuntimeLimits


class RoleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    adapter_type: str | None = None


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    atomizer: str | None = None
    planner: str | None = None
    aggregator: str | None = None
    executors: dict[str, str] = Field(default_factory=dict)


class ToolkitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class RoleRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: RoleModelConfig | None = None
    prompt: str | None = None
    demos: list[str] = Field(default_factory=list)
    toolkits: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    atomizer: RoleRuntimeConfig = Field(default_factory=RoleRuntimeConfig)
    planner: RoleRuntimeConfig = Field(default_factory=RoleRuntimeConfig)
    aggregator: RoleRuntimeConfig = Field(default_factory=RoleRuntimeConfig)
    executors: dict[str, RoleRuntimeConfig] = Field(default_factory=dict)
    max_subtasks: int = 12
    toolkits: dict[str, ToolkitConfig] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)


class DemoTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    goal: str
    task_type: str = "WRITE"
    context_input: str | None = None


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    module_name: str = "write"
    base_prompt: str | None = None
    held_out_signals: list[str] = Field(default_factory=list)
    rounds: int = 1
    proposals_per_round: int = 4
    topn: int = 2
    seed: int = 0
    max_tokens: int | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    api_keys_path: Path = Field(default=Path("config/api_keys.toml"))
    default_search_provider: str = "tavily"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent: AgentConfig = Field(default_factory=AgentConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    demo: DemoTaskConfig = Field(default_factory=lambda: DemoTaskConfig(goal="Write a multi-part project update then refine the final message"))
    optimize: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @field_validator("optimize")
    @classmethod
    def ensure_base_prompt_for_optimization(cls, value: OptimizationConfig) -> OptimizationConfig:
        return value
