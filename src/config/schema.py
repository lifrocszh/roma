"""Configuration schema."""

from pathlib import Path

from pydantic import BaseModel, Field

from src.core.registry import RuntimeLimits


class DemoTaskConfig(BaseModel):
    goal: str = "Write a multi-part project update then refine the final message"
    task_type: str = "WRITE"
    context_input: str | None = None


class RuntimeConfig(BaseModel):
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    api_keys_path: Path = Field(default=Path("config/api_keys.toml"))


class AppConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    demo: DemoTaskConfig = Field(default_factory=lambda: DemoTaskConfig(goal="Write a multi-part project update then refine the final message"))
