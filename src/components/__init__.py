from src.components.aggregator import DefaultAggregator
from src.components.atomizer import DefaultAtomizer
from src.components.executors import CodeExecutor, RetrieveExecutor, ThinkExecutor, WriteExecutor
from src.components.planner import DefaultPlanner
from src.core.models import TaskType
from src.core.registry import CodeSandbox, ComponentRegistry, RuntimeLimits, WebSearchToolkit


def build_default_registry(
    *,
    tavily_api_key: str | None = None,
    python_executable: str = "python",
    limits: RuntimeLimits | None = None,
) -> ComponentRegistry:
    """Construct a working Stage 2 registry with default components."""

    registry = ComponentRegistry(
        atomizer=DefaultAtomizer(),
        planner=DefaultPlanner(),
        aggregator=DefaultAggregator(),
        limits=limits,
    )
    registry.register_tool(WebSearchToolkit(api_key=tavily_api_key))
    registry.register_tool(CodeSandbox(python_executable=python_executable))
    registry.register_executor(ThinkExecutor(), task_types=frozenset({TaskType.GENERAL, TaskType.THINK}))
    registry.register_executor(RetrieveExecutor(), allowed_tools={"web_search"})
    registry.register_executor(WriteExecutor())
    registry.register_executor(CodeExecutor(), allowed_tools={"code_sandbox"})
    return registry


__all__ = [
    "CodeExecutor",
    "DefaultAggregator",
    "DefaultAtomizer",
    "DefaultPlanner",
    "RetrieveExecutor",
    "ThinkExecutor",
    "WriteExecutor",
    "build_default_registry",
]
