from src.components.aggregator import DefaultAggregator
from src.components.atomizer import DefaultAtomizer
from src.components.executors import UnifiedExecutor
from src.components.planner import DefaultPlanner
from src.core.registry import Calculator, CodeSandbox, ComponentRegistry, RuntimeLimits, WebSearchToolkit


def build_default_registry(
    *,
    tavily_api_key: str | None = None,
    python_executable: str = "python",
    limits: RuntimeLimits | None = None,
    verbose: bool = False,
) -> ComponentRegistry:
    registry = ComponentRegistry(
        executor=UnifiedExecutor(),
        atomizer=DefaultAtomizer(),
        planner=DefaultPlanner(),
        aggregator=DefaultAggregator(),
        limits=limits,
        verbose=verbose,
    )
    registry.register_tool(Calculator())
    registry.register_tool(WebSearchToolkit(api_key=tavily_api_key))
    registry.register_tool(CodeSandbox(python_executable=python_executable))
    return registry


__all__ = ["build_default_registry", "UnifiedExecutor"]
