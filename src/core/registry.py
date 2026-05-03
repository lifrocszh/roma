"""Stage 1 runtime registry and tool abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models import CONTRACT_VERSION, TaskType
from src.core.signatures import Aggregator, Atomizer, Executor, Planner


class RegistryError(ValueError):
    """Raised when the controller registry is incomplete or inconsistent."""


class ToolError(RuntimeError):
    """Raised when a tool invocation fails."""


class ToolResult(BaseModel):
    """Structured result for a single tool call."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = Field(default=CONTRACT_VERSION)
    tool_name: str
    ok: bool
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class BaseTool(ABC):
    """Common interface for ROMA tools."""

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        self.name = name
        self._logger = logger or logging.getLogger(f"roma.tool.{name}")

    @abstractmethod
    def invoke(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def _log_result(self, result: ToolResult) -> ToolResult:
        self._logger.info(
            "tool=%s ok=%s duration_ms=%.2f metadata=%s",
            result.tool_name,
            result.ok,
            result.duration_ms,
            result.metadata,
        )
        return result


class WebSearchToolkit(BaseTool):
    """Tavily-backed web search adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_results: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="web_search", logger=logger)
        self.api_key = api_key
        self.max_results = max_results

    def invoke(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not query:
            raise ToolError("web search requires a non-empty query")

        started = time.perf_counter()
        if not self.api_key:
            raise ToolError("missing Tavily API key")

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ToolError("tavily-python is not installed") from exc

        client = TavilyClient(api_key=self.api_key)
        response = client.search(query=query, max_results=kwargs.get("max_results", self.max_results))
        duration_ms = (time.perf_counter() - started) * 1000
        result = ToolResult(
            tool_name=self.name,
            ok=True,
            output=str(response),
            metadata={"query": query, "result_count": len(response.get("results", []))},
            duration_ms=duration_ms,
        )
        return self._log_result(result)


class CodeSandbox(BaseTool):
    """Local subprocess sandbox for code and shell execution."""

    def __init__(
        self,
        *,
        python_executable: str = "python",
        working_directory: str | Path | None = None,
        timeout_seconds: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="code_sandbox", logger=logger)
        self.python_executable = python_executable
        self.working_directory = Path(working_directory) if working_directory else None
        self.timeout_seconds = timeout_seconds

    def invoke(self, **kwargs: Any) -> ToolResult:
        language = kwargs.get("language", "python")
        code = kwargs.get("code")
        if not code:
            raise ToolError("code sandbox requires code")

        started = time.perf_counter()
        if language == "python":
            command = [self.python_executable, "-c", code]
        elif language == "shell":
            command = ["sh", "-lc", code]
        else:
            raise ToolError(f"unsupported sandbox language: {language}")

        completed = subprocess.run(
            command,
            cwd=str(self.working_directory) if self.working_directory else None,
            capture_output=True,
            text=True,
            timeout=kwargs.get("timeout_seconds", self.timeout_seconds),
            check=False,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        ok = completed.returncode == 0
        output = completed.stdout if ok else completed.stderr or completed.stdout
        result = ToolResult(
            tool_name=self.name,
            ok=ok,
            output=output.strip(),
            metadata={"language": language, "returncode": completed.returncode},
            duration_ms=duration_ms,
        )
        return self._log_result(result)


class FileToolkit(BaseTool):
    """Placeholder for Stage 1 file toolkit work."""

    def __init__(self) -> None:
        super().__init__(name="file_toolkit")

    def invoke(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("FileToolkit is intentionally not implemented in Stage 1")


class MCPToolkit(BaseTool):
    """Placeholder for Stage 1 MCP toolkit work."""

    def __init__(self) -> None:
        super().__init__(name="mcp_toolkit")

    def invoke(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError("MCPToolkit is intentionally not implemented in Stage 1")


@dataclass(slots=True)
class RuntimeLimits:
    """Guard rails for recursive ROMA execution."""

    max_recursion_depth: int = 8
    max_subtasks_per_plan: int = 12
    max_total_tasks: int = 256
    max_expansions_per_goal: int = 3
    max_parallelism: int = 4


@dataclass(slots=True)
class ExecutorBinding:
    """Executor instance plus the tools it may access."""

    executor: Executor
    tool_names: frozenset[str] = field(default_factory=frozenset)


class ExecutorToolView:
    """Restricted tool surface exposed to a specific executor."""

    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = dict(tools)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"tool {name!r} is not allowed for this executor") from exc

    def list_names(self) -> list[str]:
        return sorted(self._tools)


class ComponentRegistry:
    """Holds component instances, executor routing, and tool permissions."""

    def __init__(
        self,
        *,
        atomizer: Atomizer,
        planner: Planner,
        aggregator: Aggregator,
        limits: RuntimeLimits | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.atomizer = atomizer
        self.planner = planner
        self.aggregator = aggregator
        self.limits = limits or RuntimeLimits()
        self.logger = logger or logging.getLogger("roma.registry")
        self._tools: dict[str, BaseTool] = {}
        self._executors: dict[TaskType, ExecutorBinding] = {}
        self._lock = threading.Lock()

    def register_tool(self, tool: BaseTool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def register_executor(
        self,
        executor: Executor,
        *,
        task_types: frozenset[TaskType] | None = None,
        allowed_tools: set[str] | frozenset[str] | None = None,
    ) -> None:
        supported = task_types or executor.supported_task_types
        binding = ExecutorBinding(
            executor=executor,
            tool_names=frozenset(allowed_tools or set()),
        )
        self._bind_tools(binding)
        with self._lock:
            for task_type in supported:
                self._executors[task_type] = binding

    def get_executor(self, task_type: TaskType) -> Executor:
        try:
            return self._executors[task_type].executor
        except KeyError as exc:
            raise RegistryError(f"no executor registered for task type {task_type.value}") from exc

    def get_executor_tool_names(self, task_type: TaskType) -> list[str]:
        try:
            return sorted(self._executors[task_type].tool_names)
        except KeyError as exc:
            raise RegistryError(f"no executor registered for task type {task_type.value}") from exc

    def validate(self) -> None:
        missing = [task_type.value for task_type in TaskType if task_type not in self._executors]
        if missing:
            raise RegistryError(f"missing executors for task types: {', '.join(missing)}")

    def _bind_tools(self, binding: ExecutorBinding) -> None:
        allowed = {name: self._tools[name] for name in binding.tool_names if name in self._tools}
        missing = sorted(name for name in binding.tool_names if name not in self._tools)
        if missing:
            raise RegistryError(f"executor references unknown tools: {', '.join(missing)}")

        set_tools = getattr(binding.executor, "set_tools", None)
        if callable(set_tools):
            set_tools(ExecutorToolView(allowed))


def resolve_future_result(future: Future[Any]) -> Any:
    """Unwrap concurrent future exceptions consistently."""
    return future.result()
