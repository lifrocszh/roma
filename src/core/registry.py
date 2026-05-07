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

from src.core.models import CONTRACT_VERSION


OpenAIToolSpec = dict[str, Any]


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

    @abstractmethod
    def tool_spec(self) -> OpenAIToolSpec:
        """Return the OpenAI-compatible function-calling definition."""

    def invoke_with_args(self, args: dict[str, Any]) -> ToolResult:
        return self.invoke(**args)

    def _log_result(self, result: ToolResult) -> ToolResult:
        self._logger.info(
            "tool=%s ok=%s duration_ms=%.2f metadata=%s",
            result.tool_name,
            result.ok,
            result.duration_ms,
            result.metadata,
        )
        return result


def _verbose_tool_wrapper(tool: BaseTool) -> BaseTool:
    """Wrap a tool so every invocation prints inputs and outputs to stdout."""
    original_invoke = tool.invoke

    def _invoke_with_trace(**kwargs: Any) -> ToolResult:
        args_preview = ", ".join(
            f"{k}={repr(v)[:120]}" for k, v in kwargs.items()
        )
        print(f">>> tool [{tool.name}]({args_preview})")
        started = time.perf_counter()
        try:
            result = original_invoke(**kwargs)
            elapsed = (time.perf_counter() - started) * 1000
            status = "OK" if result.ok else "FAIL"
            out_preview = result.output[:200].replace("\n", "\\n")
            print(f"<<< tool [{tool.name}] {status} ({elapsed:.0f}ms): {out_preview}")
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            print(f"<<< tool [{tool.name}] ERROR ({elapsed:.0f}ms): {exc}")
            raise

    tool.invoke = _invoke_with_trace  # type: ignore[method-assign]
    return tool


class WebSearchToolkit(BaseTool):
    """Web tool that supports both search and page content extraction.

    Usage modes (controlled via the ``action`` kwarg):

    * ``search`` (default) — Tavily-backed web search. Expects ``query``.
    * ``extract`` — Tavily-backed page content extraction. Expects ``url``.
    """

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

    def tool_spec(self) -> OpenAIToolSpec:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information or extract content from a URL. Use 'search' action for general queries, 'extract' to get full page content from a specific URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search", "extract"],
                            "description": "'search' to find information on a topic, 'extract' to get the full content of a specific URL",
                        },
                        "query": {
                            "type": "string",
                            "description": "The search query or URL to extract from",
                        },
                    },
                    "required": ["action", "query"],
                },
            },
        }

    def _client(self) -> object:
        if not self.api_key:
            raise ToolError("missing Tavily API key")
        try:
            from tavily import TavilyClient

            return TavilyClient(api_key=self.api_key)
        except ImportError as exc:
            raise ToolError("tavily-python is not installed") from exc

    def invoke(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "search")
        if action == "extract":
            return self._extract(**kwargs)
        return self._search(**kwargs)

    def _search(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not query:
            raise ToolError("web_search (search) requires a non-empty query")

        if len(query) > 400:
            query = query[:397] + "..."

        started = time.perf_counter()
        client = self._client()
        response = client.search(query=query, max_results=kwargs.get("max_results", self.max_results))
        duration_ms = (time.perf_counter() - started) * 1000
        result = ToolResult(
            tool_name=self.name,
            ok=True,
            output=str(response),
            metadata={"query": query, "result_count": len(response.get("results", [])), "action": "search"},
            duration_ms=duration_ms,
        )
        return self._log_result(result)

    def _extract(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url") or kwargs.get("query")
        if not url:
            raise ToolError("web_search (extract) requires a url")

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        started = time.perf_counter()
        client = self._client()
        response = client.extract(urls=[url])
        duration_ms = (time.perf_counter() - started) * 1000

        results = response.get("results", [])
        failed = response.get("failed_results", [])

        if results and results[0].get("raw_content"):
            raw = results[0]["raw_content"]
            output = raw[:15000] + "\n\n[... truncated ...]" if len(raw) > 15000 else raw
            ok = True
        elif failed:
            err = failed[0].get("error", "unknown error")
            output = f"Failed to extract {url}: {err}"
            ok = False
        else:
            output = f"No content extracted from {url}"
            ok = False

        result = ToolResult(
            tool_name=self.name,
            ok=ok,
            output=output,
            metadata={
                "url": url,
                "action": "extract",
                "content_length": len(output),
                "failed_count": len(failed),
            },
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

    def tool_spec(self) -> OpenAIToolSpec:
        return {
            "type": "function",
            "function": {
                "name": "code_sandbox",
                "description": "Execute Python or shell code in a secure sandbox. Use for calculations, data processing, running scripts, or any task that needs code execution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "shell"],
                            "description": "The programming language to use",
                        },
                        "code": {
                            "type": "string",
                            "description": "The code to execute. Use print() to output results.",
                        },
                    },
                    "required": ["language", "code"],
                },
            },
        }

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

class Calculator(BaseTool):
    """Safe arithmetic calculator for evaluating math expressions.

    Supports basic arithmetic (``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``),
    parentheses, and common math functions via Python's ``math`` module.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(name="calculator", logger=logger)

    def tool_spec(self) -> OpenAIToolSpec:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /, //, %, **), parentheses, and math functions (sqrt, sin, cos, log, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate (e.g., '(3 + 5) * 2' or 'sqrt(144)')",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }

    def invoke(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get("expression") or kwargs.get("query")
        if not expression:
            raise ToolError("calculator requires an expression")

        started = time.perf_counter()
        try:
            result = self._evaluate(expression)
            print(f"Calculating: {expression} = {result}")
            ok = True
            output = str(result)
            extra = {"expression": expression, "result": result}
        except Exception as exc:
            ok = False
            output = f"Error: {exc}"
            extra = {"expression": expression}

        duration_ms = (time.perf_counter() - started) * 1000
        result = ToolResult(
            tool_name=self.name,
            ok=ok,
            output=output,
            metadata=extra,
            duration_ms=duration_ms,
        )
        return self._log_result(result)

    @staticmethod
    def _evaluate(expr: str) -> float | int:
        import ast
        import math
        import operator as op

        operators: dict[type, object] = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Mod: op.mod,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
        }

        allowed_names: dict[str, object] = {
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
            "nan": math.nan,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
            "degrees": math.degrees,
            "radians": math.radians,
        }

        def _eval(node: ast.AST) -> float | int:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise TypeError(f"unsupported constant: {node.value!r}")
            if isinstance(node, ast.UnaryOp):
                fn = operators.get(type(node.op))
                if fn is None:
                    raise TypeError(f"unsupported operator: {type(node.op).__name__}")
                return fn(_eval(node.operand))
            if isinstance(node, ast.BinOp):
                fn = operators.get(type(node.op))
                if fn is None:
                    raise TypeError(f"unsupported operator: {type(node.op).__name__}")
                return fn(_eval(node.left), _eval(node.right))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fn = allowed_names.get(node.func.id)
                    if fn is None:
                        raise NameError(f"unknown function: {node.func.id}")
                    args = [_eval(arg) for arg in node.args]
                    return fn(*args)
                raise TypeError("only simple function calls are supported")
            if isinstance(node, ast.Name):
                val = allowed_names.get(node.id)
                if val is None:
                    raise NameError(f"unknown name: {node.id}")
                return val
            raise TypeError(f"unsupported syntax: {type(node).__name__}")

        tree = ast.parse(expr.strip(), mode="eval")
        return _eval(tree.body)


@dataclass(slots=True)
class RuntimeLimits:
    """Guard rails for recursive ROMA execution."""

    max_recursion_depth: int = 8
    max_subtasks_per_plan: int = 6
    max_total_tasks: int = 256
    max_expansions_per_goal: int = 3
    max_parallelism: int = 4


class ComponentRegistry:
    """Holds component instances and a shared tool pool.

    Stores both the tool instances (for invocation) and their OpenAI-compatible
    definitions (for LLM tool selection). The atomizer decides which tools to
    grant per task, and only those tools are passed to the executor.

    When *verbose* is ``True``, every tool call is printed to stdout.
    """

    def __init__(
        self,
        *,
        executor: Any,
        atomizer: Any,
        planner: Any,
        aggregator: Any,
        limits: RuntimeLimits | None = None,
        logger: logging.Logger | None = None,
        verbose: bool = False,
    ) -> None:
        self.executor = executor
        self.atomizer = atomizer
        self.planner = planner
        self.aggregator = aggregator
        self.limits = limits or RuntimeLimits()
        self.logger = logger or logging.getLogger("roma.registry")
        self._verbose = verbose
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        if self._verbose:
            tool = _verbose_tool_wrapper(tool)
        self._tools[tool.name] = tool

    def get_tool_definitions(self) -> list[OpenAIToolSpec]:
        """Return all registered tools in OpenAI-compatible format."""
        return [tool.tool_spec() for tool in self._tools.values()]

    def get_tool(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"tool {name!r} not found in registry")

    def validate(self) -> None:
        missing = [t for t in ("calculator", "web_search", "code_sandbox") if t not in self._tools]
        if missing:
            raise RegistryError(f"missing tools: {', '.join(missing)}")
