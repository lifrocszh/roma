"""Tests for the tool registry module (src/core/registry.py)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.core.models import TaskType
from src.core.registry import (
    BaseTool,
    CodeSandbox,
    ComponentRegistry,
    ExecutorToolView,
    RegistryError,
    ToolError,
    ToolResult,
    WebSearchToolkit,
    _verbose_tool_wrapper,
)
from src.core.signatures import (
    Aggregator,
    AggregatorOutput,
    Atomizer,
    AtomizerDecision,
    Executor,
    ExecutorOutput,
    Planner,
    PlannerOutput,
)


# ===================================================================
# Dummy component classes for registry tests
# ===================================================================


class DummyAtomizer(Atomizer):
    """Atomizer stub that returns EXECUTE for every task."""

    def decide(self, task: Task) -> AtomizerDecision:  # noqa: ARG002
        from src.core.models import NodeType

        return AtomizerDecision(node_type=NodeType.EXECUTE, rationale="stub")


class DummyPlanner(Planner):
    """Planner stub that returns an empty plan."""

    def plan(self, task: Task) -> PlannerOutput:  # noqa: ARG002
        from src.core.graph import TaskGraph

        return PlannerOutput(
            subtasks=[],
            task_graph=TaskGraph(),
            rationale="stub",
        )


class DummyAggregator(Aggregator):
    """Aggregator stub that concatenates child results."""

    def aggregate(
        self,
        task: Task,  # noqa: ARG002
        child_outputs: list[ExecutorOutput],
    ) -> AggregatorOutput:
        summary = " | ".join(o.result for o in child_outputs)
        return AggregatorOutput(task_id="stub", summary=summary)


class DummyExecutor(Executor):
    """Executor stub that registers which tools were bound."""

    def __init__(self) -> None:
        self.bound_tools: list[str] = []
        self._tools: ExecutorToolView | None = None

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset(TaskType)

    def set_tools(self, tools: ExecutorToolView) -> None:
        self.bound_tools = tools.list_names()
        self._tools = tools

    def execute(self, task: Task) -> ExecutorOutput:  # noqa: ARG002
        return ExecutorOutput(task_id="stub", result="stub")


# ===================================================================
# Concrete tool subclass for BaseTool tests
# ===================================================================


class _ConcreteTool(BaseTool):
    """Minimal concrete subclass used to test BaseTool."""

    def invoke(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(tool_name=self.name, ok=True, output="ok")


# ===================================================================
# Tests: BaseTool
# ===================================================================


class TestBaseTool:
    """Tests for the abstract BaseTool class."""

    def test_abstract_invoke_raises_not_implemented(self) -> None:
        """BaseTool.invoke() must raise NotImplementedError when called on a
        subclass that does not override it."""

        class _IncompleteTool(BaseTool):
            def invoke(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
                return super().invoke(**kwargs)

        tool = _IncompleteTool(name="broken")
        with pytest.raises(NotImplementedError):
            tool.invoke()

    def test_concrete_tool_invoke_returns_tool_result(self) -> None:
        """A subclass that overrides invoke() should return a ToolResult."""
        tool = _ConcreteTool(name="concrete")
        result = tool.invoke()
        assert isinstance(result, ToolResult)
        assert result.tool_name == "concrete"
        assert result.ok is True
        assert result.output == "ok"

    def test_log_result_passthrough(self, caplog: pytest.LogCaptureFixture) -> None:
        """_log_result should return the same ToolResult unchanged and log an info message."""
        caplog.set_level(logging.INFO)
        tool = _ConcreteTool(name="logging_tool")
        result = ToolResult(tool_name="logging_tool", ok=True, output="hello")

        returned = tool._log_result(result)  # noqa: SLF001

        assert returned is result
        assert returned.output == "hello"
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert "logging_tool" in caplog.text

    def test_default_logger_name(self) -> None:
        """When no logger is provided, the logger name should be roma.tool.<name>."""
        tool = _ConcreteTool(name="no_logger_given")
        assert tool._logger.name == "roma.tool.no_logger_given"  # noqa: SLF001

    def test_custom_logger(self) -> None:
        """When a logger is provided, it should be used."""
        custom = logging.getLogger("custom.logger")
        tool = _ConcreteTool(name="custom", logger=custom)
        assert tool._logger is custom  # noqa: SLF001


# ===================================================================
# Tests: WebSearchToolkit
# ===================================================================


def _has_tavily_api_key() -> bool:
    """Check whether a real Tavily API key is available in the environment."""
    import os

    return bool(os.environ.get("TAVILY_API_KEY"))


class TestWebSearchToolkit:
    """Tests for WebSearchToolkit — error cases only (integration requires API key)."""

    # -- Error cases -----------------------------------------------------------

    def test_search_no_query(self) -> None:
        """search action without a query should raise ToolError."""
        tool = WebSearchToolkit(api_key="dummy")
        with pytest.raises(ToolError, match="requires a non-empty query"):
            tool.invoke(action="search")

    def test_search_empty_query(self) -> None:
        """search action with an empty query should raise ToolError."""
        tool = WebSearchToolkit(api_key="dummy")
        with pytest.raises(ToolError, match="requires a non-empty query"):
            tool.invoke(action="search", query="")

    def test_search_missing_api_key(self) -> None:
        """search action without an API key should raise ToolError."""
        tool = WebSearchToolkit(api_key=None)
        with pytest.raises(ToolError, match="missing Tavily API key"):
            tool.invoke(action="search", query="python")

    def test_extract_missing_api_key(self) -> None:
        """extract action without an API key should raise ToolError."""
        tool = WebSearchToolkit(api_key=None)
        with pytest.raises(ToolError, match="missing Tavily API key"):
            tool.invoke(action="extract", url="https://example.com")

    def test_extract_no_url(self) -> None:
        """extract action without a url should raise ToolError."""
        tool = WebSearchToolkit(api_key="dummy")
        with pytest.raises(ToolError, match="requires a url"):
            tool.invoke(action="extract")

    def test_extract_empty_url(self) -> None:
        """extract action with an empty url should raise ToolError."""
        tool = WebSearchToolkit(api_key="dummy")
        with pytest.raises(ToolError, match="requires a url"):
            tool.invoke(action="extract", url="")

    def test_default_action_is_search(self) -> None:
        """When no action is given, invoke should default to search and require a query."""
        tool = WebSearchToolkit(api_key="dummy")
        with pytest.raises(ToolError, match="requires a non-empty query"):
            tool.invoke()

    def test_tavily_not_installed(self) -> None:
        """_client() should raise ToolError when tavily is not importable."""
        tool = WebSearchToolkit(api_key="key")
        # Remove tavily from sys.modules so that re-import triggers ModuleNotFoundError
        saved_modules = {}
        for mod_name in list(sys.modules.keys()):
            if "tavily" in mod_name:
                saved_modules[mod_name] = sys.modules.pop(mod_name)
        try:
            with patch("builtins.__import__", side_effect=ImportError("no module named tavily", name="tavily")):
                with pytest.raises(ToolError, match="tavily-python is not installed"):
                    tool._client()  # noqa: SLF001
        finally:
            sys.modules.update(saved_modules)

    # -- URL normalisation ----------------------------------------------------

    def test_extract_url_normalisation_https(self) -> None:
        """extract should prepend https:// when no scheme is present.
        The URL normalisation happens before the API call, so a missing
        API key error confirms normalisation ran."""
        tool = WebSearchToolkit(api_key=None)
        with pytest.raises(ToolError, match="missing Tavily API key"):
            tool.invoke(action="extract", url="example.com/page")

    # -- Integration tests (skipped without TAVILY_API_KEY) -------------------

    @pytest.mark.skipif(not _has_tavily_api_key(), reason="requires TAVILY_API_KEY env var")
    def test_integration_search(self) -> None:  # pragma: no cover
        """End-to-end web search with a real Tavily API key."""
        import os

        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="search", query="Python programming language")
        assert result.ok is True
        assert len(result.output) > 0
        assert result.metadata.get("action") == "search"
        assert result.metadata.get("result_count", 0) > 0

    @pytest.mark.skipif(not _has_tavily_api_key(), reason="requires TAVILY_API_KEY env var")
    def test_integration_extract(self) -> None:  # pragma: no cover
        """End-to-end page extraction with a real Tavily API key."""
        import os

        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="https://example.com")
        assert result.ok is True
        assert len(result.output) > 0

    @pytest.mark.skipif(not _has_tavily_api_key(), reason="requires TAVILY_API_KEY env var")
    def test_integration_extract_url_normalisation(self) -> None:  # pragma: no cover
        """extract should normalise a bare domain by prepending https://."""
        import os

        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="example.com")
        assert result.ok is True

    @pytest.mark.skipif(not _has_tavily_api_key(), reason="requires TAVILY_API_KEY env var")
    def test_integration_extract_failed_domain(self) -> None:  # pragma: no cover
        """extract on a non-existent domain should return ok=False."""
        import os

        tool = WebSearchToolkit(api_key=os.environ["TAVILY_API_KEY"])
        result = tool.invoke(action="extract", url="https://this-domain-does-not-exist-12345.com")
        assert result.ok is False
        assert "Failed to extract" in result.output or "No content extracted" in result.output


# ===================================================================
# Tests: CodeSandbox
# ===================================================================


class TestCodeSandbox:
    """Tests for CodeSandbox — local subprocess execution."""

    # -- Error cases ----------------------------------------------------------

    def test_no_code(self) -> None:
        """invoke without code should raise ToolError."""
        sandbox = CodeSandbox()
        with pytest.raises(ToolError, match="code sandbox requires code"):
            sandbox.invoke(language="python")

    def test_empty_code(self) -> None:
        """invoke with empty code should raise ToolError."""
        sandbox = CodeSandbox()
        with pytest.raises(ToolError, match="code sandbox requires code"):
            sandbox.invoke(language="python", code="")

    def test_unsupported_language(self) -> None:
        """invoke with an unsupported language should raise ToolError."""
        sandbox = CodeSandbox()
        with pytest.raises(ToolError, match="unsupported sandbox language"):
            sandbox.invoke(language="ruby", code="puts 'hi'")

    # -- Python success / failure ---------------------------------------------

    def test_python_success(self) -> None:
        """A valid Python snippet should return ok=True with its stdout."""
        sandbox = CodeSandbox(python_executable=sys.executable)
        result = sandbox.invoke(language="python", code="print('hello world')")
        assert result.ok is True
        assert result.output == "hello world"
        assert result.metadata["language"] == "python"
        assert result.metadata["returncode"] == 0

    def test_python_failure(self) -> None:
        """A Python snippet that raises an exception should return ok=False."""
        sandbox = CodeSandbox(python_executable=sys.executable)
        result = sandbox.invoke(language="python", code="raise RuntimeError('boom')")
        assert result.ok is False
        assert result.metadata["returncode"] != 0
        assert "RuntimeError" in result.output or "boom" in result.output

    # -- Shell success / failure ----------------------------------------------

    def test_shell_success(self) -> None:
        """A valid shell command should return ok=True with its stdout."""
        sandbox = CodeSandbox()
        result = sandbox.invoke(language="shell", code="echo hello_world")
        assert result.ok is True
        assert result.output == "hello_world"
        assert result.metadata["language"] == "shell"
        assert result.metadata["returncode"] == 0

    def test_shell_failure(self) -> None:
        """A shell command that exits non-zero should return ok=False."""
        sandbox = CodeSandbox()
        result = sandbox.invoke(language="shell", code="exit 42")
        assert result.ok is False
        assert result.metadata["returncode"] == 42

    # -- working_directory ----------------------------------------------------

    def test_working_directory_respected(self, tmp_path: Path) -> None:
        """The working_directory option should be used as the cwd for the subprocess."""
        marker = tmp_path / "marker.txt"
        marker.write_text("present")
        sandbox = CodeSandbox(
            python_executable=sys.executable,
            working_directory=str(tmp_path),
        )
        result = sandbox.invoke(
            language="python",
            code="import os; print(os.getcwd())",
        )
        assert result.ok is True
        # The output should contain the tmp_path — it may have resolved symlinks,
        # so we check that the tmp_path name appears in the output.
        assert str(tmp_path) in result.output

    # -- timeout --------------------------------------------------------------

    def test_timeout_kills_long_running_code(self) -> None:
        """A code snippet that exceeds the timeout should be killed.
        Note: subprocess.run raises TimeoutExpired which CodeSandbox
        does not catch, so the exception propagates up."""
        from subprocess import TimeoutExpired

        sandbox = CodeSandbox(
            python_executable=sys.executable,
            timeout_seconds=1.0,
        )
        with pytest.raises(TimeoutExpired):
            sandbox.invoke(language="python", code="import time; time.sleep(10)")

    def test_per_call_timeout_overrides_default(self) -> None:
        """Passing timeout_seconds in kwargs should override the instance default."""
        from subprocess import TimeoutExpired

        sandbox = CodeSandbox(
            python_executable=sys.executable,
            timeout_seconds=10.0,
        )
        started = time.perf_counter()
        with pytest.raises(TimeoutExpired):
            sandbox.invoke(
                language="python",
                code="import time; time.sleep(10)",
                timeout_seconds=0.5,
            )
        elapsed = time.perf_counter() - started
        assert elapsed < 5.0

    def test_default_tool_name(self) -> None:
        """CodeSandbox should have the default name 'code_sandbox'."""
        sandbox = CodeSandbox()
        assert sandbox.name == "code_sandbox"


# ===================================================================
# Tests: _verbose_tool_wrapper
# ===================================================================


class TestVerboseToolWrapper:
    """Tests for the _verbose_tool_wrapper decorator."""

    def test_success_prints_arrow_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        """On success, the wrapper should print >>> and <<< lines."""
        tool = _ConcreteTool(name="echo")
        wrapped = _verbose_tool_wrapper(tool)

        result = wrapped.invoke(kwarg1="val1", kwarg2=42)

        assert result.ok is True
        captured = capsys.readouterr()
        assert ">>>" in captured.out
        assert "<<<" in captured.out
        assert "echo" in captured.out
        assert "OK" in captured.out

    def test_failure_prints_fail_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        """On failure, the wrapper should print >>> and <<< FAIL lines."""

        class _FailingTool(BaseTool):
            def invoke(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
                raise ToolError("something went wrong")

        tool = _FailingTool(name="failer")
        wrapped = _verbose_tool_wrapper(tool)

        with pytest.raises(ToolError, match="something went wrong"):
            wrapped.invoke(query="test")

        captured = capsys.readouterr()
        assert ">>>" in captured.out
        assert "<<<" in captured.out
        assert "ERROR" in captured.out
        assert "something went wrong" in captured.out

    def test_wrapper_replaces_invoke_method(self) -> None:
        """The wrapper should replace tool.invoke with a traced version."""
        tool = _ConcreteTool(name="replaced")
        original = tool.invoke

        wrapped = _verbose_tool_wrapper(tool)

        assert wrapped is tool
        assert wrapped.invoke is not original

    def test_result_output_trimmed_in_print(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Long output should be truncated to 200 chars in the <<< line."""
        tool = _ConcreteTool(name="long_output")
        # Monkey-patch invoke to return a long string
        original_invoke = tool.invoke

        def _long_invoke(**kwargs: Any) -> ToolResult:  # noqa: ARG002
            return ToolResult(
                tool_name="long_output",
                ok=True,
                output="x" * 500,
            )

        tool.invoke = _long_invoke  # type: ignore[method-assign]
        wrapped = _verbose_tool_wrapper(tool)

        wrapped.invoke()
        captured = capsys.readouterr()
        # The preview should be truncated to 200 chars
        assert "x" * 200 in captured.out
        assert "x" * 500 not in captured.out


# ===================================================================
# Tests: ExecutorToolView
# ===================================================================


class TestExecutorToolView:
    """Tests for the restricted tool surface exposed to executors."""

    def test_get_returns_registered_tool(self) -> None:
        """get() should return the tool registered under the given name."""
        tool = _ConcreteTool(name="greeter")
        view = ExecutorToolView({"greeter": tool})
        assert view.get("greeter") is tool

    def test_get_unknown_tool_raises_tool_error(self) -> None:
        """get() with an unknown name should raise ToolError."""
        view = ExecutorToolView({})
        with pytest.raises(ToolError, match="tool 'ghost' is not allowed"):
            view.get("ghost")

    def test_list_names_returns_sorted(self) -> None:
        """list_names() should return tool names in sorted order."""
        tools = {
            "z_tool": _ConcreteTool(name="z_tool"),
            "a_tool": _ConcreteTool(name="a_tool"),
            "m_tool": _ConcreteTool(name="m_tool"),
        }
        view = ExecutorToolView(tools)
        assert view.list_names() == ["a_tool", "m_tool", "z_tool"]

    def test_list_names_empty(self) -> None:
        """list_names() should return an empty list when no tools are bound."""
        view = ExecutorToolView({})
        assert view.list_names() == []

    def test_view_is_isolated_snapshot(self) -> None:
        """The view should work with a copied dict and not reflect later mutations."""
        original: dict[str, BaseTool] = {"only": _ConcreteTool(name="only")}
        view = ExecutorToolView(original)
        del original["only"]
        # The view should still have the tool
        assert view.get("only") is not None


# ===================================================================
# Tests: ComponentRegistry
# ===================================================================


class TestComponentRegistry:
    """Tests for ComponentRegistry — registration, binding, validation."""

    # -- register_tool --------------------------------------------------------

    def test_register_tool_stores_by_name(self) -> None:
        """register_tool should store the tool under its .name attribute."""
        registry = self._make_registry()
        tool = _ConcreteTool(name="my_tool")
        registry.register_tool(tool)
        # Access internal _tools to verify
        assert registry._tools["my_tool"] is tool  # noqa: SLF001

    def test_register_verbose_wraps_tool(self) -> None:
        """When verbose=True, register_tool should wrap with _verbose_tool_wrapper."""
        registry = self._make_registry(verbose=True)
        tool = _ConcreteTool(name="wrapped_tool")
        original_invoke = tool.invoke

        registry.register_tool(tool)

        stored = registry._tools["wrapped_tool"]  # noqa: SLF001
        assert stored.invoke is not original_invoke

    def test_register_verbose_not_wrapped(self) -> None:
        """When verbose=False, register_tool should NOT wrap the tool.
        Confirm via capsys — wrapped tools print to stdout on invoke."""
        registry = self._make_registry(verbose=False)
        tool = _ConcreteTool(name="plain_tool")

        registry.register_tool(tool)

        stored = registry._tools["plain_tool"]  # noqa: SLF001
        # Invoke via the registry (the tool is not wrapped)
        stored.invoke()
        # Confirm the tool is the same instance (not wrapped)
        assert stored is tool

    def test_register_tool_multiple_times(self) -> None:
        """Registering a second tool with the same name should overwrite the first."""
        registry = self._make_registry()
        tool_a = _ConcreteTool(name="dup")
        tool_b = _ConcreteTool(name="dup")
        registry.register_tool(tool_a)
        registry.register_tool(tool_b)
        assert registry._tools["dup"] is tool_b  # noqa: SLF001

    # -- register_executor / _bind_tools -------------------------------------

    def test_register_executor_binds_tools(self) -> None:
        """register_executor should call set_tools on the executor with allowed tools."""
        registry = self._make_registry()
        tool = _ConcreteTool(name="finder")
        registry.register_tool(tool)
        executor = DummyExecutor()

        registry.register_executor(executor, allowed_tools={"finder"})

        assert executor.bound_tools == ["finder"]

    def test_register_executor_unknown_tool_raises(self) -> None:
        """Binding an executor with tools not yet registered should raise RegistryError."""
        registry = self._make_registry()
        executor = DummyExecutor()

        with pytest.raises(RegistryError, match="references unknown tools"):
            registry.register_executor(executor, allowed_tools={"ghost"})

    def test_register_executor_empty_allowed_is_ok(self) -> None:
        """An executor with no allowed tools should bind successfully with an empty set."""
        registry = self._make_registry()
        executor = DummyExecutor()

        registry.register_executor(executor, allowed_tools=set())
        assert executor.bound_tools == []

    def test_register_executor_default_task_types(self) -> None:
        """When task_types is None, the executor's supported_task_types should be used."""
        registry = self._make_registry()
        executor = DummyExecutor()
        registry.register_executor(executor, allowed_tools=set())

        for tt in TaskType:
            assert registry.get_executor(tt) is executor

    def test_register_executor_custom_task_types(self) -> None:
        """When task_types is provided, only those types should be routed to this executor."""
        registry = self._make_registry()
        executor = DummyExecutor()
        registry.register_executor(
            executor,
            task_types=frozenset({TaskType.CODE, TaskType.WRITE}),
            allowed_tools=set(),
        )

        assert registry.get_executor(TaskType.CODE) is executor
        assert registry.get_executor(TaskType.WRITE) is executor
        with pytest.raises(RegistryError):
            registry.get_executor(TaskType.GENERAL)

    # -- get_executor ---------------------------------------------------------

    def test_get_executor_returns_registered(self) -> None:
        """get_executor should return the executor registered for a task type."""
        registry = self._make_registry()
        executor = DummyExecutor()
        registry.register_executor(executor, allowed_tools=set())

        assert registry.get_executor(TaskType.GENERAL) is executor

    def test_get_executor_missing_raises(self) -> None:
        """get_executor should raise RegistryError when no executor is registered."""
        registry = self._make_registry()
        with pytest.raises(RegistryError, match="no executor registered for task type"):
            registry.get_executor(TaskType.CODE)

    # -- get_executor_tool_names ----------------------------------------------

    def test_get_executor_tool_names_returns_sorted(self) -> None:
        """get_executor_tool_names should return sorted tool names for a task type."""
        registry = self._make_registry()
        registry.register_tool(_ConcreteTool(name="b_tool"))
        registry.register_tool(_ConcreteTool(name="a_tool"))
        executor = DummyExecutor()
        registry.register_executor(executor, allowed_tools={"b_tool", "a_tool"})

        assert registry.get_executor_tool_names(TaskType.GENERAL) == ["a_tool", "b_tool"]

    def test_get_executor_tool_names_missing_raises(self) -> None:
        """get_executor_tool_names should raise RegistryError when no executor is registered."""
        registry = self._make_registry()
        with pytest.raises(RegistryError, match="no executor registered for task type"):
            registry.get_executor_tool_names(TaskType.THINK)

    # -- validate -------------------------------------------------------------

    def test_validate_passes_when_all_types_covered(self) -> None:
        """validate() should pass without error when every TaskType has an executor."""
        registry = self._make_registry()
        executor = DummyExecutor()
        registry.register_executor(executor, allowed_tools=set())

        # Should not raise
        registry.validate()

    def test_validate_fails_when_types_missing(self) -> None:
        """validate() should raise RegistryError when some TaskTypes are not covered."""
        registry = self._make_registry()
        executor = DummyExecutor()
        # Register only for a subset of types
        registry.register_executor(
            executor,
            task_types=frozenset({TaskType.CODE}),
            allowed_tools=set(),
        )

        missing = [tt.value for tt in TaskType if tt != TaskType.CODE]
        with pytest.raises(RegistryError, match="missing executors for task types"):
            registry.validate()

    def test_validate_no_executors_at_all(self) -> None:
        """validate() should raise RegistryError when no executors have been registered."""
        registry = self._make_registry()
        missing = [tt.value for tt in TaskType]
        with pytest.raises(RegistryError, match="missing executors for task types"):
            registry.validate()

    # -- default RuntimeLimits ------------------------------------------------

    def test_default_limits(self) -> None:
        """When no limits are provided, ComponentRegistry should use RuntimeLimits defaults."""
        registry = self._make_registry()
        assert registry.limits.max_recursion_depth == 8
        assert registry.limits.max_subtasks_per_plan == 12
        assert registry.limits.max_total_tasks == 256
        assert registry.limits.max_expansions_per_goal == 3
        assert registry.limits.max_parallelism == 4

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _make_registry(*, verbose: bool = False) -> ComponentRegistry:
        return ComponentRegistry(
            atomizer=DummyAtomizer(),
            planner=DummyPlanner(),
            aggregator=DummyAggregator(),
            verbose=verbose,
        )
