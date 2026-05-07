"""Run command — solves a task through the ROMA pipeline."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import typer
from dotenv import load_dotenv

load_dotenv()

from src.components import build_default_registry
from src.config.loader import load_config
from src.core.controller import RomaController
from src.core.models import ExecutionTrace, Task, TaskType


def _resolve(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _safe_echo(text: str) -> None:
    try:
        typer.echo(text)
    except UnicodeEncodeError:
        typer.echo(text.encode("ascii", errors="xmlcharrefreplace").decode("ascii"))


def _print_trace_tree(trace: ExecutionTrace, indent: int = 0) -> None:
    prefix = "  " * indent
    arrow = "" if indent == 0 else "\u2514\u2500 "
    node = trace.node_type.value if trace.node_type else "?"
    _safe_echo(f"{prefix}{arrow}[{node}] {trace.task_id} ({trace.task_type.value if trace.task_type else '?'})")
    _safe_echo(f"{prefix}  goal: {trace.goal[:80]}")
    if trace.output_summary:
        _safe_echo(f"{prefix}  result: {trace.output_summary[:120].replace(chr(10), ' ')}")
    for child in trace.child_traces:
        _print_trace_tree(child, indent + 1)


def run_command(
    task: str,
    config_path: Path | None = None,
    context: str | None = None,
    api_keys_path: Path | None = None,
    quiet: bool = False,
    show_trace: bool = False,
) -> None:
    config = load_config(config_path)
    runtime = config.runtime

    if api_keys_path:
        _load_toml_keys(Path(api_keys_path))
    _load_toml_keys(runtime.api_keys_path)

    model = _resolve("ROMA_MODEL", "openrouter_model")
    if model:
        os.environ.setdefault("ROMA_MODEL", model)
    elif _resolve("DEEPSEEK_API_KEY", "deepseek_api_key"):
        os.environ.setdefault("ROMA_MODEL", "deepseek-v4-flash")
        os.environ.setdefault("ROMA_BASE_URL", "https://api.deepseek.com")
    elif _resolve("OPENAI_API_KEY", "openai_api_key"):
        os.environ.setdefault("ROMA_MODEL", "gpt-5-mini")

    root = Task(
        id="root",
        goal=task,
        task_type=TaskType.GENERAL,
        context_input=context or config.demo.context_input,
    )

    registry = build_default_registry(
        tavily_api_key=_resolve("TAVILY_API_KEY", "tavily_api_key"),
        python_executable=sys.executable,
        limits=runtime.limits,
        verbose=not quiet,
    )
    controller = RomaController(registry, event_callback=_stream_event if not quiet else None)
    outcome = controller.solve(root)

    if show_trace:
        _print_trace_tree(outcome.trace)

    if quiet:
        _safe_echo(outcome.output.result)
    else:
        sep = "=" * 60
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Result: {root.goal}")
        _safe_echo(sep)
        _safe_echo("")
        _safe_echo(outcome.output.result)
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Subtasks: {len(outcome.trace.child_traces)}")
        _safe_echo(sep)


def _load_toml_keys(path: Path) -> None:
    if not path.exists():
        return
    try:
        import tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for k, v in data.get("providers", {}).items():
            if isinstance(v, str) and v.strip() and not os.getenv(k.upper()):
                os.environ[k.upper()] = v.strip()
    except ImportError:
        pass


def _stream_event(kind: str, payload: dict[str, object], trace: object) -> None:
    task_id = getattr(trace, "task_id", "?")
    depth = str(task_id).count(".")
    indent = "  " * depth

    if kind == "task_started":
        _safe_echo(f"{indent}> start:  {payload.get('task_type', '?')} :: {task_id}")
    elif kind == "atomizer_decision":
        _safe_echo(f"{indent}> decide: {payload.get('node_type')} \u2014 {payload.get('rationale', '')}")
    elif kind == "planner_output":
        _safe_echo(f"{indent}> plan:   {payload.get('subtask_count')} subtasks")
    elif kind == "executor_selected":
        tools = payload.get("allowed_tools", [])
        ts = f" [{', '.join(tools)}]" if tools else ""
        _safe_echo(f"{indent}> run:    {payload.get('executor')} ({payload.get('task_type')}){ts}")
    elif kind == "executor_completed":
        _safe_echo(f"{indent}> done:   {(payload.get('result_preview') or '')[:160]}")
    elif kind == "child_started":
        _safe_echo(f"{indent}> child:  {payload.get('task_id')} \u2014 {(payload.get('goal') or '')[:120]}")
    elif kind == "child_completed":
        _safe_echo(f"{indent}> child:  {payload.get('task_id')} \u2713 {(payload.get('result_preview') or '')[:120]}")
    elif kind == "batch_started":
        _safe_echo(f"{indent}> batch:  {payload.get('task_ids')}")
    elif kind == "aggregation_completed":
        _safe_echo(f"{indent}> merge:  {(payload.get('summary_preview') or '')[:120]}")
    elif kind == "guard_triggered":
        _safe_echo(f"{indent}> GUARD:  {payload.get('error')}")
    elif kind == "planner_validation_failed":
        _safe_echo(f"{indent}> ERROR:  planner \u2014 {payload.get('error')}")
    elif kind == "task_failed":
        _safe_echo(f"{indent}> ERROR:  {payload.get('error')}")
