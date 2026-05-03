"""Stage 4 `roma run` command."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tomllib

import typer

from src.components import build_default_registry
from src.config.loader import load_config
from src.core.controller import RomaController
from src.core.models import Task, TaskType


def run_command(
    task: str,
    config_path: Path | None = None,
    context: str | None = None,
    api_keys_path: Path | None = None,
    quiet: bool = False,
) -> None:
    config = load_config(config_path)
    demo = config.demo
    runtime = config.runtime
    resolved_api_keys_path = Path(api_keys_path) if api_keys_path is not None else runtime.api_keys_path
    api_keys = _load_api_keys(resolved_api_keys_path)
    deepseek_key = api_keys.get("deepseek_api_key") or api_keys.get("openai_api_key")
    if deepseek_key and not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if deepseek_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = deepseek_key
    os.environ.setdefault("ROMA_MODEL", "deepseek-v4-flash")
    os.environ.setdefault("ROMA_BASE_URL", "https://api.deepseek.com")
    root_task = Task(
        id="run-root",
        goal=task,
        task_type=TaskType.GENERAL,
        context_input=context or demo.context_input,
    )
    tavily_api_key = api_keys.get("tavily_api_key")
    registry = build_default_registry(
        tavily_api_key=tavily_api_key,
        python_executable=sys.executable,
        limits=runtime.limits,
    )
    controller = RomaController(registry, event_callback=None if quiet else _stream_event)
    outcome = controller.solve(root_task)
    typer.echo("Task:")
    typer.echo(f"  {root_task.goal}")
    typer.echo("Result:")
    typer.echo(outcome.output.result)
    typer.echo(f"result={outcome.output.result}")
    typer.echo("Trace:")
    typer.echo(f"  child_traces={len(outcome.trace.child_traces)}")


def _load_tavily_api_key(api_keys_path: Path) -> str | None:
    return _load_api_keys(api_keys_path).get("tavily_api_key")


def _load_api_keys(api_keys_path: Path) -> dict[str, str]:
    if not api_keys_path.exists():
        return {}
    data = tomllib.loads(api_keys_path.read_text(encoding="utf-8"))
    providers = data.get("providers", {})
    result: dict[str, str] = {}
    for key_name in ("tavily_api_key", "deepseek_api_key", "openai_api_key"):
        value = providers.get(key_name)
        if isinstance(value, str) and value.strip():
            result[key_name] = value.strip()
    return result


def _stream_event(kind: str, payload: dict[str, object], trace: object) -> None:
    task_id = getattr(trace, "task_id", "unknown")
    depth = str(task_id).count(".")
    indent = "  " * depth
    if kind == "task_started":
        typer.echo(f"{indent}task_started: {task_id}")
    elif kind == "atomizer_decision":
        typer.echo(f"{indent}atomizer: {payload.get('node_type')} - {payload.get('rationale')}")
    elif kind == "planner_output":
        typer.echo(
            f"{indent}planner: {payload.get('subtask_count')} subtasks, batches={payload.get('dependency_batches')}"
        )
    elif kind == "batch_started":
        typer.echo(f"{indent}batch_started: {payload.get('task_ids')}")
    elif kind == "child_started":
        typer.echo(f"{indent}child_started: {payload.get('task_id')} -> {payload.get('goal')}")
    elif kind == "child_completed":
        typer.echo(f"{indent}child_completed: {payload.get('task_id')} :: {payload.get('result_preview')}")
    elif kind == "executor_selected":
        typer.echo(f"{indent}executor: {payload.get('executor')} ({payload.get('task_type')})")
    elif kind == "executor_completed":
        typer.echo(f"{indent}executor_completed: {payload.get('result_preview')}")
    elif kind == "aggregation_completed":
        typer.echo(f"{indent}aggregation: {payload.get('summary_preview')}")
    elif kind == "guard_triggered":
        typer.echo(f"{indent}guard_triggered: {payload.get('error')}")
    elif kind == "planner_validation_failed":
        typer.echo(f"{indent}planner_validation_failed: {payload.get('error')}")
    elif kind == "task_failed":
        typer.echo(f"{indent}task_failed: {payload.get('error')}")
