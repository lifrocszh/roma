"""Stage 4 `roma run` command."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tomllib

import typer


def _safe_echo(text: str) -> None:
    """Print *text* handling Unicode characters that may not be representable
    in the current terminal encoding (e.g. cp1252 on Windows)."""
    try:
        typer.echo(text)
    except UnicodeEncodeError:
        # Fall back to ASCII with replacement
        cleaned = text.encode("ascii", errors="xmlcharrefreplace").decode("ascii")
        typer.echo(cleaned)

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

    # --- API key resolution: openrouter > deepseek > openai ---
    openrouter_key = api_keys.get("openrouter_api_key")
    deepseek_key = api_keys.get("deepseek_api_key")
    openai_key = api_keys.get("openai_api_key")

    if openrouter_key and not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    if deepseek_key and not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if openai_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = openai_key

    # --- Model selection ---
    openrouter_model = api_keys.get("openrouter_model")
    if openrouter_model:
        os.environ.setdefault("ROMA_MODEL", openrouter_model)
    elif deepseek_key:
        os.environ.setdefault("ROMA_MODEL", "deepseek-v4-flash")
        os.environ.setdefault("ROMA_BASE_URL", "https://api.deepseek.com")
    elif openai_key:
        os.environ.setdefault("ROMA_MODEL", "gpt-5-mini")

    root_task = Task(
        id="run-root",
        goal=task,
        task_type=TaskType.GENERAL,
        context_input=context or demo.context_input,
    )
    tavily_api_key = api_keys.get("tavily_api_key")

    # In normal mode (not quiet) show both trace events and verbose tool calls
    show_traces = not quiet
    registry = build_default_registry(
        tavily_api_key=tavily_api_key,
        python_executable=sys.executable,
        limits=runtime.limits,
        verbose=show_traces,
    )
    controller = RomaController(registry, event_callback=_stream_event if show_traces else None)
    outcome = controller.solve(root_task)

    # Extract the final answer: the last child trace's output is the "answer" subtask
    child_outputs = outcome.trace.child_traces
    final_answer = (
        child_outputs[-1].output_summary
        if child_outputs
        else outcome.output.result
    )

    if quiet:
        _safe_echo(final_answer)
    else:
        sep = "=" * 60
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Result: {root_task.goal}")
        _safe_echo(sep)
        _safe_echo("")
        _safe_echo(final_answer)
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Subtasks: {len(child_outputs)}")
        _safe_echo(sep)


def _load_tavily_api_key(api_keys_path: Path) -> str | None:
    return _load_api_keys(api_keys_path).get("tavily_api_key")


def _load_api_keys(api_keys_path: Path) -> dict[str, str]:
    if not api_keys_path.exists():
        return {}
    data = tomllib.loads(api_keys_path.read_text(encoding="utf-8"))
    providers = data.get("providers", {})
    result: dict[str, str] = {}
    for key_name in ("openrouter_api_key", "openrouter_model", "tavily_api_key", "deepseek_api_key", "openai_api_key"):
        value = providers.get(key_name)
        if isinstance(value, str) and value.strip():
            result[key_name] = value.strip()
    return result


def _stream_event(kind: str, payload: dict[str, object], trace: object) -> None:
    """Stream a trace event to stdout with clean formatting."""
    task_id = getattr(trace, "task_id", "unknown")
    depth = str(task_id).count(".")
    indent = "  " * depth

    if kind == "task_started":
        _safe_echo(f"{indent}> start:  {payload.get('task_type', '?')} :: {task_id}")

    elif kind == "atomizer_decision":
        node = payload.get("node_type", "?")
        reason = payload.get("rationale", "")
        _safe_echo(f"{indent}> decide: {node} — {reason}")

    elif kind == "planner_output":
        count = payload.get("subtask_count", 0)
        _safe_echo(f"{indent}> plan:   {count} subtasks")

    elif kind == "executor_selected":
        ex = payload.get("executor", "?")
        tt = payload.get("task_type", "?")
        tools = payload.get("allowed_tools", [])
        tool_str = f" [{', '.join(tools)}]" if tools else ""
        _safe_echo(f"{indent}> run:    {ex} ({tt}){tool_str}")

    elif kind == "executor_completed":
        preview = (payload.get("result_preview", "") or "")[:160]
        _safe_echo(f"{indent}> done:   {preview}")

    elif kind == "child_started":
        goal = (payload.get("goal", "") or "")[:120]
        _safe_echo(f"{indent}> child:  {payload.get('task_id')} — {goal}")

    elif kind == "child_completed":
        preview = (payload.get("result_preview", "") or "")[:120]
        _safe_echo(f"{indent}> child:  {payload.get('task_id')} ✓ {preview}")

    elif kind == "batch_started":
        _safe_echo(f"{indent}> batch:  {payload.get('task_ids')}")

    elif kind == "aggregation_completed":
        preview = (payload.get("summary_preview", "") or "")[:120]
        _safe_echo(f"{indent}> merge:  {preview}")

    elif kind == "guard_triggered":
        _safe_echo(f"{indent}> GUARD:  {payload.get('error')}")

    elif kind == "planner_validation_failed":
        _safe_echo(f"{indent}> ERROR:  planner — {payload.get('error')}")

    elif kind == "task_failed":
        _safe_echo(f"{indent}> ERROR:  {payload.get('error')}")
