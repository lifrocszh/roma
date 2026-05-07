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
from src.core.models import ExecutionTrace, Task


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


def _make_streamer():
    """Build an event callback that streams formatted output in real-time.

    Returns (event_callback, get_total_subtasks) where:
      - event_callback is passed to RomaController
      - get_total_subtasks() returns the total count after execution
    """

    stack: list[dict] = []
    subtask_counter = 0

    def _indent(task_id: str) -> tuple[str, str, str]:
        """Return (prefix, branch_char, marker) for the current depth."""
        depth = task_id.count(".") if task_id != "root" else 0
        if depth == 0:
            return ("", "", ">")
        parent_depth = depth - 1
        # Check if this task is the last sibling at its level
        is_last = False
        if len(stack) >= 2:
            parent = stack[-2]
            siblings = parent.get("children", [])
            if siblings and siblings[-1] == task_id:
                is_last = True
        bar = " " if is_last else "\u2502"
        base = bar * parent_depth
        branch = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        prefix = "   ".join(c for c in base) if base else ""
        marker = ">" * (depth + 1)
        return (prefix, branch, marker)

    def _register_child(parent_id: str, child_id: str) -> None:
        for frame in stack:
            if frame["id"] == parent_id:
                frame.setdefault("children", []).append(child_id)
                break

    def event_callback(kind: str, payload: dict, trace: ExecutionTrace) -> None:
        nonlocal subtask_counter
        task_id = trace.task_id
        task_type = "?"

        if kind == "task_started":
            stack.append({"id": task_id, "type": task_type, "goal": trace.goal, "children": []})
            prefix, branch, _ = _indent(task_id)
            label = f"{task_id} ({task_type})"
            _safe_echo(f"{prefix}{branch}{label} - {trace.goal[:100]}")

        elif kind == "atomizer_decision":
            node = payload.get("node_type", "?")
            prefix, _, marker = _indent(task_id)
            granted = payload.get("granted_tools", [])
            tools_str = f" tools={granted}" if granted else ""
            if node == "PLAN":
                _safe_echo(f"{prefix}    {marker} {task_id}: Atomizer")
                _safe_echo(f"{prefix}    {marker} {task_id}: Atomizer decision - Cannot atomize")
            else:
                _safe_echo(f"{prefix}    {marker} {task_id}: Atomizer")
                _safe_echo(f"{prefix}    {marker} {task_id}: Atomizer decision - Can atomize {tools_str}")

        elif kind == "planner_output":
            prefix, _, marker = _indent(task_id)
            deps = payload.get("dependency_batches", [])
            flat = [tid for batch in deps for tid in batch]
            _safe_echo(f"{prefix}    {marker} {task_id}: Planner")
            _safe_echo(f"{prefix}    {marker} {task_id}: Planner subtasks - {', '.join(flat)}")
            for cid in flat:
                _register_child(task_id, cid)

        elif kind == "child_started":
            cid = payload.get("task_id", "?")
            _register_child(task_id, cid)

        elif kind == "executor_completed":
            prefix, _, marker = _indent(task_id)
            _safe_echo(f"{prefix}    {marker} {task_id}: Executor")
            preview = (payload.get("result_preview", "") or "")[:200]
            _safe_echo(f"{prefix}    {marker} {task_id}: Executor result - {preview}")
            subtask_counter += 1

        elif kind == "executor_failed":
            prefix, _, marker = _indent(task_id)
            _safe_echo(f"{prefix}    {marker} {task_id}: Executor")
            error = payload.get("error", "unknown error")
            _safe_echo(f"{prefix}    {marker} {task_id}: Executor result - ERROR: {error}")

        elif kind == "aggregation_completed":
            pass

    def get_total() -> int:
        return subtask_counter

    return event_callback, get_total


def run_command(
    task: str,
    config_path: Path | None = None,
    context: str | None = None,
    api_keys_path: Path | None = None,
    quiet: bool = False,
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
        context_input=context or config.demo.context_input,
    )

    show_verbose = not quiet
    streamer, get_total = _make_streamer()

    registry = build_default_registry(
        tavily_api_key=_resolve("TAVILY_API_KEY", "tavily_api_key"),
        python_executable=sys.executable,
        limits=runtime.limits,
        verbose=False,
    )
    controller = RomaController(registry, event_callback=streamer if show_verbose else None)
    outcome = controller.solve(root)

    total = get_total()
    sep = "=" * 60
    if not quiet:
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Result: {root.goal}")
        _safe_echo(sep)
        _safe_echo("")
    _safe_echo(outcome.output.result)
    if not quiet:
        _safe_echo("")
        _safe_echo(sep)
        _safe_echo(f"  Subtasks: {total}")
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
