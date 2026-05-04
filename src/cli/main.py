"""Typer CLI entrypoints for the ROMA reimplementation."""

from __future__ import annotations

from pathlib import Path

import typer

from src.cli.optimize import optimize_command
from src.cli.run import run_command
from src.core.models import TaskType
from src.evaluation.eval_MMLU_pro import run_benchmark_async as eval_mmlu_command


app = typer.Typer(
    help=(
        "ROMA command line interface.\n\n"
        "Current state:\n"
        "- Stage 0 foundations are implemented.\n"
        "- Stage 1 recursive controller and tool runtime are implemented.\n"
        "- Stage 2 default components are implemented.\n"
        "- Stage 3 GEPA+ prompt optimization is implemented.\n"
        "- Use 'roma run' or 'roma run-demo' to execute the workflow."
    ),
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)



@app.command("eval-mmlu")
def eval_mmlu(
    num_questions: int = typer.Option(500, "--num-questions", "-n", help="Number of questions to sample."),
    parallel: int = typer.Option(5, "--parallel", "-p", help="Number of parallel roma run calls."),
    output: Path = typer.Option(Path("outputs.json"), "--output", "-o", help="Output JSON file path."),
) -> None:
    """Run a balanced MMLU-Pro evaluation. Each question is answered via `roma run`."""
    import asyncio
    asyncio.run(
        eval_mmlu_command(
            num_questions=num_questions,
            parallel=parallel,
            output=output,
        )
    )


@app.callback()
def cli() -> None:
    """Run ROMA commands and demo workflows."""


@app.command("stage0-status")
def stage0_status() -> None:
    """Show the currently implemented architecture stages."""
    typer.echo("Stage 0 implemented: core contracts, task graph, artifact store, and signatures.")
    typer.echo("Stage 1 implemented: recursive controller, guards, scheduling, registry, and tools.")
    typer.echo("Stage 2 implemented: atomizer, planner, specialized executors, aggregator, and seed prompts.")
    typer.echo("Stage 3 implemented: proposal generation, judging, verification, merging, and GEPA+ optimization.")


@app.command("run")
def run(
    task: str = typer.Argument(..., help="Natural-language task to run."),
    config: str = typer.Option(None, "--config", help="Path to a YAML or JSON config file."),
    context: str = typer.Option(None, "--context", help="Optional context passed into the root task."),
    api_keys: str = typer.Option(None, "--api-keys", help="Path to a local TOML file with API keys."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress intermediate streamed events and print only final output."),
) -> None:
    """Run a natural-language task through ROMA.

    In normal mode all trace events and tool calls are shown.
    Use ``--quiet`` to print only the final answer.
    """
    run_command(task=task, config_path=config, context=context, api_keys_path=api_keys, quiet=quiet)


@app.command("optimize")
def optimize(config: str = typer.Option(None, "--config", help="Path to a YAML or JSON config file.")) -> None:
    """Optimize a configured ROMA prompt."""
    optimize_command(config_path=config)


@app.command("run-demo")
def run_demo(
    task: str = typer.Argument(
        "Write a multi-part project update then refine the final message",
        help="Task to run through the current Stage 2 ROMA workflow.",
    ),
    task_type: TaskType = typer.Option(
        TaskType.WRITE,
        "--task-type",
        case_sensitive=False,
        help="Task type used to route the demo through the registry.",
    ),
    context: str = typer.Option(
        "Status: tests pass. Risk: deployment timing.",
        "--context",
        help="Optional context passed into the root task.",
    ),
    api_keys_path: Path = typer.Option(
        Path("config/api_keys.toml"),
        "--api-keys",
        help="Path to a local TOML file containing provider API keys.",
    ),
) -> None:
    """Run the current ROMA demo workflow with the default Stage 2 components."""
    run_command(task=task, context=context, api_keys_path=api_keys_path)


def main() -> None:
    """Run the Typer application."""
    app()
