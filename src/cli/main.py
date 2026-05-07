"""ROMI CLI — run command only."""

from pathlib import Path

import typer

from src.cli.run import run_command

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("run")
def run(
    task: str = typer.Argument(..., help="Natural-language task to run."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
    context: str = typer.Option(None, "--context", help="Optional context passed to the root task."),
    api_keys: str = typer.Option(None, "--api-keys", help="Path to TOML or .env file with API keys."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress intermediate streaming, show only result."),
) -> None:
    run_command(task=task, config_path=config, context=context, api_keys_path=api_keys, quiet=quiet)


def main() -> None:
    app()
