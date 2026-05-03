from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.cli.main import app


runner = CliRunner()


def test_run_demo_command_executes_workflow(tmp_path: Path) -> None:
    api_keys = tmp_path / "api_keys.toml"
    api_keys.write_text("[providers]\ntavily_api_key = \"\"\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "run-demo",
            "Write a multi-part project update then refine the final message",
            "--api-keys",
            str(api_keys),
        ],
    )

    assert result.exit_code == 0
    assert "Task:" in result.stdout
    assert "Result:" in result.stdout
    assert "Trace:" in result.stdout
    assert "child_traces=3" in result.stdout
