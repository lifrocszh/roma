from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli.main import app
from src.config.loader import ConfigLoadError, load_config


runner = CliRunner()


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "roma.yaml"
    config.write_text(
        """
demo:
  goal: "Write a release summary"
  task_type: "WRITE"
optimize:
  module_name: "write"
  rounds: 1
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.demo.goal == "Write a release summary"
    assert loaded.optimize.module_name == "write"


def test_load_config_from_json(tmp_path: Path) -> None:
    config = tmp_path / "roma.json"
    config.write_text(
        json.dumps(
            {
                "demo": {"goal": "Summarize the plan", "task_type": "WRITE"},
                "optimize": {"module_name": "atomizer", "rounds": 2},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.demo.goal == "Summarize the plan"
    assert loaded.optimize.rounds == 2


def test_load_config_rejects_invalid_file(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("demo: [", encoding="utf-8")

    try:
        load_config(config)
        raise AssertionError("expected ConfigLoadError")
    except ConfigLoadError:
        pass


def test_run_command_uses_configured_demo(tmp_path: Path) -> None:
    config = tmp_path / "roma.yaml"
    config.write_text(
        """
demo:
  goal: "Write a compact update"
  task_type: "WRITE"
  context_input: "Context from config"
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", "Write a compact update", "--config", str(config)])

    assert result.exit_code == 0
    assert "Write a compact update" in result.stdout


def test_run_command_accepts_deepseek_key_name(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "roma.yaml"
    config.write_text(
        """
demo:
  goal: "Write a compact update"
  task_type: "WRITE"
  context_input: "Context from config"
""".strip(),
        encoding="utf-8",
    )
    api_keys = tmp_path / "api_keys.toml"
    api_keys.write_text(
        """
[providers]
deepseek_api_key = "ds-test-key"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ROMA_MODEL", raising=False)
    monkeypatch.delenv("ROMA_BASE_URL", raising=False)

    result = runner.invoke(app, ["run", "Write a compact update", "--config", str(config), "--api-keys", str(api_keys)])

    assert result.exit_code == 0


def test_run_command_quiet_suppresses_streamed_events(tmp_path: Path) -> None:
    config = tmp_path / "roma.yaml"
    config.write_text(
        """
demo:
  goal: "Write a compact update"
  task_type: "WRITE"
  context_input: "Context from config"
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", "Write a compact update", "--config", str(config), "--quiet"])

    assert result.exit_code == 0
    assert "task_started:" not in result.stdout
    assert "atomizer:" not in result.stdout
    # Quiet mode now prints only the final answer (no "Result:" or "result=" prefix)


def test_optimize_command_uses_configured_base_prompt(tmp_path: Path) -> None:
    config = tmp_path / "roma.yaml"
    config.write_text(
        """
optimize:
  module_name: "write"
  base_prompt: "Role: write clear, purpose-driven prose."
  rounds: 1
  proposals_per_round: 2
  topn: 1
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["optimize", "--config", str(config)])

    assert result.exit_code == 0
    assert "module=write" in result.stdout
    assert "final_delta=" in result.stdout
