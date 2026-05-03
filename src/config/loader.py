"""Stage 4 config loader for ROMA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.schema import AppConfig


class ConfigLoadError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigLoadError(f"configuration file not found: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    try:
        if config_path.suffix.lower() in {".json"}:
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw) or {}
        return AppConfig.model_validate(data)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigLoadError(f"failed to parse configuration file {config_path}: {exc}") from exc
    except ValidationError as exc:
        raise ConfigLoadError(f"invalid configuration file {config_path}: {exc}") from exc


def dump_config(config: AppConfig) -> dict[str, Any]:
    return config.model_dump(mode="python")
