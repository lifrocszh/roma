"""Config loader."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config.schema import AppConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    raw = config_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw) or {}
        return AppConfig.model_validate(data)
    except (yaml.YAMLError, ValidationError):
        return AppConfig()
