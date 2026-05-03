"""Stage 4 `roma optimize` command."""

from __future__ import annotations

from pathlib import Path

import typer

from src.config.loader import load_config
from src.optimization import GEPAPlus
from src.prompts import (
    AGGREGATOR_PROMPT,
    ATOMIZER_PROMPT,
    EXECUTOR_CODE_PROMPT,
    EXECUTOR_RETRIEVE_PROMPT,
    EXECUTOR_THINK_PROMPT,
    EXECUTOR_WRITE_PROMPT,
    PLANNER_PROMPT,
)


def optimize_command(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    opt = config.optimize
    base_prompt = opt.base_prompt or _resolve_prompt(opt.module_name)
    result = GEPAPlus(seed=opt.seed).optimize(
        opt.module_name,
        base_prompt,
        held_out_signals=opt.held_out_signals,
        rounds=opt.rounds,
        proposals_per_round=opt.proposals_per_round,
        topn=opt.topn,
        max_tokens=opt.max_tokens,
    )
    typer.echo(f"module={result.module_name}")
    typer.echo(f"best_score={result.best_score}")
    typer.echo(f"final_delta={result.final_delta}")
    typer.echo(result.optimized_prompt)


def _resolve_prompt(module_name: str) -> str:
    mapping = {
        "atomizer": ATOMIZER_PROMPT,
        "planner": PLANNER_PROMPT,
        "write": EXECUTOR_WRITE_PROMPT,
        "retrieve": EXECUTOR_RETRIEVE_PROMPT,
        "think": EXECUTOR_THINK_PROMPT,
        "code": EXECUTOR_CODE_PROMPT,
        "aggregator": AGGREGATOR_PROMPT,
    }
    return mapping.get(module_name.lower(), EXECUTOR_WRITE_PROMPT)
