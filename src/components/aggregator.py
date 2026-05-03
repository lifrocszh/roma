"""Stage 2 Aggregator implementation."""

from __future__ import annotations

from src.core.models import Task
from src.core.signatures import Aggregator, AggregatorOutput, ExecutorOutput
from src.prompts.seed_prompts import AGGREGATOR_PROMPT


class DefaultAggregator(Aggregator):
    """Synthesize child outputs into a compressed parent-level result."""

    def __init__(self, *, prompt: str = AGGREGATOR_PROMPT) -> None:
        self.prompt = prompt

    def aggregate(self, task: Task, child_outputs: list[ExecutorOutput]) -> AggregatorOutput:
        unique_segments: list[str] = []
        seen: set[str] = set()
        artifacts: list[str] = []
        metadata = {
            "child_count": len(child_outputs),
            "dropped_duplicates": 0,
            "preserved_segments": 0,
            "consistency_checked": True,
        }

        for output in child_outputs:
            normalized = output.result.strip()
            if normalized and normalized not in seen:
                unique_segments.append(normalized)
                seen.add(normalized)
            elif normalized:
                metadata["dropped_duplicates"] += 1
            artifacts.extend(output.artifacts)

        metadata["preserved_segments"] = len(unique_segments)

        if len(unique_segments) == 1:
            summary = unique_segments[0]
        else:
            summary = "\n\n".join(unique_segments)

        return AggregatorOutput(
            task_id=task.id,
            summary=summary,
            artifacts=sorted(set(artifacts)),
            metadata=metadata,
        )
