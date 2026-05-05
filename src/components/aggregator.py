"""Stage 2 Aggregator implementation - LLM-driven."""

from __future__ import annotations

from src.core.inference import llm_judge
from src.core.models import Task
from src.core.signatures import Aggregator, AggregatorOutput, ExecutorOutput
from src.prompts.seed_prompts import AGGREGATOR_PROMPT


class DefaultAggregator(Aggregator):
    """LLM-driven aggregator that synthesizes child outputs into a coherent result."""

    def __init__(self, *, prompt: str = AGGREGATOR_PROMPT) -> None:
        self.prompt = prompt

    def aggregate(self, task: Task, child_outputs: list[ExecutorOutput]) -> AggregatorOutput:
        if len(child_outputs) == 0:
            return AggregatorOutput(
                task_id=task.id,
                summary="No outputs to aggregate.",
                metadata={"child_count": 0},
            )

        if len(child_outputs) == 1:
            return AggregatorOutput(
                task_id=task.id,
                summary=child_outputs[0].result,
                metadata={"child_count": 1, "preserved_segments": 1},
            )

        segments = []
        for i, out in enumerate(child_outputs):
            segments.append(f"## Segment {i+1}: {out.result}")

        context = "\n\n".join(segments)

        result = llm_judge(
            system=self.prompt,
            instructions=f"Synthesize the following segments into a coherent answer for: {task.goal}",
            context=context,
            output_schema={
                "reasoning": "Step-by-step analysis of how to merge these segments",
                "summary": "The synthesized, coherent response that merges all segments without redundancy",
            },
            temperature=0.2,
        )

        if result and result.get("summary"):
            summary = result["summary"]
        else:
            summary = "\n\n".join(out.result for out in child_outputs)

        return AggregatorOutput(
            task_id=task.id,
            summary=summary,
            metadata={
                "child_count": len(child_outputs),
                "preserved_segments": len(child_outputs),
            },
        )
