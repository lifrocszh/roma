from __future__ import annotations

import json
import logging

from src.core.inference import build_client, get_default_model
from src.core.models import Task
from src.core.signatures import AggregatorOutput, ExecutorOutput
from src.prompts.seed_prompts import AGGREGATOR_PROMPT


_log = logging.getLogger("roma.aggregator")


class DefaultAggregator:
    def __init__(self, *, prompt: str = AGGREGATOR_PROMPT, model: str | None = None) -> None:
        self.prompt = prompt
        self._model = model or get_default_model()

    def aggregate(self, task: Task, child_outputs: list[ExecutorOutput]) -> AggregatorOutput:
        if len(child_outputs) == 0:
            return AggregatorOutput(task_id=task.id, summary="No outputs to aggregate.", metadata={"child_count": 0})

        if len(child_outputs) == 1:
            return AggregatorOutput(task_id=task.id, summary=child_outputs[0].result, metadata={"child_count": 1})

        segments = "\n\n".join(f"## Segment {i+1}\n{out.result}" for i, out in enumerate(child_outputs))

        client = build_client()
        if client is None:
            combined = "\n\n".join(out.result for out in child_outputs)
            return AggregatorOutput(task_id=task.id, summary=combined, metadata={"child_count": len(child_outputs)})

        schema_lines = "\n".join(
            f'  "{k}": "{v}"'
            for k, v in {
                "reasoning": "Step-by-step analysis of how to merge these segments",
                "summary": "The synthesized, coherent response merging all segments without redundancy",
            }.items()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}\n\n"
                    "You MUST respond with valid JSON only, containing exactly these fields:\n"
                    f"{schema_lines}\n\n"
                    "Think step by step before producing your final answer."
                ),
            },
            {
                "role": "user",
                "content": f"Task: Synthesize the following into a coherent answer for: {task.goal}\n\nContext:\n{segments}",
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content) if (content and content.strip()) else None
            summary = result["summary"] if result and result.get("summary") else "\n\n".join(out.result for out in child_outputs)
        except Exception:
            _log.warning("LLM call failed: model=%s", self._model, exc_info=True)
            summary = "\n\n".join(out.result for out in child_outputs)

        return AggregatorOutput(task_id=task.id, summary=summary, metadata={"child_count": len(child_outputs)})
