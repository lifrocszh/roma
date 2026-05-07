from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from src.core.inference import build_client, get_default_model
from src.core.models import Task
from src.core.registry import BaseTool, ToolError
from src.core.signatures import ExecutorOutput
from src.prompts.seed_prompts import EXECUTOR_PROMPT


_log = logging.getLogger("roma.executor")
_MAX_TOOL_ROUNDS = 5


class UnifiedExecutor:
    def __init__(self, *, model: str | None = None) -> None:
        self._model = model or get_default_model()
        self._tools: dict[str, BaseTool] = {}

    def set_tools(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    def execute(self, task: Task) -> ExecutorOutput:
        client = build_client()
        if client is None:
            return ExecutorOutput(task_id=task.id, result=f"Analysis for '{task.goal}'", metadata={"strategy": "no_client"})

        context = task.context_input or ""
        messages = [
            {"role": "system", "content": EXECUTOR_PROMPT},
            {"role": "user", "content": f"Task: {task.goal}\n\nContext:\n{context}" if context.strip() else f"Task: {task.goal}"},
        ]

        tool_defs = [t.tool_spec() for t in self._tools.values()] if self._tools else None
        kwargs: dict[str, Any] = dict(model=self._model, messages=messages, temperature=0.0, max_tokens=4096)
        if tool_defs:
            kwargs["tools"] = tool_defs
            kwargs["tool_choice"] = "auto"

        try:
            final_content = None

            for _ in range(_MAX_TOOL_ROUNDS):
                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                if msg.content and msg.content.strip():
                    final_content = msg.content.strip()

                if not msg.tool_calls:
                    if final_content:
                        return ExecutorOutput(task_id=task.id, result=final_content, metadata={"strategy": "direct"})
                    return ExecutorOutput(task_id=task.id, result=f"Analysis for '{task.goal}'", metadata={"strategy": "empty"})

                messages.append(msg)
                for tc in msg.tool_calls:
                    tool = self._tools.get(tc.function.name)
                    if tool is None:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: tool {tc.function.name} not available",
                        })
                        continue
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        result = tool.invoke(**args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result.output,
                        })
                    except ToolError as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: {e}",
                        })
                    except json.JSONDecodeError:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: invalid arguments: {tc.function.arguments}",
                        })

                kwargs["messages"] = messages

            final = client.chat.completions.create(**kwargs)
            text = (final.choices[0].message.content or "").strip() or final_content or f"Analysis for '{task.goal}'"
            return ExecutorOutput(task_id=task.id, result=text, metadata={"strategy": "tool_loop"})

        except Exception:
            _log.warning("LLM call failed: model=%s", self._model, exc_info=True)
            return ExecutorOutput(task_id=task.id, result=f"Analysis for '{task.goal}'", metadata={"strategy": "error"})
