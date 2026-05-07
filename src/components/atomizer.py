from __future__ import annotations

import json
import logging

from openai import OpenAI

from src.core.inference import build_client, get_default_model
from src.core.models import NodeType, Task
from src.core.signatures import AtomizerDecision
from src.prompts.seed_prompts import ATOMIZER_PROMPT


_MAX_PLAN_DEPTH = 2
_log = logging.getLogger("roma.atomizer")


class DefaultAtomizer:
    def __init__(self, *, prompt: str = ATOMIZER_PROMPT, model: str | None = None) -> None:
        self.prompt = prompt
        self._model = model or get_default_model()
        self._tool_definitions: list[dict] = []

    def set_tool_definitions(self, defs: list[dict]) -> None:
        self._tool_definitions = defs

    def decide(self, task: Task) -> AtomizerDecision:
        if task.metadata.get("force_node_type") == NodeType.EXECUTE.value:
            return AtomizerDecision(
                node_type=NodeType.EXECUTE,
                rationale="marked as leaf by planner",
                granted_tools=["calculator", "code_sandbox"],
            )

        depth = task.metadata.get("_depth", 0)
        if isinstance(depth, int) and depth >= _MAX_PLAN_DEPTH:
            return AtomizerDecision(
                node_type=NodeType.EXECUTE,
                rationale=f"max plan depth {_MAX_PLAN_DEPTH} reached",
                granted_tools=["calculator", "web_search", "code_sandbox"],
            )

        context = task.context_input or ""
        tools_desc = ""
        available_names = ""
        if self._tool_definitions:
            lines = []
            names = []
            for td in self._tool_definitions:
                fn = td.get("function", {})
                name = fn.get("name", "?")
                desc = fn.get("description", "")
                lines.append(f"- {name}: {desc}")
                names.append(name)
            tools_desc = "\nAvailable tools:\n" + "\n".join(lines)
            available_names = ", ".join(names)

        schema_lines = "\n".join(
            f'  "{k}": "{v}"'
            for k, v in {
                "reasoning": "Step-by-step analysis of whether this task needs decomposition",
                "node_type": "PLAN if the task requires decomposition, coordination, or multiple steps; EXECUTE if one pass suffices",
                "rationale": "Brief explanation of the decision",
                "granted_tools": f"If EXECUTE: list of tool names the task needs. If PLAN: empty list. Available: {available_names}" if available_names else "none",
            }.items()
        )

        client = build_client()
        if client is None:
            return AtomizerDecision(
                node_type=NodeType.EXECUTE,
                rationale="no LLM client available, fallback to execute",
                granted_tools=["calculator", "web_search", "code_sandbox"],
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.prompt}{tools_desc}\n\n"
                    "You MUST respond with valid JSON only, containing exactly these fields:\n"
                    f"{schema_lines}\n\n"
                    "Think step by step before producing your final answer."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task.goal}\n\nContext:\n{context}" if context.strip() else f"Task: {task.goal}",
            },
        ]

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content) if (content and content.strip()) else None
        except Exception:
            _log.warning("LLM call failed: model=%s", self._model, exc_info=True)
            result = None

        if result and result.get("node_type") == "PLAN":
            return AtomizerDecision(
                node_type=NodeType.PLAN,
                rationale=result.get("rationale", "task needs decomposition"),
            )

        granted = result.get("granted_tools", []) if result else []
        if isinstance(granted, str):
            try:
                granted = json.loads(granted)
            except (json.JSONDecodeError, TypeError):
                granted = ["calculator", "web_search", "code_sandbox"]
        if not isinstance(granted, list):
            granted = ["calculator", "web_search", "code_sandbox"]

        return AtomizerDecision(
            node_type=NodeType.EXECUTE,
            rationale=result.get("rationale", "single pass sufficient") if result else "fallback to execute",
            granted_tools=granted,
        )
