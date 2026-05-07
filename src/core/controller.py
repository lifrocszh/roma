from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import time
from typing import Any

from src.core.graph import TaskGraph
from src.core.models import ExecutionTrace, NodeType, Task
from src.core.registry import ComponentRegistry
from src.core.signatures import AtomizerDecision, ExecutorOutput


class ControllerError(RuntimeError):
    pass


class RecursionGuardError(ControllerError):
    pass


class PlannerValidationError(ControllerError):
    pass


class TaskExecutionError(ControllerError):
    def __init__(self, task_id: str, message: str) -> None:
        super().__init__(f"task {task_id}: {message}")
        self.task_id = task_id


@dataclass(slots=True)
class SolveOutcome:
    task: Task
    output: ExecutorOutput
    trace: ExecutionTrace


@dataclass(slots=True)
class _SolveState:
    total_tasks_seen: int = 0
    expansion_counts: dict[tuple[str, str], int] = field(default_factory=dict)


class RomaController:
    def __init__(
        self,
        registry: ComponentRegistry,
        *,
        logger: logging.Logger | None = None,
        event_callback: Any | None = None,
    ) -> None:
        self.registry = registry
        self.logger = logger or logging.getLogger("roma.controller")
        self.event_callback = event_callback
        self._seed_atomizer()

    def _seed_atomizer(self) -> None:
        set_defs = getattr(self.registry.atomizer, "set_tool_definitions", None)
        if callable(set_defs):
            set_defs(self.registry.get_tool_definitions())

    def solve(self, task: Task) -> SolveOutcome:
        self.registry.validate()
        return self._solve(task=task, depth=0, state=_SolveState())

    def _solve(self, *, task: Task, depth: int, state: _SolveState) -> SolveOutcome:
        if depth > self.registry.limits.max_recursion_depth:
            raise RecursionGuardError(f"max depth {self.registry.limits.max_recursion_depth} exceeded")
        state.total_tasks_seen += 1
        if state.total_tasks_seen > self.registry.limits.max_total_tasks:
            raise RecursionGuardError(f"max total tasks {self.registry.limits.max_total_tasks} exceeded")

        trace = ExecutionTrace(
            task_id=task.id,
            goal=task.goal,
            # task_type=task.task_type,
            parent_task_id=task.parent_id,
            input_context=task.context_input,
        )
        self._emit("task_started", {"depth": depth, "goal": task.goal[:100]}, trace)

        try:
            task_with_depth = task.model_copy(update={"metadata": {**task.metadata, "_depth": depth}})
            decision = self.registry.atomizer.decide(task_with_depth)
            trace.node_type = decision.node_type
            self._emit("atomizer_decision", {
                "node_type": decision.node_type.value,
                "rationale": decision.rationale,
                "granted_tools": decision.granted_tools,
            }, trace)

            if decision.node_type == NodeType.EXECUTE:
                return self._execute(task, trace, depth, decision)
            return self._plan_and_solve(task, trace, decision, depth, state)
        except (RecursionGuardError, PlannerValidationError, ControllerError):
            raise
        except Exception as exc:
            self._emit("task_failed", {"error": str(exc)}, trace)
            raise TaskExecutionError(task.id, str(exc)) from exc

    def _execute(self, task: Task, trace: ExecutionTrace, depth: int, decision: AtomizerDecision) -> SolveOutcome:
        granted = {name: self.registry._tools[name] for name in decision.granted_tools if name in self.registry._tools}
        set_tools = getattr(self.registry.executor, "set_tools", None)
        if callable(set_tools):
            set_tools(granted)

        executor = self.registry.executor
        started = time.perf_counter()
        output = executor.execute(task)
        duration_ms = (time.perf_counter() - started) * 1000
        updated = task.model_copy(update={"result": output.result})
        trace.output_summary = output.result
        self._emit("executor_completed", {"duration_ms": duration_ms, "result_preview": output.result[:200], "granted_tools": decision.granted_tools}, trace)
        return SolveOutcome(task=updated, output=output, trace=trace)

    def _plan_and_solve(
        self, task: Task, trace: ExecutionTrace, decision: AtomizerDecision,
        depth: int, state: _SolveState,
    ) -> SolveOutcome:
        plan = self.registry.planner.plan(task)
        self._validate_plan(task, plan, decision, state)

        batches = plan.task_graph.dependency_batches(external_ids={task.id})
        self._emit("planner_output", {
            "subtask_count": len(plan.subtasks),
            "dependency_batches": [[s.id for s in b] for b in batches],
        }, trace)

        outcomes = self._solve_subtasks(task, trace, plan.task_graph, depth, state)

        ordered = [outcomes[s.id].output for s in plan.task_graph.topological_order(external_ids={task.id})]
        aggregation = self.registry.aggregator.aggregate(task, ordered)
        trace.output_summary = aggregation.summary
        self._emit("aggregation_completed", {"child_count": len(ordered), "summary_preview": aggregation.summary[:200]}, trace)

        updated = task.model_copy(update={"result": aggregation.summary})
        output = ExecutorOutput(task_id=aggregation.task_id, result=aggregation.summary, metadata=aggregation.metadata)
        return SolveOutcome(task=updated, output=output, trace=trace)

    def _solve_subtasks(
        self, task: Task, trace: ExecutionTrace, task_graph: TaskGraph,
        depth: int, state: _SolveState,
    ) -> dict[str, SolveOutcome]:
        outcomes: dict[str, SolveOutcome] = {}
        batches = task_graph.dependency_batches(external_ids={task.id})
        parallelism = max(1, min(self.registry.limits.max_parallelism, max((len(b) for b in batches), default=1)))

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for batch in batches:
                self._emit("batch_started", {"task_ids": [s.id for s in batch]}, trace)

                prepared = []
                for subtask in batch:
                    deps = [f"{d}: {outcomes[d].output.result}" for d in subtask.dependencies if d in outcomes]
                    if deps:
                        ctx = "\n".join(deps)
                        existing = subtask.context_input or ""
                        subtask = subtask.model_copy(update={"context_input": f"{existing}\n\nPrior results:\n{ctx}" if existing else f"Prior results:\n{ctx}"})
                    prepared.append(subtask)

                futures = {s.id: pool.submit(self._solve, task=s, depth=depth + 1, state=state) for s in prepared}
                for subtask in sorted(prepared, key=lambda x: x.id):
                    self._emit("child_started", {"task_id": subtask.id, "goal": subtask.goal[:120]}, trace)
                    outcome = futures[subtask.id].result()
                    outcomes[subtask.id] = outcome
                    trace.append_child(outcome.trace)
                    self._emit("child_completed", {"task_id": subtask.id, "result_preview": outcome.output.result[:120]}, trace)
        return outcomes

    def _validate_plan(self, task: Task, plan: Any, decision: AtomizerDecision, state: _SolveState) -> None:
        if not plan.subtasks:
            raise PlannerValidationError("planner returned no subtasks")
        if len(plan.subtasks) > self.registry.limits.max_subtasks_per_plan:
            raise RecursionGuardError(f"max subtasks per plan exceeded: {len(plan.subtasks)}")

        plan.task_graph.validate(external_ids={task.id})
        for st in plan.subtasks:
            if st.parent_id != task.id:
                raise PlannerValidationError(f"subtask {st.id} parent_id mismatch: got {st.parent_id!r}, expected {task.id!r}")

        sig = task.goal.strip().lower()
        state.expansion_counts[sig] = state.expansion_counts.get(sig, 0) + 1
        if state.expansion_counts[sig] > self.registry.limits.max_expansions_per_goal:
            raise RecursionGuardError(f"repeated expansion limit for goal: {task.goal[:80]}")

    def _emit(self, kind: str, payload: dict[str, Any], trace: ExecutionTrace | None = None) -> None:
        if self.event_callback:
            try:
                self.event_callback(kind, payload, trace)
            except Exception:
                pass
