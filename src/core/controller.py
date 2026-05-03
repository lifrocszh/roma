"""Stage 1 recursive controller for ROMA."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import time
from typing import Any

from src.core.graph import TaskGraph
from src.core.models import ExecutionTrace, NodeType, Task
from src.core.registry import ComponentRegistry
from src.core.signatures import AggregatorOutput, AtomizerDecision, ExecutorOutput, PlannerOutput


class ControllerError(RuntimeError):
    """Base class for controller failures."""


class RecursionGuardError(ControllerError):
    """Raised when recursion guard rails are triggered."""


class PlannerValidationError(ControllerError):
    """Raised when planner output is incompatible with Stage 1 constraints."""


class TaskExecutionError(ControllerError):
    """Wraps errors with task-local execution context."""

    def __init__(self, task_id: str, message: str) -> None:
        super().__init__(f"task {task_id}: {message}")
        self.task_id = task_id


@dataclass(slots=True)
class SolveOutcome:
    """Result of solving a task, including its trace."""

    task: Task
    output: ExecutorOutput
    trace: ExecutionTrace


@dataclass(slots=True)
class _SolveState:
    """Mutable guard state shared across recursive calls within one solve."""

    total_tasks_seen: int = 0
    expansion_counts: dict[tuple[str, str], int] = field(default_factory=dict)


class RomaController:
    """Recursive task controller implementing the Stage 1 ROMA loop."""

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

    def solve(self, task: Task) -> SolveOutcome:
        """Solve a root task using recursive decomposition and aggregation."""
        self.registry.validate()
        state = _SolveState()
        return self._solve(task=task, depth=0, state=state)

    def _solve(self, *, task: Task, depth: int, state: _SolveState) -> SolveOutcome:
        self._check_global_guards(task=task, depth=depth, state=state)
        trace = ExecutionTrace(
            task_id=task.id,
            goal=task.goal,
            task_type=task.task_type,
            parent_task_id=task.parent_id,
            input_context=task.context_input,
        )
        self._emit_event(trace, "task_started", {"depth": depth, "task_type": task.task_type.value})
        self.logger.info("task=%s depth=%s goal=%s", task.id, depth, task.goal)

        try:
            decision = self.registry.atomizer.decide(task)
            trace.node_type = decision.node_type
            self._emit_event(
                trace,
                "atomizer_decision",
                {"node_type": decision.node_type.value, "rationale": decision.rationale},
            )
            self.logger.info(
                "task=%s depth=%s atomizer=%s rationale=%s",
                task.id,
                depth,
                decision.node_type.value,
                decision.rationale,
            )

            if decision.node_type == NodeType.EXECUTE:
                return self._execute_atomic_task(task=task, trace=trace, depth=depth)

            return self._solve_planned_task(
                task=task,
                trace=trace,
                decision=decision,
                depth=depth,
                state=state,
            )
        except RecursionGuardError as exc:
            self._emit_event(trace, "guard_triggered", {"error": str(exc)})
            raise
        except PlannerValidationError as exc:
            self._emit_event(trace, "planner_validation_failed", {"error": str(exc)})
            raise
        except ControllerError:
            raise
        except Exception as exc:
            self._emit_event(trace, "task_failed", {"error": str(exc)})
            raise TaskExecutionError(task.id, str(exc)) from exc

    def _execute_atomic_task(self, *, task: Task, trace: ExecutionTrace, depth: int) -> SolveOutcome:
        executor = self.registry.get_executor(task.task_type)
        allowed_tools = self.registry.get_executor_tool_names(task.task_type)
        self._emit_event(
            trace,
            "executor_selected",
            {
                "executor": type(executor).__name__,
                "task_type": task.task_type.value,
                "allowed_tools": allowed_tools,
            },
        )
        self.logger.info(
            "task=%s depth=%s executor=%s task_type=%s tools=%s",
            task.id,
            depth,
            type(executor).__name__,
            task.task_type.value,
            allowed_tools,
        )
        started = time.perf_counter()
        output = executor.execute(task)
        duration_ms = (time.perf_counter() - started) * 1000
        updated_task = task.model_copy(update={"result": output.result})
        trace.output_summary = output.result
        self._emit_event(
            trace,
            "executor_completed",
            {
                "duration_ms": duration_ms,
                "artifact_count": len(output.artifacts),
                "result_preview": output.result[:200],
            },
        )
        return SolveOutcome(task=updated_task, output=output, trace=trace)

    def _solve_planned_task(
        self,
        *,
        task: Task,
        trace: ExecutionTrace,
        decision: AtomizerDecision,
        depth: int,
        state: _SolveState,
    ) -> SolveOutcome:
        planner_output = self.registry.planner.plan(task)
        self._validate_plan(task=task, planner_output=planner_output, decision=decision, state=state)

        task_graph = planner_output.task_graph
        batches = task_graph.dependency_batches(external_ids={task.id})
        self._emit_event(
            trace,
            "planner_output",
            {
                "subtask_count": len(planner_output.subtasks),
                "dependency_batches": [[subtask.id for subtask in batch] for batch in batches],
                "rationale": planner_output.rationale,
            },
        )
        self.logger.info(
            "task=%s depth=%s planned_subtasks=%s batches=%s",
            task.id,
            depth,
            len(planner_output.subtasks),
            [[subtask.id for subtask in batch] for batch in batches],
        )

        child_outcomes = self._solve_subtasks(
            task=task,
            trace=trace,
            task_graph=task_graph,
            depth=depth,
            state=state,
        )
        ordered_child_outputs = [
            child_outcomes[subtask.id].output
            for subtask in task_graph.topological_order(external_ids={task.id})
        ]
        aggregation = self.registry.aggregator.aggregate(task, ordered_child_outputs)
        trace.output_summary = aggregation.summary
        self._emit_event(
            trace,
            "aggregation_completed",
            {
                "child_count": len(ordered_child_outputs),
                "artifact_count": len(aggregation.artifacts),
                "summary_preview": aggregation.summary[:200],
            },
        )
        self.logger.info(
            "task=%s depth=%s aggregated_children=%s summary=%s",
            task.id,
            depth,
            len(ordered_child_outputs),
            aggregation.summary[:120],
        )
        updated_task = task.model_copy(update={"result": aggregation.summary})
        output = ExecutorOutput(
            task_id=aggregation.task_id,
            result=aggregation.summary,
            artifacts=aggregation.artifacts,
            metadata=aggregation.metadata,
        )
        return SolveOutcome(task=updated_task, output=output, trace=trace)

    def _solve_subtasks(
        self,
        *,
        task: Task,
        trace: ExecutionTrace,
        task_graph: TaskGraph,
        depth: int,
        state: _SolveState,
    ) -> dict[str, SolveOutcome]:
        outcomes: dict[str, SolveOutcome] = {}
        batches = task_graph.dependency_batches(external_ids={task.id})
        parallelism = max(1, min(self.registry.limits.max_parallelism, max((len(batch) for batch in batches), default=1)))

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for batch_index, batch in enumerate(batches):
                self._emit_event(
                    trace,
                    "batch_started",
                    {
                        "batch_index": batch_index,
                        "task_ids": [subtask.id for subtask in batch],
                        "parallel_size": len(batch),
                    },
                )
                self.logger.info(
                    "task=%s depth=%s batch=%s task_ids=%s",
                    task.id,
                    depth,
                    batch_index,
                    [subtask.id for subtask in batch],
                )
                futures = {
                    subtask.id: pool.submit(self._solve, task=subtask, depth=depth + 1, state=state)
                    for subtask in batch
                }
                for subtask in sorted(batch, key=lambda item: item.id):
                    self._emit_event(
                        trace,
                        "child_started",
                        {
                            "batch_index": batch_index,
                            "task_id": subtask.id,
                            "goal": subtask.goal,
                            "task_type": subtask.task_type.value,
                        },
                    )
                    outcome = futures[subtask.id].result()
                    outcomes[subtask.id] = outcome
                    trace.append_child(outcome.trace)
                    self._emit_event(
                        trace,
                        "child_completed",
                        {"batch_index": batch_index, "task_id": subtask.id, "result_preview": outcome.output.result[:120]},
                    )
        return outcomes

    def _emit_event(self, trace: ExecutionTrace, kind: str, payload: dict[str, Any]) -> None:
        trace.append_event(kind, payload)
        if self.event_callback is not None:
            try:
                self.event_callback(kind, payload, trace)
            except Exception:
                pass

    def _check_global_guards(self, *, task: Task, depth: int, state: _SolveState) -> None:
        limits = self.registry.limits
        if depth > limits.max_recursion_depth:
            self.logger.warning("task=%s guard=max_recursion_depth depth=%s", task.id, depth)
            raise RecursionGuardError(
                f"maximum recursion depth exceeded for task {task.id}: {depth} > {limits.max_recursion_depth}"
            )
        state.total_tasks_seen += 1
        if state.total_tasks_seen > limits.max_total_tasks:
            self.logger.warning("task=%s guard=max_total_tasks total=%s", task.id, state.total_tasks_seen)
            raise RecursionGuardError(
                f"maximum total tasks exceeded: {state.total_tasks_seen} > {limits.max_total_tasks}"
            )

    def _validate_plan(
        self,
        *,
        task: Task,
        planner_output: PlannerOutput,
        decision: AtomizerDecision,
        state: _SolveState,
    ) -> None:
        if decision.node_type != NodeType.PLAN:
            raise PlannerValidationError(f"task {task.id} received planner output after EXECUTE decision")
        if not planner_output.subtasks:
            raise PlannerValidationError(f"task {task.id} planner returned no subtasks")
        if len(planner_output.subtasks) > self.registry.limits.max_subtasks_per_plan:
            raise RecursionGuardError(
                f"task {task.id} exceeded max subtasks per plan: "
                f"{len(planner_output.subtasks)} > {self.registry.limits.max_subtasks_per_plan}"
            )

        planner_output.task_graph.validate(external_ids={task.id})
        graph_ids = set(planner_output.task_graph.tasks)
        listed_ids = {subtask.id for subtask in planner_output.subtasks}
        if graph_ids != listed_ids:
            raise PlannerValidationError(
                f"task {task.id} planner graph ids do not match returned subtasks: {graph_ids} != {listed_ids}"
            )

        for subtask in planner_output.subtasks:
            if subtask.parent_id != task.id:
                raise PlannerValidationError(
                    f"subtask {subtask.id} must reference parent_id {task.id}, got {subtask.parent_id!r}"
                )

        signature = (task.goal.strip().lower(), task.task_type.value)
        state.expansion_counts[signature] = state.expansion_counts.get(signature, 0) + 1
        if state.expansion_counts[signature] > self.registry.limits.max_expansions_per_goal:
            raise RecursionGuardError(
                f"task {task.id} exceeded repeated expansion limit for goal {task.goal!r}: "
                f"{state.expansion_counts[signature]} > {self.registry.limits.max_expansions_per_goal}"
            )
