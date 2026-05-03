# ROMA Reimplementation Plan

Overview

ROMA is a **Recursive Open Meta-Agent Framework** with four core roles - **Atomizer**, **Planner**, **Executor**, and **Aggregator** - plus a **GEPA+** prompt optimizer.

This plan defines the implementation in six stages. Each stage is written to be internally coherent, dependency-safe, and testable before moving to the next stage.

---

## Stage 0: Project Foundations and Shared Contracts

**Goal**: Establish the lowest-level data contracts and invariants that every later phase depends on.

**Why this stage exists**: ROMA is recursive and multi-role by design, so the system needs stable types, explicit task metadata, and a common artifact model before any controller or prompt logic can be built.

### Functional Requirements

- Define a canonical `Task` model with:
  - `id`, `goal`, `task_type`, `dependencies`, `context_input`, `result`, and `artifacts`.
  - Support for parent/child lineage so recursive calls can be traced without ambiguity.
- Define a `TaskGraph` abstraction that:
  - Stores tasks as nodes in a DAG.
  - Detects cycles before execution starts.
  - Provides topological ordering and dependency-aware batching.
- Define a stable `NodeType` enum for recursive execution routing:
  - `PLAN` for decomposition nodes.
  - `EXECUTE` for direct execution nodes.
- Define an `ExecutionTrace` structure that:
  - Mirrors the recursion tree.
  - Records inputs, decisions, outputs, and child traces.
  - Can be serialized for debugging and evaluation.
- Define an `ArtifactStore` abstraction that:
  - Stores intermediate outputs by key instead of embedding them in prompts.
  - Supports in-memory operation first, with optional file-backed persistence later.
  - Exposes read/write/list/delete operations with deterministic keys.
- Define signatures for Atomizer, Planner, Executor, and Aggregator.
- Keep all contracts versioned and stable so later prompt or optimizer changes do not force structural rewrites.

### Runtime Requirements

- The data layer must fail fast on invalid task graphs, missing task fields, duplicate identifiers, and dependency references to unknown nodes.
- The artifact layer must return stable handles that can be passed through the controller without copying large payloads into memory repeatedly.
- Trace records must be append-only during a run so the execution history remains auditable.

### Acceptance Criteria

- A task can be instantiated, stored in a graph, validated, and serialized without running any LLM logic.
- Invalid graphs are rejected before execution.
- The same task can be referenced through the trace and artifact store without losing identity.

### Files

- `./src/core/models.py`
- `./src/core/graph.py`
- `./src/core/artifact_store.py`
- `./src/core/signatures.py`

---

## Stage 1: Recursive Controller and Tool Runtime

**Goal**: Implement the algorithmic control loop that decides whether to atomize, plan, execute, or aggregate a task.

**Why this stage exists**: ROMA's paper centers on recursive decomposition. The controller is the core behavior of the system, so it must exist before specialized prompts or optimization can be made meaningful.

### Functional Requirements

- Implement the recursive `solve(task)` procedure from the paper:
  - Call the Atomizer first.
  - If the task is atomic, route to the correct Executor.
  - If the task is non-atomic, route to the Planner, who will spawn subtasks.
  - Recursively solve each subtask.
  - Aggregate child outputs into the final response.
- Enforce recursion safety:
  - Maximum recursion depth.
  - Maximum subtasks per planning step.
  - Protection against repeated expansion of the same task.
  - Guard rails for runaway planning loops.
- Support dependency-aware scheduling:
  - Do not execute a subtask until its predecessors are complete.
  - Allow independent subtasks to run in parallel.
  - Preserve deterministic result ordering where required by aggregation.
- Define a `BaseTool` interface and concrete tool integrations:
  - `WebSearchToolkit` with tavily
  - `FileToolkit` (do not implement)
  - `MCPToolkit` (do not implement)
  - `CodeSandbox`
- Ensure each executor only receives the toolkits it is allowed to use.

### Runtime Requirements

- The controller must emit verbose execution logs while running:
  - Current task id and depth.
  - Atomizer decision and rationale.
  - Planner output size and dependency structure.
  - Executor chosen for the task type.
  - Aggregation inputs and output summary.
  - Recursion stop conditions when guards trigger.
- Parallel execution must be observable in the trace so that independent branches can be debugged after the run.
- Tool calls must be logged with input/output summaries, timing, and failure reasons.
- Failure in one branch must not silently corrupt sibling branches; it should either isolate the error or propagate it with clear trace context.

### Acceptance Criteria

- A single root task can be solved end to end with recursive decomposition.
- A task with multiple independent subtasks executes with parallelism when safe.
- Depth and subtask guards prevent infinite recursion or unbounded branching.

### Files

- `./src/core/controller.py`
- `./src/core/registry.py`

---

## Stage 2: Standard ROMA Components and Seed Prompts

**Goal**: Implement the four role-specific modules with sensible default prompts and concrete task behavior.

**Why this stage exists**: The controller needs actual semantics for atomization, planning, execution, and aggregation. The seed prompts are the baseline behavior before optimization.

### Functional Requirements

- Implement an Atomizer that:
  - Classifies tasks as atomic or non-atomic.
  - Chooses `PLAN` or `EXECUTE` for the next controller action.
  - Applies the paper's decision logic around deliverables, dependencies, and coordination complexity.
- Implement a Planner that:
  - Decomposes non-atomic tasks into MECE subtasks.
  - Produces explicit dependency edges between subtasks.
  - Keeps creative-writing tasks structured around foundation, development, and synthesis when appropriate.
- Implement specialized Executors:
  - `RetrieveExecutor` for search and evidence gathering.
  - `ThinkExecutor` for multi-step reasoning.
  - `WriteExecutor` for prose generation and synthesis.
  - `CodeExecutor` for code generation and sandboxed execution.
- Implement an Aggregator that:
  - Synthesizes child outputs.
  - Compresses redundant content without losing required information.
  - Verifies cross-child consistency before returning upward.
- Store the default prompts from the paper as seed instructions:
  - Make them pluggable through module configuration.
  - Keep them editable without changing controller code.

### Runtime Requirements

- Each component must produce verbose runtime diagnostics:
  - Atomizer should explain why a task is atomic or not.
  - Planner should explain the decomposition strategy and dependency graph.
  - Executors should report the strategy used, tools invoked, and evidence produced.
  - Aggregator should report what was merged, what was dropped, and what was preserved.
- Default prompts must be readable as standalone module contracts, not just hidden templates.
- The output format of each module must remain schema-conformant even when the content changes.

### Acceptance Criteria

- A task classified as atomic bypasses planning.
- A non-atomic task yields a valid subtask structure.
- Executors return outputs that the aggregator can consume without manual intervention.

### Files

- `./src/components/atomizer.py`
- `./src/components/planner.py`
- `./src/components/executors/`
- `./src/components/aggregator.py`
- `./src/prompts/seed_prompts.py`

---

## Stage 3: GEPA+ Prompt Optimization

**Goal**: Build the multi-component prompt optimizer that can improve different ROMA roles together.

**Why this stage exists**: The paper's GEPA+ component is not a standalone prompt editor; it is a structured optimization loop that must respect module interfaces, contract checks, and task-specific quality signals.

### Functional Requirements

- Implement a proposal generator that:
  - Samples multiple independent prompt candidates in parallel.
  - Supports diverse model sources and temperatures.
  - Produces delta edits rather than rewriting prompts from scratch.
- Implement a judge subsystem that:
  - Scores prompt variants against held-out traces or tasks.
  - Uses rubric-based evaluation where possible.
  - Produces both scalar scores and explanation text.
- Implement a verifier subsystem that:
  - Runs fast structural checks.
  - Validates schema conformance.
  - Validates task-specific outputs such as citations, code behavior, or graph invariants.
- Implement a merger that:
  - Decomposes proposals into atomic edits.
  - Resolves conflicts between competing edits.
  - Deduplicates near-equivalent instructions.
  - Preserves compatibility across modules.
- Implement a budget-aware optimization loop that:
  - Caps proposal count, judge calls, and verifier calls.
  - Tracks improvement per round.
  - Penalizes excessive prompt drift.

### Runtime Requirements

- Optimization runs must be verbose and auditable:
  - Log each proposal candidate and its origin.
  - Log every judge and verifier score.
  - Log which edits were merged, rejected, or deferred.
  - Log the total tokens consumed per round.
  - Log the time taken per round.
  - Log the final prompt delta in human-readable form.
- Optimization must be reproducible when the same seed, dataset, and model configuration are reused.
- A failed verifier should block promotion of a prompt candidate rather than silently lowering confidence.

### Acceptance Criteria

- The system can improve at least one module prompt using a held-out evaluation loop.
- The optimizer can report which changes were kept and why.
- The optimized prompt still satisfies the module schema and controller expectations.

### Files

- `./src/optimization/gepa_plus.py`
- `./src/optimization/proposer.py`
- `./src/optimization/judge.py`
- `./src/optimization/verifier.py`
- `./src/optimization/merger.py`

---

## Stage 4: Configuration, CLI, and Execution Packaging

**Goal**: Make ROMA configurable, runnable, and reproducible from the command line.

**Why this stage exists**: The implementation needs a consistent runtime interface so the controller, prompts, and optimizer can be executed by a user or benchmark harness without editing source code.

### Functional Requirements

- Implement a configuration schema that:
  - Mirrors the paper's agent configuration structure.
  - Supports per-role model settings.
  - Supports per-role prompts, demos, and toolkits.
  - Supports recursion and execution limits.
- Implement configuration loading from YAML and JSON.
- Implement CLI entry points:
  - `roma run "<task>"`
  - `roma optimize --config ... --dataset ...`
  - `roma serve` if an API entry point is provided later.
- Implement DSPy integration glue so modules can be swapped without rewriting the controller.
- Provide LM provider abstraction for multiple model vendors.

### Runtime Requirements

- CLI runs must print enough information to understand:
  - Which config was loaded.
  - Which models were assigned to which roles.
  - Which toolkits were enabled.
  - Which limits are active.
  - Which output artifacts were produced.
- Configuration errors must be explained clearly, including the exact invalid field or missing value.
- Runtime output must distinguish between task output, trace output, and optimizer output.

### Acceptance Criteria

- A user can run a single task from the CLI without writing Python code.
- A config file can reproduce the same role assignments and prompt settings across runs.
- Invalid config files fail fast with actionable error messages.

### Files

- `./src/config/schema.py`
- `./src/config/loader.py`
- `./src/cli/run.py`
- `./src/cli/optimize.py`
- `./src/__init__.py`
- `./src/main.py`

---

## Stage 5: Evaluation, Benchmarking, and Trace Reporting

**Goal**: Reproduce the paper's evaluation workflow and provide a harness for measuring quality, cost, and trace behavior.

**Why this stage exists**: ROMA is only useful if its recursive behavior and optimization loop can be compared on benchmarks with repeatable metrics and readable traces.

### Functional Requirements

- Implement benchmark adapters for the paper's target tasks:
  - SEAL-0
  - FRAMES
  - SimpleQA
  - EQ-Bench
  - AbGen
- Implement an evaluation runner that:
  - Executes tasks with the configured ROMA stack.
  - Sends outputs to a judge when required.
  - Captures metrics for correctness, quality, cost, and latency.
- Implement a trace reporter that:
  - Renders recursive execution structure.
  - Shows subtask creation and aggregation boundaries.
  - Shows prompt optimization changes when available.
- Implement benchmarking summaries that can compare multiple runs or ablations.

### Runtime Requirements

- Evaluation output must be verbose enough to reconstruct what happened during the run:
  - Task input.
  - Selected role path.
  - Intermediate subtask outputs.
  - Final answer.
  - Judge score or metric result.
  - Token and latency totals if available.
- Trace output should make it easy to distinguish:
  - A genuine recursion boundary.
  - A tool call.
  - A planner decomposition.
  - An aggregator synthesis step.
- Benchmark runs must preserve run metadata so results can be compared across configurations.

### Acceptance Criteria

- A benchmark task can be executed and scored without manual intervention.
- Recursive traces can be inspected after the run.
- Cost and quality metrics are available at both per-run and aggregated levels.

### Files

- `./eval/evaluation/benchmarks.py`
- `./eval/evaluation/judge.py`
- `./eval/evaluation/metrics.py`
- `./eval/evaluation/trace_reporter.py`

---

## Dependency Graph

```
Stage 0: Project Foundations and Shared Contracts
    ↓
Stage 1: Recursive Controller and Tool Runtime
    ↓
Stage 2: Standard ROMA Components and Seed Prompts
    ├──→ Stage 3: GEPA+ Prompt Optimization
    └──→ Stage 4: Configuration, CLI, and Execution Packaging
            ↓
Stage 5: Evaluation, Benchmarking, and Trace Reporting
```

Stages 3 and 4 are intentionally parallelizable after the core controller and components exist.

---

## Key Design Decisions

1. **DSPy-first**: All four ROMA modules are DSPy programs with typed I/O signatures so optimization can operate on structured contracts.
2. **Task-as-Node**: Every task and subtask is a graph node, which makes recursive execution and trace reconstruction explicit.
3. **Artifact-by-reference**: Intermediate outputs are stored by handle instead of being copied through prompts, which controls context growth.
4. **Heterogeneous by design**: Different roles may use different models, temperatures, or tool access policies.
5. **GEPA+ is modular**: Proposal generation, judgment, verification, and merging are independent layers that can be improved separately.

## Estimated Effort

| Stage | Focus                            | Complexity | Depends On |
| ----- | -------------------------------- | ---------- | ---------- |
| 0     | Foundations and shared contracts | Medium     | None       |
| 1     | Recursive controller and tools   | High       | Stage 0    |
| 2     | ROMA components and prompts      | High       | Stage 1    |
| 3     | GEPA+ optimization               | Very High  | Stage 2    |
| 4     | Config, CLI, packaging           | Medium     | Stage 1    |
| 5     | Evaluation and benchmarking      | Medium     | Stages 2-4 |

Total: approximately 30+ source files across 6 implementation stages.
