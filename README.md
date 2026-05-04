# roma

ROMA is a recursive multi-agent framework with a CLI front end. The current codebase implements:

- Ccore contracts and data structures
- Recursive controller and tool runtime
- Default atomizer, planner, executors, and aggregator
- GEPA+ prompt optimization
- Configuration, CLI, and execution packaging

This README documents the code as it exists now, not the full paper target. Stage 5 evaluation and benchmark reproduction are not implemented yet.

## Setup

### Requirements

- Python 3.12+
- `uv`
- Optional: `config/api_keys.toml` if you want live model output or search

### Install

From the project root:

```bash
uv sync
```

That creates the environment and installs the package in editable mode.

### API Keys

The runtime reads provider keys from a TOML file. The example template is [config/api_keys.toml.example](C:/Study/Projects/roma/config/api_keys.toml.example).

Current expected shape:

```toml
[providers]
deepseek_api_key = ""
tavily_api_key = ""

[runtime]
default_search_provider = "tavily"
```

Current meaning:

- `deepseek_api_key`: used for model calls through DeepSeek's OpenAI-compatible API
- `tavily_api_key`: used by the web search toolkit
- `default_search_provider`: currently informational; the implemented search runtime is Tavily

Current model runtime defaults:

- base URL: `https://api.deepseek.com`
- model: `deepseek-v4-flash`

If the model call fails or no model key is present, executors fall back to deterministic placeholder behavior.

## Quick Start

Show help:

```bash
uv run roma --help
```

Run a natural-language task:

```bash
uv run roma run "Explain how ROMA works"
```

Run with extra context:

```bash
uv run roma run "Write a short project update" --context "Status: tests pass. Risk: deployment timing."
```

Run the demo workflow:

```bash
uv run roma run-demo
```

Run prompt optimization:

```bash
uv run roma optimize
```

Run with a config file:

```bash
uv run roma run "Explain the current architecture" --config config/roma.yaml
uv run roma optimize --config config/roma.yaml
```

## Commands

The package exposes a `roma` command with these implemented subcommands:

- `stage0-status`
- `run`
- `run-demo`
- `optimize`

### `roma stage0-status`

Prints the currently implemented architecture stages.

Example:

```bash
uv run roma stage0-status
```

### `roma run`

Primary user-facing execution path. This command accepts a normal natural-language task and lets the controller route it internally.

Basic form:

```bash
uv run roma run "Explain how ROMA works"
```

Current arguments:

- positional `task`
  meaning: the natural-language request
  example:

  ```bash
  uv run roma run "Summarize the architecture"
  ```

- `--config PATH`
  meaning: load YAML or JSON config before running
  example:

  ```bash
  uv run roma run "Write a compact update" --config config/roma.yaml
  ```

- `--context TEXT`
  meaning: attach supporting context to the root task
  example:

  ```bash
  uv run roma run "Write a compact update" --context "Status: tests pass."
  ```

- `--api-keys PATH`
  meaning: override the default TOML API-key file path
  example:

  ```bash
  uv run roma run "Explain how ROMA works" --api-keys config/api_keys.toml
  ```

What `roma run` does:

1. Loads config if provided.
2. Loads API keys from TOML.
3. Sets runtime model environment variables if a model key is present.
4. Creates a root `Task` with:
   - `goal=<your task>`
   - `task_type=GENERAL`
   - `context_input=<optional context>`
5. Builds the default registry.
6. Runs the recursive controller.
7. Streams subtask events as they happen.
8. Prints the final result at the end.

### `roma run-demo`

Convenience entrypoint for exercising the current Stage 2 workflow with a writing-oriented default task.

Basic form:

```bash
uv run roma run-demo
```

Current arguments:

- positional `task`
  meaning: override the default demo task

- `--task-type`
  meaning: explicit task type for the demo path
  accepted values:
  - `GENERAL`
  - `RETRIEVE`
  - `THINK`
  - `WRITE`
  - `CODE`

- `--context`
  meaning: optional context passed into the root task

- `--api-keys`
  meaning: path to the TOML API-key file

Example:

```bash
uv run roma run-demo "Write a multi-part project update then refine the final message" --context "Status: deployment is pending."
```

### `roma optimize`

Runs the deterministic Stage 3 GEPA+ prompt optimizer.

Basic form:

```bash
uv run roma optimize
```

With config:

```bash
uv run roma optimize --config config/roma.yaml
```

Current arguments:

- `--config PATH`
  meaning: path to a YAML or JSON config file that defines the optimization settings

What `roma optimize` does:

1. Loads config.
2. Reads the `optimize` block.
3. Resolves the base prompt from config or built-in seed prompts.
4. Generates multiple prompt candidates.
5. Scores them with the judge.
6. Verifies them with structural checks.
7. Merges the best verified candidates.
8. Prints the optimized prompt and optimization summary.

## Rough Workflow

At a high level, the runtime flow is:

1. The CLI receives a natural-language task.
2. A root `Task` object is created.
3. The controller asks the Atomizer whether the task is atomic.
4. If atomic:
   - route to one executor
   - produce a result
5. If non-atomic:
   - route to the Planner
   - produce a dependency-aware subtask DAG
   - recursively solve each subtask
   - aggregate child outputs into a parent result
6. Return the final answer.

Data flow direction:

- top-down during decomposition
- left-to-right across dependency batches
- bottom-up during aggregation

Current CLI behavior:

- subtask and execution events are streamed in real time
- final answer is printed after the recursive solve completes

## Agents and Executors

ROMA currently has four role classes plus type-specialized executors.

### Atomizer

File: [src/components/atomizer.py](C:/Study/Projects/roma/src/components/atomizer.py)

What it does:

- decides whether a task should be `EXECUTE` or `PLAN`
- looks for signals like sequencing, multi-part writing, dependencies, and staged work
- can respect planner-marked leaf tasks that should bypass replanning

In practice:

- simple direct tasks usually become `EXECUTE`
- complex multi-step tasks usually become `PLAN`

### Planner

File: [src/components/planner.py](C:/Study/Projects/roma/src/components/planner.py)

What it does:

- decomposes non-atomic tasks into a compact subtask DAG
- sets parent-child lineage
- adds dependency edges between subtasks
- uses task-type-specific decomposition templates

Current planner behavior by type:

- `WRITE`: foundation -> development -> synthesis
- `RETRIEVE`: query framing -> evidence gathering -> summary
- `CODE`: analysis -> implementation -> verification
- general tasks: inputs -> analysis -> answer

### Aggregator

File: [src/components/aggregator.py](C:/Study/Projects/roma/src/components/aggregator.py)

What it does:

- merges child outputs into one parent-scoped result
- drops duplicate segments
- preserves unique content
- returns a compressed final summary rather than raw concatenation where possible

### Executors

Executors are leaf-node workers selected by `task_type`.

Files:

- [src/components/executors/think.py](C:/Study/Projects/roma/src/components/executors/think.py)
- [src/components/executors/write.py](C:/Study/Projects/roma/src/components/executors/write.py)
- [src/components/executors/retrieve.py](C:/Study/Projects/roma/src/components/executors/retrieve.py)
- [src/components/executors/code.py](C:/Study/Projects/roma/src/components/executors/code.py)

#### `ThinkExecutor`

Used for:

- `GENERAL`
- `THINK`

What it does:

- calls the model for reasoning when a DeepSeek key is available
- otherwise falls back to a deterministic reasoning template

#### `WriteExecutor`

Used for:

- `WRITE`

What it does:

- calls the model for prose generation when a DeepSeek key is available
- otherwise falls back to deterministic writing templates
- uses a different fallback style for narrative-like tasks

#### `RetrieveExecutor`

Used for:

- `RETRIEVE`

What it does:

- calls the web search tool if available
- optionally asks the model to synthesize the search results
- falls back to context-based evidence synthesis if no search tool is available

#### `CodeExecutor`

Used for:

- `CODE`

What it does:

- executes provided code in the sandbox if a runnable snippet exists
- otherwise falls back to a code-oriented planning response

## Tools Given To The Runtime

The Stage 1 runtime defines a base tool interface and concrete tool integrations.

### `WebSearchToolkit`

File: [src/core/registry.py](C:/Study/Projects/roma/src/core/registry.py)

What it does:

- runs Tavily search queries
- returns structured tool output

Used by:

- `RetrieveExecutor`

Requires:

- `tavily_api_key`

### `CodeSandbox`

File: [src/core/registry.py](C:/Study/Projects/roma/src/core/registry.py)

What it does:

- runs Python or shell code in a subprocess
- returns captured output and status

Used by:

- `CodeExecutor`

### `FileToolkit`

Declared but intentionally not implemented.

### `MCPToolkit`

Declared but intentionally not implemented.

## Configuration

The config loader accepts YAML or JSON. The schema is defined in [src/config/schema.py](C:/Study/Projects/roma/src/config/schema.py).

Minimal example:

```yaml
demo:
  goal: "Write a multi-part project update then refine the final message"
  task_type: "WRITE"
  context_input: "Status: tests pass. Risk: deployment timing."

optimize:
  module_name: "write"
  base_prompt: "Role: write clear, purpose-driven prose."
  held_out_signals:
    - schema
    - contract
    - minimal
  rounds: 2
  proposals_per_round: 4
  topn: 2
  seed: 21

runtime:
  api_keys_path: "config/api_keys.toml"
```

## What Each Config Section Means

### `agent`

Current role:

- typed container for per-role runtime settings
- not fully consumed by the current CLI path
- included to preserve the Stage 4 schema shape

Fields inside `agent`:

- `atomizer`
- `planner`
- `aggregator`
- `executors`
- `max_subtasks`
- `toolkits`
- `execution`

#### `agent.atomizer`

#### `agent.planner`

#### `agent.aggregator`

Each is a `RoleRuntimeConfig`.

Supported fields:

- `model`
  meaning: per-role model metadata
- `prompt`
  meaning: role-specific prompt override
- `demos`
  meaning: example/demo strings for that role
- `toolkits`
  meaning: allowed toolkit names for that role

#### `agent.executors`

Dictionary keyed by executor name such as:

- `GENERAL`
- `THINK`
- `WRITE`
- `RETRIEVE`
- `CODE`

Each value is a `RoleRuntimeConfig`.

#### `agent.max_subtasks`

Intended meaning:

- maximum subtasks a planning step should produce

Current note:

- the controller limit is actually enforced through `runtime.limits.max_subtasks_per_plan`
- this field currently exists in the schema for config structure alignment

#### `agent.toolkits`

Dictionary of toolkit configuration entries.

Each toolkit entry supports:

- `enabled`
- `config`

Example:

```yaml
agent:
  toolkits:
    web_search:
      enabled: true
      config:
        max_results: 5
```

#### `agent.execution`

Arbitrary execution-related settings bucket.

Current note:

- schema-supported
- not strongly consumed by the implemented runtime yet

### `runtime`

Controls runtime plumbing used directly by the CLI.

Fields:

- `limits`
- `api_keys_path`
- `default_search_provider`

#### `runtime.limits`

Maps to `RuntimeLimits` in [src/core/registry.py](C:/Study/Projects/roma/src/core/registry.py).

Supported fields:

- `max_recursion_depth`
  meaning: maximum recursion depth before the controller aborts

- `max_subtasks_per_plan`
  meaning: hard cap on subtasks returned by the planner for one node

- `max_total_tasks`
  meaning: total node cap for one solve run

- `max_expansions_per_goal`
  meaning: maximum times the same goal can be expanded

- `max_parallelism`
  meaning: maximum number of parallel worker threads used for dependency batches

Example:

```yaml
runtime:
  limits:
    max_recursion_depth: 8
    max_subtasks_per_plan: 12
    max_total_tasks: 256
    max_expansions_per_goal: 3
    max_parallelism: 4
```

#### `runtime.api_keys_path`

Meaning:

- path to the TOML file that stores provider keys

Example:

```yaml
runtime:
  api_keys_path: "config/api_keys.toml"
```

#### `runtime.default_search_provider`

Meaning:

- label for the search provider to use

Current implementation:

- the search runtime is Tavily
- this field is mostly informational in the current code

### `demo`

Controls the default task payload stored in config.

Fields:

- `goal`
  meaning: default natural-language task text

- `task_type`
  meaning: textual type hint stored in config

- `context_input`
  meaning: optional default supporting context

Current note:

- `roma run` now accepts a natural-language task directly and always enters the controller as `GENERAL`
- this config block remains useful as a stored task template

### `optimize`

Controls the Stage 3 optimizer.

Fields:

- `module_name`
  meaning: which module prompt to optimize
  typical values:
  - `atomizer`
  - `planner`
  - `write`
  - `retrieve`
  - `think`
  - `code`
  - `aggregator`

- `base_prompt`
  meaning: explicit prompt text to optimize
  behavior: if omitted, the CLI resolves the built-in seed prompt for `module_name`

- `held_out_signals`
  meaning: rubric-like keywords used by the judge when scoring prompt candidates

- `rounds`
  meaning: number of optimization rounds

- `proposals_per_round`
  meaning: how many candidates to generate each round

- `topn`
  meaning: how many verified high-scoring proposals to keep during merge

- `seed`
  meaning: deterministic seed for the proposal generator

- `max_tokens`
  meaning: optional budget cap that can stop optimization early

## Core Runtime Fields

These fields appear repeatedly in the code and config.

### `goal`

The human request. This is the actual thing you want the system to do.

Example:

```text
Write a compact update
```

### `task_type`

The internal task category used for routing and executor selection.

Current values:

- `GENERAL`
- `RETRIEVE`
- `THINK`
- `WRITE`
- `CODE`

Important current behavior:

- `roma run` starts with `GENERAL`
- downstream planning and execution still use typed routing internally

### `context_input`

Optional supporting context for a task.

Use it for:

- background facts
- status notes
- constraints
- prior findings
- local evidence

Example:

```text
Status: tests pass. Risk: deployment timing.
```

## How Optimization Works

The Stage 3 optimizer is implemented in:

- [src/optimization/proposer.py](C:/Study/Projects/roma/src/optimization/proposer.py)
- [src/optimization/judge.py](C:/Study/Projects/roma/src/optimization/judge.py)
- [src/optimization/verifier.py](C:/Study/Projects/roma/src/optimization/verifier.py)
- [src/optimization/merger.py](C:/Study/Projects/roma/src/optimization/merger.py)
- [src/optimization/gepa_plus.py](C:/Study/Projects/roma/src/optimization/gepa_plus.py)

At a high level, `roma optimize` is a prompt-improvement pass over one module at a time. It does not optimize the whole agent end-to-end in one shot. You point it at a target module such as `write` or `planner`, it starts from a base prompt, proposes several delta edits, scores them, verifies them, merges the best ones, and returns a revised prompt string.

### What It Is Optimizing

Current supported module names are:

- `atomizer`
- `planner`
- `write`
- `retrieve`
- `think`
- `code`
- `aggregator`

If `base_prompt` is omitted in config, the CLI resolves a built-in seed prompt from [src/prompts/seed_prompts.py](C:/Study/Projects/roma/src/prompts/seed_prompts.py).

Use cases:

- tighten the `write` prompt to make output structure more explicit
- make the `planner` prompt emphasize MECE decomposition more strongly
- make the `atomizer` prompt preserve routing contracts more clearly
- improve the `retrieve` prompt so evidence summaries are more explicit

### Actual Optimization Loop

The implemented loop is deterministic and local. It works like this:

1. Start from `base_prompt`.
2. Generate `proposals_per_round` candidates.
3. Each candidate is a delta-style edit, not a full rewrite.
4. Score every candidate with the judge.
5. Verify every candidate with structural checks.
6. Reject candidates that fail verification.
7. Merge the best verified candidates up to `topn`.
8. Use the merged prompt as the next round's base prompt.
9. Stop when rounds are exhausted, the token budget is exceeded, or no verified proposals are kept.

### What Each Optimization Field Does

These fields live under the `optimize` section in config.

#### `module_name`

The target module prompt to optimize.

Example:

```yaml
optimize:
  module_name: "write"
```

#### `base_prompt`

Explicit prompt text to optimize.

If present:

- the optimizer uses this exact string as the starting point

If omitted:

- the CLI picks the built-in seed prompt for `module_name`

Example:

```yaml
optimize:
  module_name: "write"
  base_prompt: |
    Role: write clear, purpose-driven prose.
```

#### `held_out_signals`

Keywords given to the judge as a lightweight rubric.

What they do in the current code:

- they do not trigger model-based evaluation
- they increase the score when candidate text overlaps those signals

Good examples:

- `schema`
- `contract`
- `minimal`
- `structure`
- `clarity`

Example:

```yaml
optimize:
  held_out_signals:
    - schema
    - contract
    - minimal
```

#### `rounds`

How many optimization rounds to run.

Current effect:

- `1` means one propose-score-verify-merge cycle
- higher values allow iterative refinement, but only if the merger keeps proposals

Example:

```yaml
optimize:
  rounds: 2
```

#### `proposals_per_round`

How many prompt candidates to generate in one round.

Current effect:

- higher values widen search
- they also increase scoring and verification work
- token estimation also rises with this value

Example:

```yaml
optimize:
  proposals_per_round: 4
```

#### `topn`

How many verified top-scoring candidates to keep during merge.

Current effect:

- `1` keeps only the best verified candidate
- `2` allows a slightly broader merge

Example:

```yaml
optimize:
  topn: 2
```

#### `seed`

Deterministic seed for proposal generation.

Current effect:

- same seed + same prompt + same config gives the same local optimization behavior

Example:

```yaml
optimize:
  seed: 21
```

#### `max_tokens`

Optional budget cap for the optimizer.

Current behavior:

- the optimizer estimates tokens from prompt text length
- if one round exceeds the cap, it records a note and stops

Example:

```yaml
optimize:
  max_tokens: 2000
```

### What The Proposer Does

The proposer:

- generates multiple prompt candidates
- uses deterministic seeds
- emits delta-style edits
- appends contract-preservation language

Current proposal style:

- add one explicit instruction
- keep the base prompt mostly intact
- emphasize structure, contract, minimal edits, or ambiguity reduction

For `write`, the proposer uses write-specific additions such as:

- preserving output structure
- making organization explicit
- keeping ambiguity low

### What The Judge Does

The judge is a deterministic rubric scorer.

Current scoring signals include:

- presence of contract language
- mention of minimal edits
- explicit schema language
- ambiguity-reduction language
- overlap with `held_out_signals`

The judge returns:

- a numeric score
- a short explanation string

This is not an LLM-as-a-judge implementation yet. It is a lightweight local heuristic judge.

### What The Verifier Does

The verifier blocks unsafe promotions.

Current checks:

- required markers for the target module
- explicit contract or schema preservation language

Examples:

- `planner` proposals must preserve planner-like markers such as decomposition or dependencies
- `write` proposals must preserve structure-related language
- `aggregator` proposals must keep merge/compress/conflict language

If verification fails:

- the candidate is not promoted
- the merge step cannot keep it

This is important because the current system treats verification as a hard gate, not just a soft penalty.

### What The Merger Does

The merger takes judged proposals and keeps only verified top candidates.

Current merge behavior:

- rank by score
- reject unverified proposals
- deduplicate equivalent additions
- keep at most `topn`
- append a final merge contract line

The result is:

- `merged_prompt`
- `kept`
- `rejected`
- `deferred`
- `delta_summary`

### What The Command Prints

`roma optimize` prints:

- `module=<module_name>`
- `best_score=<score>`
- `final_delta=<summary>`
- the final optimized prompt text

Example:

```bash
uv run roma optimize --config config/roma.yaml
```

Typical output shape:

```text
module=write
best_score=2.5
final_delta=+ Always preserve the write interface contract and document the output structure.
Role: write clear, purpose-driven prose.
...
```

### How To Use It Well

Practical guidance for the current implementation:

1. Start with one module, not all modules.
2. Use a short, explicit `base_prompt` if you are testing prompt mechanics.
3. Keep `held_out_signals` concrete and low-count.
4. Use `rounds=1` and `proposals_per_round=2` for quick iterations.
5. Increase `proposals_per_round` only when you want broader exploration.
6. Use a fixed `seed` when comparing runs.
7. Use `topn=1` if you want tighter, more conservative merges.
8. Use `topn=2` if you want a slightly broader merged prompt.

### Recommended Starting Configs

Fast local iteration:

```yaml
optimize:
  module_name: "write"
  rounds: 1
  proposals_per_round: 2
  topn: 1
  seed: 21
  held_out_signals:
    - structure
    - contract
```

Broader search:

```yaml
optimize:
  module_name: "planner"
  rounds: 2
  proposals_per_round: 4
  topn: 2
  seed: 21
  held_out_signals:
    - MECE
    - dependency
    - schema
```

### Current Limitations

The current optimizer is useful, but it is still a local deterministic implementation.

Current limitations:

- no token accounting from a provider
- no automatic prompt persistence back into config or source files
- no multi-module joint optimization command yet

So the right mental model is:

- but it is still an internal, local prompt-evolution scaffold rather than a full production optimizer

## Evaluation usage

```bash
# Default: 500 questions, 5 parallel
uv run roma eval-mmlu

# Custom settings
uv run roma eval-mmlu -n 100 -p 5 -o results.json
```

## Practical Examples

Run a basic task:

```bash
uv run roma run "Explain the current ROMA architecture"
```

Run a writing task with context:

```bash
uv run roma run "Write a short project update" --context "Status: tests pass. Risk: deployment timing."
```

Run the demo workflow:

```bash
uv run roma run-demo
```

Run prompt optimization:

```bash
uv run roma optimize --config config/roma.yaml
```

Run with a separate API key file:

```bash
uv run roma run "Explain how ROMA works" --api-keys config/api_keys.toml
```

## Notes

- `roma run` is the normal user-facing path.
- `roma run-demo` is still useful for a pre-shaped demo workflow.
- live LLM output depends on a valid `deepseek_api_key`
- if the model call fails, the executors fall back to deterministic placeholder logic
- live search depends on a valid `tavily_api_key`
- the repository currently stops at Stage 4
