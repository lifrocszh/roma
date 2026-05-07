# roma

ROMA is a recursive multi-agent framework. Given a natural-language task, it decomposes it into subtasks, executes them (with LLM reasoning and optional tool calls), and synthesizes the results — all driven by an LLM backend.

```
User Task → Atomizer → (PLAN or EXECUTE?)
                         │
                   ┌─────┴─────┐
                   │           │
                EXECUTE      PLAN
                   │           │
               Executor     Planner
                   │           │
               Result ──→ Subtask DAG
                               │
                    ┌──────────┼──────────┐
                    │       (parallel)     │
                Executor   Executor   Executor
                    │          │          │
                    └──────────┼──────────┘
                               │
                           Aggregator
                               │
                           Final Result
```

## Setup

### Requirements

- Python 3.12+
- `uv`

### Install

```bash
uv sync
```

### API Keys

The runtime reads provider keys from environment variables or a TOML file. Copy the template and fill in your keys:

```bash
cp config/api_keys.toml.example config/api_keys.toml
```

Supported providers (checked in this order):

| Env variable                     | TOML key             | Used for                          |
| -------------------------------- | -------------------- | --------------------------------- |
| `ROMA_API_KEY` + `ROMA_BASE_URL` | —                    | Custom OpenAI-compatible endpoint |
| `OPENROUTER_API_KEY`             | `openrouter_api_key` | OpenRouter                        |
| `DEEPSEEK_API_KEY`               | `deepseek_api_key`   | DeepSeek                          |
| `OPENAI_API_KEY`                 | `openai_api_key`     | OpenAI                            |
| `TAVILY_API_KEY`                 | `tavily_api_key`     | Web search tool                   |

The default model is `deepseek-v4-flash`. Override with `ROMA_MODEL`.

## Quick Start

```bash
# Show help
uv run roma --help

# Run a task
uv run roma run "What is the available energy for the alpha decay of Po-210?"

# With a config file
uv run roma run "Explain how ROMA works" --config config/roma.yaml

# Quiet mode (only the final answer)
uv run roma run "What is 2+2?" --quiet
```

## CLI

The only command is `roma run`.

```
uv run roma run <task> [--config PATH] [--context TEXT] [--api-keys PATH] [--quiet]
```

| Argument            | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `task` (positional) | The natural-language task                             |
| `--config`          | Path to YAML config file                              |
| `--context`         | Optional supporting context for the root task         |
| `--api-keys`        | Override the default API keys TOML path               |
| `--quiet`           | Suppress streaming output, show only the final result |

## Components

### Atomizer (`src/components/atomizer.py`)

Decides whether a task should be decomposed (`PLAN`) or executed directly (`EXECUTE`). Uses the LLM with the full list of available tools so it can also decide which tools the executor will need. Biased toward planning.

Depth limit: tasks nested deeper than 2 levels always execute directly.

### Planner (`src/components/planner.py`)

Decomposes a task into a DAG of subtasks. The LLM produces a JSON array of subtasks with `id`, `goal`, and `dependencies`. The planner validates the graph (cycle detection, orphan checks) and returns a topologically sorted execution plan.

### Executor (`src/components/executors.py`)

The single unified executor. Gets a set of **granted tools** from the atomizer, passes them to the LLM as OpenAI function definitions, and handles the tool-calling loop:

1. Sends the task + tool definitions to the LLM
2. If the LLM calls a tool, executes it and feeds the result back
3. Repeats up to 5 rounds until the LLM produces a final answer

### Aggregator (`src/components/aggregator.py`)

Synthesizes multiple child outputs into a single coherent response using the LLM. Skips the LLM when there are 0 or 1 child outputs.

### Controller (`src/core/controller.py`)

Recursive orchestration engine. Manages recursion depth, total task count, repeated expansion guards, and parallel subtask execution via `ThreadPoolExecutor`. Each execution receives only the tools granted by the atomizer.

## Tools

All tools follow a common `BaseTool` interface with an OpenAI-compatible `tool_spec()` method.

| Tool           | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `calculator`   | Safe AST-based arithmetic evaluator. Supports math functions (sqrt, sin, log, etc.) |
| `web_search`   | Tavily-backed web search and URL content extraction                                 |
| `code_sandbox` | Executes Python or shell code in a local subprocess                                 |

Tools are registered in the `ComponentRegistry` and exposed to the LLM via the executor's function-calling loop.

## Configuration

Config is YAML. Minimal example:

```yaml
runtime:
  limits:
    max_recursion_depth: 20
    max_subtasks_per_plan: 12
    max_total_tasks: 256
    max_expansions_per_goal: 3
    max_parallelism: 4
  api_keys_path: "config/api_keys.toml"
```

### Runtime limits

| Field                     | Default | Description                                  |
| ------------------------- | ------- | -------------------------------------------- |
| `max_recursion_depth`     | 20      | Max depth before the controller aborts       |
| `max_subtasks_per_plan`   | 12      | Hard cap on subtasks from one planning step  |
| `max_total_tasks`         | 256     | Total node cap for one solve run             |
| `max_expansions_per_goal` | 3       | Max times the same goal text can be expanded |
| `max_parallelism`         | 4       | Max parallel threads for dependency batches  |

## Project Structure

```
src/
├── cli/
│   ├── main.py          # typer CLI entry point
│   └── run.py           # run command logic, env setup, streaming
├── components/
│   ├── aggregator.py    # DefaultAggregator
│   ├── atomizer.py      # DefaultAtomizer
│   ├── executors.py     # UnifiedExecutor
│   ├── planner.py       # DefaultPlanner
│   └── __init__.py      # build_default_registry
├── config/
│   ├── loader.py        # YAML config loader
│   └── schema.py        # Pydantic config models
├── core/
│   ├── controller.py    # RomaController — recursive solve engine
│   ├── graph.py         # TaskGraph — DAG validation and traversal
│   ├── inference.py     # OpenAI client factory
│   ├── models.py        # Task, ExecutionTrace, NodeType
│   ├── registry.py      # ComponentRegistry, tools, RuntimeLimits
│   └── signatures.py    # AtomizerDecision, PlannerOutput, etc.
└── prompts/
    └── seed_prompts.py  # Default prompts for each component
```

## Evaluation

```bash
# Run MMLU-Pro evaluation
uv run python eval_mmlu.py

# Run with a limit and save results
uv run python eval_mmlu.py --limit 100 --output mmlu_results.xlsx

# Use direct LLM call instead of ROMA pipeline
uv run python eval_mmlu.py --direct

# Override the model
uv run python eval_mmlu.py --model gpt-4o
```

The eval script produces an Excel file with a Summary sheet (accuracy per category) and a Details sheet (every question with predictions, raw output, and ROMA intermediate steps).
