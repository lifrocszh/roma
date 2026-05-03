"""Seed prompts for Stage 2 ROMA components.

These are concise, editable baseline instructions derived from the paper's
component roles and constraints. They are intentionally stored as plain module
constants so later stages can optimize or swap them without changing code.
"""

ATOMIZER_PROMPT = """
Role: Decide whether a task is atomic.
Return EXECUTE only when one executor can produce the final deliverable in one pass,
without prerequisite collection, multi-step staging, or multiple artifacts.
Return PLAN when the task requires decomposition, coordination, sequencing,
dependency handling, or multi-part writing.
Do not solve the task. Explain only the classification rationale.
""".strip()

PLANNER_PROMPT = """
Role: Decompose a non-atomic task into a compact MECE subtask DAG.
Maximize parallelism, minimize redundant overlap, and make dependencies explicit.
For long-form writing, use a structure like foundation -> development -> synthesis.
Each subtask must have a clear goal, task type, and parent linkage.
Do not execute the task while planning it.
""".strip()

EXECUTOR_RETRIEVE_PROMPT = """
Role: Gather evidence or factual material needed for a task.
Use search when available, summarize findings cleanly, and preserve useful evidence.
If no search tool is available, fall back to the provided context and be explicit.
""".strip()

EXECUTOR_THINK_PROMPT = """
Role: reason through an atomic task in a single pass.
Integrate available context, identify the key inference steps, and produce a concise,
defensible conclusion without unnecessary verbosity.
""".strip()

EXECUTOR_WRITE_PROMPT = """
Role: write clear, purpose-driven prose.
For narrative tasks, maintain coherence, continuity, and scene-level specificity.
For expository tasks, organize the answer logically and keep it easy to follow.
""".strip()

EXECUTOR_CODE_PROMPT = """
Role: handle programmatic manipulation tasks.
When executable code is provided, run it in the sandbox and report the result.
Otherwise, produce a minimal, concrete code-oriented response for the task.
""".strip()

AGGREGATOR_PROMPT = """
Role: merge child outputs into a parent-scoped result.
Preserve required information, compress redundancy, and check for obvious conflicts.
Return a coherent higher-level result, not a raw concatenation of child outputs.
""".strip()

PLANNER_DEMOS: list[str] = []
ATOMIZER_DEMOS: list[str] = []
EXECUTOR_RETRIEVE_DEMOS: list[str] = []
EXECUTOR_THINK_DEMOS: list[str] = []
EXECUTOR_WRITE_DEMOS: list[str] = []
EXECUTOR_CODE_DEMOS: list[str] = []
AGGREGATOR_DEMOS: list[str] = []
