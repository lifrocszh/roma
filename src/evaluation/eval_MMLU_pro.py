from __future__ import annotations

import asyncio
import json
import re
import random
import sys
from pathlib import Path

import datasets
import typer

app = typer.Typer(help="MMLU-Pro benchmark evaluation.")

random.seed(42)

CATEGORIES = [
    "computer science",
    "math",
    "chemistry",
    "engineering",
    "law",
    "biology",
    "health",
    "physics",
    "business",
    "philosophy",
    "economics",
    "other",
    "psychology",
    "history",
]

OPTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def form_options(options: list[str]) -> str:
    lines = ["Options are:"]
    for opt, o in zip(options, OPTS):
        lines.append(f"({o}): {opt}")
    return "\n".join(lines) + "\n"


def get_prediction(output: str) -> str:
    """Extract the answer letter from the ROMA output."""
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    print(f"  [extraction failed, random guess]")
    return random.choice(OPTS)


def extract_final_answer(full_output: str) -> str:
    sep = "=" * 60
    parts = full_output.split(sep)
    # parts[0] = traces before first sep
    # parts[1] = Result: header
    # parts[2] = final answer body (may include trailing newlines)
    # parts[3] = Subtasks line
    if len(parts) >= 4:
        return parts[2].strip()
    # Fallback: if separators not found, return the whole thing
    return full_output.strip()


def sample_balanced(
    dataset, n_total: int = 500, seed: int = 42
) -> list[dict]:
    """Sample up to *n_total* questions, distributed evenly across categories."""
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for entry in dataset:
        by_cat[entry["category"]].append(entry)

    per_cat = max(1, n_total // len(CATEGORIES))
    selected: list[dict] = []
    for cat in CATEGORIES:
        pool = by_cat[cat]
        if len(pool) <= per_cat:
            chosen = list(pool)
        else:
            chosen = rng.sample(pool, per_cat)
        selected.extend(chosen)

    if len(selected) > n_total:
        selected = rng.sample(selected, n_total)

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Run a single question through `python -c "run_command(task=stdin.read())"`
# Pipes the query via stdin to avoid Windows 8191-char command-line limit.
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Resolve the project root directory (where pyproject.toml lives)."""
    return str(Path(__file__).resolve().parent.parent.parent)


async def _run_via_roma(
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    query: str,
    *,
    timeout_sec: int = 300,
) -> str:
    """Pipe *query* via stdin into `run_command(quiet=True)` and return stdout.

    Uses a direct ``python -c`` subprocess instead of ``uv run roma run --quiet``
    to avoid the Windows 8191-character command-line length limit.
    """
    async with sem:
        print(f"  [{idx}/{total}] submitting ...", end="", flush=True)

        # Direct call to run_command via a small wrapper script.
        # The script prints the result to stdout and captures any encoding errors.
        script = (
            "import sys; sys.path.insert(0, '.'); "
            "from src.cli.run import run_command; "
            "import io; "
            "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'); "
            "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace'); "
            "run_command(task=sys.stdin.read(), quiet=False)"
        )

        env = dict(__import__('os').environ)
        env["PYTHONIOENCODING"] = "utf-8"

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=_project_root(),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=query.encode("utf-8")),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            print(f" timeout")
            return ""

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            print(f" failed (rc={proc.returncode})")
            return f"[ROMA ERROR]\n{err}"

        result = stdout.decode("utf-8", errors="replace").strip()
        print(f" done")
        return result


async def run_benchmark_async(
    *,
    num_questions: int = 500,
    parallel: int = 5,
    output: Path = Path("outputs.json"),
) -> None:
    print("Loading MMLU-Pro test split ...")
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro")

    print(f"Sampling {num_questions} questions (balanced across categories) ...")
    test_entries = sample_balanced(dataset["test"], n_total=num_questions)

    sem = asyncio.Semaphore(parallel)

    answers: list[dict] = []
    per_category_accuracy: dict[str, list[int]] = {c: [0, 0] for c in CATEGORIES}
    success = 0
    fail = 0

    n = len(test_entries)
    print(f"Answering {n} questions ({parallel} parallel) via roma run ...")
    tasks = []
    for idx, entry in enumerate(test_entries, start=1):
        query = (
            "Q: "
            + entry["question"]
            + "\n"
            + form_options(entry["options"])
            + "\n"
            + "Please answer the multi-choice question. Derive your final answer as \"The answer is ...\"."
        )
        tasks.append(_run_via_roma(sem, idx, n, query))

    results = await asyncio.gather(*tasks)

    for entry, roma_output in zip(test_entries, results):
        final_text = extract_final_answer(roma_output)
        prediction = get_prediction(final_text)
        answers.append({
            "question": entry["question"],
            "options": entry["options"],
            "correct_answer": entry["answer"],
            "chosen_answer": prediction,
            "response": final_text,
            "category": entry["category"],
            "src": entry.get("src", ""),
            "cot_content": roma_output,
        })
        correct = entry["answer"] == prediction

        if correct:
            success += 1
            per_category_accuracy[entry["category"]][0] += 1
        else:
            fail += 1
            per_category_accuracy[entry["category"]][1] += 1

        total_so_far = success + fail
        acc = success / total_so_far
        status = "CORRECT" if correct else "WRONG"
        print(f"  [{total_so_far}/{n}] {status}: predicted={prediction}, expected={entry['answer']}, acc={acc:.4f}")

    # --- Build accuracy summary ---
    per_cat: dict[str, dict] = {}
    for k, v in per_category_accuracy.items():
        total = v[0] + v[1]
        if total > 0:
            per_cat[k] = {"correct": v[0], "total": total, "accuracy": round(v[0] / total, 4)}

    summary = {
        "overall_accuracy": round(success / (success + fail), 4),
        "correct": success,
        "total": success + fail,
        "per_category": per_cat,
    }

    print(f"\n{'='*55}")
    print(f"  FINAL OVERALL ACCURACY: {summary['overall_accuracy']}  ({success}/{success + fail})")
    print(f"{'='*55}")

    for k, v in per_cat.items():
        print(f"    {k:25s}: {v['accuracy']:.4f} ({v['correct']}/{v['total']})")

    # Write results with summary attached
    output_data = {"summary": summary, "results": answers}
    print(f"\nWriting detailed results to {output} ...")
    output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.command()
def eval_mmlu(
    num_questions: int = typer.Option(500, "--num-questions", "-n", help="Number of questions to sample."),
    parallel: int = typer.Option(5, "--parallel", "-p", help="Number of parallel roma run calls."),
    output: Path = typer.Option(Path("outputs.json"), "--output", "-o", help="Output JSON file path."),
) -> None:
    """Run a balanced MMLU-Pro evaluation. Each question is answered via `roma run`."""
    asyncio.run(
        run_benchmark_async(
            num_questions=num_questions,
            parallel=parallel,
            output=output,
        )
    )


if __name__ == "__main__":
    app()
