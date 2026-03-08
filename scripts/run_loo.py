"""CLI entry point for running the LLM leave-one-out evaluation on ARC-AGI tasks.

Usage:
    python scripts/run_loo.py --dry-run
    python scripts/run_loo.py --task-id abc123
    python scripts/run_loo.py --num-tasks 5
    python scripts/run_loo.py --task-dir data/ARC-AGI/data/evaluation/
"""
import argparse
import asyncio
import logging
import pathlib
import sys

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Fix Windows asyncio event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.table import Table

from src.data_loader import load_task, load_all_tasks
from src.hypothesis_agent import build_prompt, build_chain
from src.leave_one_out import run_task_loo
from src.evaluate import evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM leave-one-out on ARC-AGI tasks")
    parser.add_argument(
        "--task-dir",
        default="data/ARC-AGI/data/training/",
        help="Directory of ARC task JSON files",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Run a single task by filename stem (without .json)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Limit number of tasks to process",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory to save JSON results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt for first LOO iteration of first task; no API calls",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    return parser.parse_args()


def do_dry_run(task_dir: str, model: str) -> None:
    """Print the prompt for the first LOO iteration of the first task."""
    task_dir = pathlib.Path(task_dir)
    tasks = sorted(task_dir.glob("*.json"))
    if not tasks:
        console.print("[red]No tasks found in task-dir.[/red]")
        sys.exit(1)

    task = load_task(tasks[0])
    console.print(f"\n[bold]Dry run — task: {task.task_id}[/bold]")
    console.print(
        f"[dim]Model: {model} | Train pairs: {len(task.train)} | "
        f"Test pairs: {len(task.test)}[/dim]\n"
    )
    console.print("[bold yellow]== LOO iteration 0: holding out train[0] ==[/bold yellow]")
    demo_pairs = task.train[1:]
    test_input = task.train[0].input
    prompt = build_prompt(demo_pairs, test_input)
    console.print(prompt)
    console.print("\n[dim](Dry run complete — no API calls made)[/dim]")


async def run(args) -> None:
    task_dir = pathlib.Path(args.task_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_id:
        task_path = task_dir / f"{args.task_id}.json"
        tasks = {args.task_id: load_task(task_path)}
    else:
        tasks = load_all_tasks(task_dir)
        if args.num_tasks:
            tasks = dict(list(tasks.items())[: args.num_tasks])

    console.print(f"\n[bold]Running LOO on {len(tasks)} task(s) with model={args.model}[/bold]\n")

    llm = ChatOpenAI(model=args.model, temperature=0)
    chain = build_chain(llm)
    all_results = []

    for task_id, task in tasks.items():
        console.print(
            f"  Processing [cyan]{task_id}[/cyan] "
            f"({len(task.train)} train, {len(task.test)} test)..."
        )
        try:
            result = await run_task_loo(task, chain=chain, model=args.model)
        except Exception as e:
            console.print(f"  [red]ERROR on {task_id}: {e}[/red]")
            continue

        all_results.append(result)

        out_path = output_dir / f"{task_id}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Summary table
    table = Table(title="LOO Results Summary", show_lines=True)
    table.add_column("task_id", style="cyan")
    table.add_column("loo_acc", justify="right")
    table.add_column("test_acc", justify="right")
    table.add_column("pairs (train/test)", justify="right")

    for r in all_results:
        table.add_row(
            r.task_id,
            f"{r.loo_accuracy:.2f}",
            f"{r.test_accuracy:.2f}",
            f"{len(r.loo_results)}/{len(r.test_results)}",
        )

    console.print(table)

    if all_results:
        agg = evaluate_batch(all_results)
        console.print(f"\n[bold]Aggregate stats ({agg['num_tasks']} tasks):[/bold]")
        console.print(f"  Mean LOO accuracy:     {agg['mean_loo_accuracy']:.3f}")
        console.print(f"  Mean test accuracy:    {agg['mean_test_accuracy']:.3f}")
        console.print(f"  Mean cell accuracy:    {agg['mean_cell_accuracy']:.3f}")
        console.print(f"  Tasks w/ perfect LOO:  {agg['tasks_with_perfect_loo']}")
        console.print(f"  Tasks w/ perfect test: {agg['tasks_with_perfect_test']}")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        do_dry_run(args.task_dir, args.model)
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
