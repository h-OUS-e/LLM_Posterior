"""Load and parse ARC-AGI task JSON files into Pydantic models."""
import json
import pathlib
from pydantic import BaseModel


class IOPair(BaseModel):
    """A single input/output demonstration pair."""

    input: list[list[int]]
    output: list[list[int]]


class Task(BaseModel):
    """An ARC task with train/test pairs."""

    task_id: str
    train: list[IOPair]
    test: list[IOPair]


def load_task(path: str | pathlib.Path) -> Task:
    """Load a single ARC task from a JSON file.

    Args:
        path: Path to the task JSON file. The task_id is derived from the filename stem.

    Returns:
        Parsed Task model.
    """
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Task(
        task_id=path.stem,
        train=[IOPair(**p) for p in data["train"]],
        test=[IOPair(**p) for p in data["test"]],
    )


def load_all_tasks(directory: str | pathlib.Path) -> dict[str, Task]:
    """Load all ARC tasks from a directory.

    Args:
        directory: Path to directory containing task JSON files.

    Returns:
        Dict mapping task_id -> Task.
    """
    directory = pathlib.Path(directory)
    return {
        path.stem: load_task(path)
        for path in sorted(directory.glob("*.json"))
    }
