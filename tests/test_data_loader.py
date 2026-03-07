"""Tests for data_loader module."""
import json
import pathlib
import pytest
from src.data_loader import IOPair, Task, load_task, load_all_tasks


SAMPLE_TASK = {
    "train": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
    ],
    "test": [
        {"input": [[0, 1, 0], [1, 0, 1]], "output": [[1, 0, 1], [0, 1, 0]]}
    ],
}


@pytest.fixture
def sample_task_file(tmp_path):
    f = tmp_path / "abc123.json"
    f.write_text(json.dumps(SAMPLE_TASK))
    return f


def test_load_task_returns_task_model(sample_task_file):
    task = load_task(sample_task_file)
    assert isinstance(task, Task)
    assert task.task_id == "abc123"
    assert len(task.train) == 2
    assert len(task.test) == 1


def test_iopair_fields(sample_task_file):
    task = load_task(sample_task_file)
    pair = task.train[0]
    assert isinstance(pair, IOPair)
    assert pair.input == [[0, 1], [1, 0]]
    assert pair.output == [[1, 0], [0, 1]]


def test_load_all_tasks(tmp_path):
    for name in ["task1", "task2"]:
        (tmp_path / f"{name}.json").write_text(json.dumps(SAMPLE_TASK))
    tasks = load_all_tasks(tmp_path)
    assert set(tasks.keys()) == {"task1", "task2"}
    assert all(isinstance(t, Task) for t in tasks.values())


def test_load_task_uses_stem_as_id(sample_task_file):
    task = load_task(sample_task_file)
    assert task.task_id == sample_task_file.stem
