"""Tests for evaluate module."""
import pytest
from src.leave_one_out import HypothesisResult, TaskResult
from src.evaluate import evaluate_task, evaluate_batch


def make_result(exact_match: bool, cell_accuracy: float, is_test: bool = False) -> HypothesisResult:
    return HypothesisResult(
        hypothesis="h", reasoning="r",
        predicted_output=[[1]], confidence=0.9,
        held_out_index=None if is_test else 0,
        is_test=is_test,
        actual_output=[[1]] if exact_match else [[0]],
        exact_match=exact_match,
        cell_accuracy=cell_accuracy,
    )


def make_task_result(task_id, loo_exact, test_exact) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        loo_results=[make_result(e, 1.0 if e else 0.0) for e in loo_exact],
        test_results=[make_result(e, 1.0 if e else 0.0, is_test=True) for e in test_exact],
        loo_accuracy=sum(loo_exact) / len(loo_exact) if loo_exact else 0.0,
        test_accuracy=sum(test_exact) / len(test_exact) if test_exact else 0.0,
    )


def test_evaluate_task_exact_match_rate():
    tr = make_task_result("t1", [True, True, False], [True])
    metrics = evaluate_task(tr)
    assert metrics["loo_exact_match_rate"] == pytest.approx(2 / 3)
    assert metrics["test_exact_match_rate"] == 1.0


def test_evaluate_task_mean_cell_accuracy():
    # loo=[True, False] → cell_accuracy=[1.0, 0.0], test=[True] → [1.0]
    # mean = (1.0 + 0.0 + 1.0) / 3 = 2/3
    tr = make_task_result("t1", [True, False], [True])
    metrics = evaluate_task(tr)
    assert metrics["mean_cell_accuracy"] == pytest.approx(2 / 3)


def test_evaluate_batch_aggregates():
    results = [
        make_task_result("t1", [True, True], [True]),
        make_task_result("t2", [False, False], [False]),
    ]
    agg = evaluate_batch(results)
    assert agg["num_tasks"] == 2
    assert agg["mean_loo_accuracy"] == pytest.approx(0.5)
    assert agg["mean_test_accuracy"] == pytest.approx(0.5)
    assert agg["tasks_with_perfect_loo"] == 1
