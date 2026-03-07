"""Scoring and metrics for ARC-AGI hypothesis evaluation results."""
from src.leave_one_out import HypothesisResult, TaskResult


def evaluate_task(task_result: TaskResult) -> dict:
    """Compute per-task metrics from a TaskResult.

    Args:
        task_result: Result from run_task_loo for one task.

    Returns:
        Dict with:
            - task_id: task identifier.
            - loo_exact_match_rate: fraction of LOO predictions that exactly matched.
            - test_exact_match_rate: fraction of test predictions that exactly matched.
            - mean_cell_accuracy: mean cell accuracy across all LOO + test results.
            - num_loo_pairs: number of LOO iterations run.
            - num_test_pairs: number of test pairs evaluated.
    """
    all_results = task_result.loo_results + task_result.test_results

    def _exact_rate(results: list[HypothesisResult]) -> float:
        if not results:
            return 0.0
        return sum(r.exact_match for r in results) / len(results)

    mean_cell_acc = (
        sum(r.cell_accuracy for r in all_results) / len(all_results)
        if all_results else 0.0
    )

    return {
        "task_id": task_result.task_id,
        "loo_exact_match_rate": _exact_rate(task_result.loo_results),
        "test_exact_match_rate": _exact_rate(task_result.test_results),
        "mean_cell_accuracy": mean_cell_acc,
        "num_loo_pairs": len(task_result.loo_results),
        "num_test_pairs": len(task_result.test_results),
    }


def evaluate_batch(results: list[TaskResult]) -> dict:
    """Aggregate metrics across all tasks.

    Args:
        results: List of TaskResults from multiple tasks.

    Returns:
        Dict with aggregate stats.
    """
    if not results:
        return {"num_tasks": 0}

    per_task = [evaluate_task(r) for r in results]

    return {
        "num_tasks": len(results),
        "mean_loo_accuracy": sum(m["loo_exact_match_rate"] for m in per_task) / len(per_task),
        "mean_test_accuracy": sum(m["test_exact_match_rate"] for m in per_task) / len(per_task),
        "mean_cell_accuracy": sum(m["mean_cell_accuracy"] for m in per_task) / len(per_task),
        "tasks_with_perfect_loo": sum(1 for m in per_task if m["loo_exact_match_rate"] == 1.0),
        "tasks_with_perfect_test": sum(1 for m in per_task if m["test_exact_match_rate"] == 1.0),
    }
