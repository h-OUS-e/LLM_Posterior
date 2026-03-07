"""Leave-one-out orchestration for ARC-AGI hypothesis evaluation.

For a Task with N train pairs:
  - LOO: hold out train[i], use remaining N-1 as demos, predict train[i].output
  - Test: use all N train pairs as demos, predict test[j].output

All N+M LLM calls for one task run concurrently via asyncio.gather.
"""
import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from src.data_loader import Task
from src.grid_utils import grids_match, grid_diff
from src.hypothesis_agent import HypothesisOutput, generate_hypothesis

logger = logging.getLogger(__name__)


class HypothesisResult(BaseModel):
    """Result of a single hypothesis generation call, augmented with ground truth."""

    hypothesis: str
    reasoning: str
    predicted_output: list[list[int]]
    confidence: float
    held_out_index: int | None  # which train pair was held out; None for test pairs
    is_test: bool
    actual_output: list[list[int]]
    exact_match: bool
    cell_accuracy: float


class TaskResult(BaseModel):
    """Aggregated results for a single ARC task."""

    task_id: str
    loo_results: list[HypothesisResult]
    test_results: list[HypothesisResult]
    loo_accuracy: float   # fraction of LOO predictions that exact matched
    test_accuracy: float  # fraction of test predictions that exact matched


def _make_result(
    hyp: HypothesisOutput,
    actual: list[list[int]],
    held_out_index: int | None,
    is_test: bool,
) -> HypothesisResult:
    """Convert a HypothesisOutput + ground truth into a HypothesisResult."""
    diff = grid_diff(hyp.predicted_output, actual)
    return HypothesisResult(
        hypothesis=hyp.hypothesis,
        reasoning=hyp.reasoning,
        predicted_output=hyp.predicted_output,
        confidence=hyp.confidence,
        held_out_index=held_out_index,
        is_test=is_test,
        actual_output=actual,
        exact_match=grids_match(hyp.predicted_output, actual),
        cell_accuracy=diff["cell_accuracy"],
    )


async def _run_loo_call(
    task: Task,
    held_out_idx: int,
    llm: Any,
    model: str,
) -> HypothesisResult:
    """Run one LOO iteration: hold out train[held_out_idx], predict its output."""
    demo_pairs = [p for i, p in enumerate(task.train) if i != held_out_idx]
    test_input = task.train[held_out_idx].input
    actual = task.train[held_out_idx].output
    hyp = await generate_hypothesis(demo_pairs, test_input, llm=llm, model=model)
    return _make_result(hyp, actual, held_out_index=held_out_idx, is_test=False)


async def _run_test_call(
    task: Task,
    test_idx: int,
    llm: Any,
    model: str,
) -> HypothesisResult:
    """Run one test evaluation: use all train pairs as demos, predict test[test_idx].output."""
    hyp = await generate_hypothesis(
        task.train, task.test[test_idx].input, llm=llm, model=model
    )
    actual = task.test[test_idx].output
    return _make_result(hyp, actual, held_out_index=None, is_test=True)


async def run_task_loo(
    task: Task,
    llm: Any = None,
    model: str = "gpt-4o",
) -> TaskResult:
    """Run leave-one-out evaluation for a single task.

    Runs all N LOO calls and M test calls concurrently via asyncio.gather.

    Args:
        task: The ARC task to evaluate.
        llm: Optional pre-configured LLM (for testing). If None, creates ChatOpenAI.
        model: OpenAI model name (used only when llm is None).

    Returns:
        TaskResult with LOO and test results plus accuracy metrics.
    """
    loo_coros = [_run_loo_call(task, i, llm, model) for i in range(len(task.train))]
    test_coros = [_run_test_call(task, j, llm, model) for j in range(len(task.test))]

    all_results = await asyncio.gather(*loo_coros, *test_coros)
    n = len(task.train)
    loo_results = list(all_results[:n])
    test_results = list(all_results[n:])

    def _accuracy(results: list[HypothesisResult]) -> float:
        if not results:
            return 0.0
        return sum(r.exact_match for r in results) / len(results)

    return TaskResult(
        task_id=task.task_id,
        loo_results=loo_results,
        test_results=test_results,
        loo_accuracy=_accuracy(loo_results),
        test_accuracy=_accuracy(test_results),
    )
