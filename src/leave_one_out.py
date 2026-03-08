"""Leave-one-out orchestration for ARC-AGI hypothesis evaluation.

For a Task with N train pairs:
  - LOO: hold out train[i], use remaining N-1 as demos, predict train[i].output
  - Test: use all N train pairs as demos, predict test[j].output

All N+M LLM calls for one task run concurrently via asyncio.gather.
After the initial pass, any LOO result with exact_match=False is refined
via a second LLM call that receives spatial error feedback (not ground truth).
"""
import asyncio
import logging

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.data_loader import Task
from src.grid_utils import grids_match, grid_diff
from src.hypothesis_agent import HypothesisOutput, generate_hypothesis, format_demo_pairs
from src.refine_agent import refine_hypothesis, build_refinement_chain
from src.synthesize_agent import MasterOutput, synthesize_hypotheses, build_synthesis_chain
from src.generator_agent import GeneratorResult, generate_from_hypothesis, build_generator_chain

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
    # Refinement fields (populated only for LOO calls where exact_match=False)
    refined_hypothesis: str | None = None
    refined_reasoning: str | None = None
    refined_predicted_output: list[list[int]] | None = None
    refined_confidence: float | None = None
    refined_exact_match: bool | None = None
    refined_cell_accuracy: float | None = None


class TaskResult(BaseModel):
    """Aggregated results for a single ARC task."""

    task_id: str
    loo_results: list[HypothesisResult]
    test_results: list[HypothesisResult]
    loo_accuracy: float   # fraction of LOO predictions that exact matched
    test_accuracy: float  # fraction of test predictions that exact matched
    refined_loo_accuracy: float | None = None  # accuracy using refined where available
    master_result: MasterOutput | None = None  # synthesised unified hypothesis
    generator_results: list[GeneratorResult] | None = None  # per-test generator outputs


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
    chain: Runnable,
    model: str,
) -> tuple[HypothesisResult, str, list[list[int]]]:
    """Run one LOO iteration and return result plus context needed for refinement."""
    demo_pairs = [p for i, p in enumerate(task.train) if i != held_out_idx]
    test_input = task.train[held_out_idx].input
    actual = task.train[held_out_idx].output
    hyp = await generate_hypothesis(demo_pairs, test_input, chain=chain, model=model)
    result = _make_result(hyp, actual, held_out_index=held_out_idx, is_test=False)
    demo_pairs_str = format_demo_pairs(demo_pairs)
    return result, demo_pairs_str, test_input


async def _run_test_call(
    task: Task,
    test_idx: int,
    chain: Runnable,
    model: str,
) -> HypothesisResult:
    """Run one test evaluation: use all train pairs as demos, predict test[test_idx].output."""
    hyp = await generate_hypothesis(
        task.train, task.test[test_idx].input, chain=chain, model=model
    )
    actual = task.test[test_idx].output
    return _make_result(hyp, actual, held_out_index=None, is_test=True)


async def _run_refinement(
    result: HypothesisResult,
    demo_pairs_str: str,
    test_input: list[list[int]],
    refine_chain: Runnable,
    model: str,
) -> HypothesisResult:
    """Refine a wrong LOO prediction using spatial error feedback."""
    initial_hyp = HypothesisOutput(
        hypothesis=result.hypothesis,
        reasoning=result.reasoning,
        predicted_output=result.predicted_output,
        confidence=result.confidence,
    )
    refined = await refine_hypothesis(
        demo_pairs_str=demo_pairs_str,
        test_input=test_input,
        history=[initial_hyp],
        predicted=result.predicted_output,
        actual=result.actual_output,
        chain=refine_chain,
        model=model,
    )
    diff = grid_diff(refined.predicted_output, result.actual_output)
    return result.model_copy(update={
        "refined_hypothesis": refined.hypothesis,
        "refined_reasoning": refined.reasoning,
        "refined_predicted_output": refined.predicted_output,
        "refined_confidence": refined.confidence,
        "refined_exact_match": grids_match(refined.predicted_output, result.actual_output),
        "refined_cell_accuracy": diff["cell_accuracy"],
    })


async def run_task_loo(
    task: Task,
    chain: Runnable | None = None,
    model: str = "gpt-4.1-mini",
    compress_analysts: bool = True,
) -> TaskResult:
    """Run leave-one-out evaluation for a single task.

    Runs all N LOO calls and M test calls concurrently. For LOO results where
    exact_match=False, runs a refinement call with spatial error feedback.

    Args:
        task: The ARC task to evaluate.
        chain: Optional pre-built LCEL chain (for testing). If None, builds from model name.
        model: OpenAI model name (used only when chain is None).

    Returns:
        TaskResult with LOO and test results, accuracy metrics, and refined_loo_accuracy.
    """
    if chain is None:
        from langchain_openai import ChatOpenAI
        from src.hypothesis_agent import build_chain
        chain = build_chain(ChatOpenAI(model=model, temperature=0))

    loo_coros = [_run_loo_call(task, i, chain, model) for i in range(len(task.train))]
    test_coros = [_run_test_call(task, j, chain, model) for j in range(len(task.test))]

    all_raw = await asyncio.gather(*loo_coros, *test_coros)
    n = len(task.train)
    loo_raw_list = list(all_raw[:n])   # each is (HypothesisResult, str, list)
    test_results = list(all_raw[n:])   # each is HypothesisResult
    print(f"[{task.task_id}] LOO+test done ({n} loo, {len(task.test)} test)", flush=True)

    loo_results_initial = [item[0] for item in loo_raw_list]

    def _accuracy(results: list[HypothesisResult]) -> float:
        if not results:
            return 0.0
        return sum(r.exact_match for r in results) / len(results)

    # Build refinement chain (same LLM, different prompt)
    from langchain_openai import ChatOpenAI
    refine_chain = build_refinement_chain(ChatOpenAI(model=model, temperature=0))

    # Run refinement concurrently for all wrong LOO predictions
    refine_coros = []
    refine_indices = []
    for idx, (result, demo_pairs_str, test_input) in enumerate(loo_raw_list):
        if not result.exact_match:
            refine_coros.append(
                _run_refinement(result, demo_pairs_str, test_input, refine_chain, model)
            )
            refine_indices.append(idx)

    loo_results = list(loo_results_initial)
    if refine_coros:
        refined_results = await asyncio.gather(*refine_coros)
        for idx, refined in zip(refine_indices, refined_results):
            loo_results[idx] = refined
    print(f"[{task.task_id}] refinement done ({len(refine_coros)} refined)", flush=True)

    # refined_loo_accuracy: use refined_exact_match where available, else exact_match
    def _refined_exact(r: HypothesisResult) -> bool:
        return r.refined_exact_match if r.refined_exact_match is not None else r.exact_match

    refined_loo_accuracy = (
        sum(_refined_exact(r) for r in loo_results) / len(loo_results)
        if loo_results else 0.0
    )

    synth_chain = build_synthesis_chain(ChatOpenAI(model=model, temperature=0))
    master_result = await synthesize_hypotheses(task, loo_results, chain=synth_chain, model=model, compress=compress_analysts)
    print(f"[{task.task_id}] synthesis done", flush=True)

    if master_result.unified_hypothesis and task.test:
        gen_chain = build_generator_chain(ChatOpenAI(model=model, temperature=0))
        gen_coros = [
            generate_from_hypothesis(
                hypothesis=master_result.unified_hypothesis,
                key_insight=master_result.reasoning,
                train_pairs=task.train,
                test_input=pair.input,
                chain=gen_chain,
                model=model,
            )
            for pair in task.test
        ]
        generator_results = list(await asyncio.gather(*gen_coros))
        print(f"[{task.task_id}] generator done ({len(generator_results)} test outputs)", flush=True)
    else:
        generator_results = None

    return TaskResult(
        task_id=task.task_id,
        loo_results=loo_results,
        test_results=test_results,
        loo_accuracy=_accuracy(loo_results),
        test_accuracy=_accuracy(test_results),
        refined_loo_accuracy=refined_loo_accuracy,
        master_result=master_result,
        generator_results=generator_results,
    )
