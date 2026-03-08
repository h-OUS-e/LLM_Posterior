"""Tests for leave_one_out orchestration."""
import pytest
from unittest.mock import patch, AsyncMock
from src.data_loader import IOPair, Task
from src.hypothesis_agent import HypothesisOutput
from src.leave_one_out import HypothesisResult, TaskResult, run_task_loo
from src.synthesize_agent import MasterOutput
from src.generator_agent import GeneratorResult


TASK = Task(
    task_id="test_task",
    train=[
        IOPair(input=[[0, 1]], output=[[1, 0]]),
        IOPair(input=[[1, 1]], output=[[0, 0]]),
        IOPair(input=[[0, 0]], output=[[1, 1]]),
    ],
    test=[
        IOPair(input=[[0, 1, 0]], output=[[1, 0, 1]]),
    ],
)

# Exactly matches train[0].output = [[1, 0]]
GOOD_HYPOTHESIS = HypothesisOutput(
    hypothesis="invert bits",
    reasoning="each 0->1, 1->0",
    predicted_output=[[1, 0]],
    confidence=0.9,
)

# [[9, 9]] won't match any train output, avoiding accidental matches
BAD_HYPOTHESIS = HypothesisOutput(
    hypothesis="wrong",
    reasoning="wrong",
    predicted_output=[[9, 9]],
    confidence=0.1,
)

MASTER_FALLBACK = MasterOutput(
    agreements="",
    disagreements="",
    key_insight="",
    unified_hypothesis="",
    train_predictions=[],
)

GEN_FALLBACK = GeneratorResult(step_by_step_trace="", predicted_output=[])


async def test_run_task_loo_returns_task_result():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert isinstance(result, TaskResult)
    assert result.task_id == "test_task"
    assert len(result.loo_results) == 3
    assert len(result.test_results) == 1


async def test_loo_results_have_correct_held_out_index():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    held_out_indices = [r.held_out_index for r in result.loo_results]
    assert held_out_indices == [0, 1, 2]


async def test_test_results_have_is_test_true():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert all(r.is_test for r in result.test_results)
    assert not any(r.is_test for r in result.loo_results)


async def test_loo_accuracy_calculation():
    call_count = 0

    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        nonlocal call_count
        call_count += 1
        # First LOO call: GOOD → [[1,0]] matches train[0].output [[1,0]]
        # Remaining calls: BAD → [[9,9]] matches nothing
        if call_count == 1:
            return GOOD_HYPOTHESIS
        return BAD_HYPOTHESIS

    mock_refine = AsyncMock(return_value=BAD_HYPOTHESIS)
    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.refine_hypothesis", mock_refine), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert result.loo_accuracy == pytest.approx(1 / 3)


async def test_task_result_has_refined_loo_accuracy():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    mock_refine = AsyncMock(return_value=BAD_HYPOTHESIS)
    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.refine_hypothesis", mock_refine), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert hasattr(result, "refined_loo_accuracy")
    assert result.refined_loo_accuracy is not None


async def test_task_result_has_master_result():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    mock_synth = AsyncMock(return_value=MASTER_FALLBACK)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert result.master_result is not None
    assert isinstance(result.master_result, MasterOutput)


async def test_task_result_has_generator_results():
    async def mock_generate(demo_pairs, test_input, chain=None, model="gpt-4o"):
        return GOOD_HYPOTHESIS

    # Master with non-empty hypothesis so generator branch fires
    master_with_hyp = MasterOutput(
        agreements="a", disagreements="d", key_insight="k",
        unified_hypothesis="invert bits",
        train_predictions=[],
    )
    mock_synth = AsyncMock(return_value=master_with_hyp)
    mock_gen = AsyncMock(return_value=GEN_FALLBACK)
    with patch("src.leave_one_out.generate_hypothesis", mock_generate), \
         patch("src.leave_one_out.synthesize_hypotheses", mock_synth), \
         patch("src.leave_one_out.generate_from_hypothesis", mock_gen):
        result = await run_task_loo(TASK)

    assert result.generator_results is not None
    assert len(result.generator_results) == len(TASK.test)
    assert isinstance(result.generator_results[0], GeneratorResult)
