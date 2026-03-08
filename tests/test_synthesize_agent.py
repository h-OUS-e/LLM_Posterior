"""Tests for synthesize_agent module."""
from unittest.mock import AsyncMock, call

from src.data_loader import IOPair, Task
from src.leave_one_out import HypothesisResult
from src.synthesize_agent import (
    MasterOutput,
    _format_analyst_summaries,
    synthesize_hypotheses,
)


TASK = Task(
    task_id="t1",
    train=[
        IOPair(input=[[0, 1]], output=[[1, 0]]),
        IOPair(input=[[1, 1]], output=[[0, 0]]),
    ],
    test=[IOPair(input=[[0, 0]], output=[[1, 1]])],
)

# LOO result with no refinement
LOO_INITIAL = HypothesisResult(
    hypothesis="flip bits.",
    reasoning="0→1, 1→0",
    predicted_output=[[1, 0]],
    confidence=0.9,
    held_out_index=0,
    is_test=False,
    actual_output=[[1, 0]],
    exact_match=True,
    cell_accuracy=1.0,
)

# LOO result with refinement fields populated
LOO_REFINED = HypothesisResult(
    hypothesis="wrong guess.",
    reasoning="bad",
    predicted_output=[[0, 0]],
    confidence=0.3,
    held_out_index=1,
    is_test=False,
    actual_output=[[0, 0]],
    exact_match=False,
    cell_accuracy=0.5,
    refined_hypothesis="invert all bits.",
    refined_reasoning="each cell flips",
    refined_predicted_output=[[0, 0]],
    refined_confidence=0.8,
    refined_exact_match=True,
    refined_cell_accuracy=1.0,
)

# Master where train_predictions match TASK.train outputs exactly
MASTER_CORRECT = MasterOutput(
    agreements="both invert bits",
    disagreements="none",
    key_insight="always invert",
    unified_hypothesis="invert all bits",
    train_predictions=[[[1, 0]], [[0, 0]]],   # matches TASK.train outputs
)

# Master where train[0] prediction is wrong
MASTER_WRONG_0 = MasterOutput(
    agreements="agreement",
    disagreements="disagreement",
    key_insight="insight",
    unified_hypothesis="wrong",
    train_predictions=[[[9, 9]], [[0, 0]]],   # train[0] wrong, train[1] correct
)

# Master where both predictions are wrong
MASTER_ALL_WRONG = MasterOutput(
    agreements="a",
    disagreements="d",
    key_insight="k",
    unified_hypothesis="wrong",
    train_predictions=[[[9, 9]], [[9, 9]]],
)


# --- _format_analyst_summaries ---

def test_format_analyst_summaries_labels():
    text = _format_analyst_summaries([LOO_INITIAL, LOO_REFINED])
    assert "Analyst 1" in text
    assert "Analyst 2" in text


def test_format_analyst_summaries_uses_refined():
    text = _format_analyst_summaries([LOO_INITIAL, LOO_REFINED])
    assert "invert all bits." in text
    assert "wrong guess." in text  # original still shown


# --- synthesize_hypotheses ---

async def test_synthesize_returns_master_output():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_CORRECT

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    assert isinstance(result, MasterOutput)
    assert result.unified_hypothesis == "invert all bits"
    assert result.master_train_accuracy == 1.0
    assert result.train_exact_matches == [True, True]


async def test_synthesize_passes_all_train_pairs():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_CORRECT

    await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "Pair 1:" in kwargs["train_pairs"]
    assert "Pair 2:" in kwargs["train_pairs"]


async def test_synthesize_includes_analyst_summaries():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_CORRECT

    await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "Analyst 1" in kwargs["analyst_summaries"]
    assert "Analyst 2" in kwargs["analyst_summaries"]
    assert "invert all bits." in kwargs["analyst_summaries"]  # refined hypothesis
    assert "correction_feedback" in kwargs
    assert "test_inputs" not in kwargs  # test inputs handled by generator agent


async def test_synthesize_no_retry_when_all_correct():
    """Chain should be called exactly once when train predictions are all correct."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_CORRECT

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    assert mock_chain.ainvoke.call_count == 1
    assert result.master_train_accuracy == 1.0


async def test_synthesize_retries_on_wrong_prediction():
    """Chain retries once if train predictions fail; uses second result when better."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = [MASTER_WRONG_0, MASTER_CORRECT]

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    assert mock_chain.ainvoke.call_count == 2
    # Second call should include correction_feedback
    second_kwargs = mock_chain.ainvoke.call_args_list[1][0][0]
    assert "correction_feedback" in second_kwargs
    assert "✗" in second_kwargs["correction_feedback"]
    # Final result should use the better (correct) attempt
    assert result.master_train_accuracy == 1.0


async def test_synthesize_keeps_better_attempt_on_failed_retry():
    """When both attempts are wrong, keeps the one with higher accuracy."""
    # MASTER_WRONG_0: train[0] wrong, train[1] correct → accuracy 0.5
    # MASTER_ALL_WRONG: both wrong → accuracy 0.0
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = [MASTER_WRONG_0, MASTER_ALL_WRONG]

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    # First attempt (0.5) beats second (0.0)
    assert result.unified_hypothesis == "wrong"
    assert result.master_train_accuracy == 0.5
    assert result.train_exact_matches == [False, True]


async def test_synthesize_fallback_on_error():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("fail")

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL], chain=mock_chain)

    assert result.unified_hypothesis == ""
    assert result.train_predictions == []
    assert result.master_train_accuracy == 0.0
