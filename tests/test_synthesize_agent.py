"""Tests for synthesize_agent module."""
from unittest.mock import AsyncMock

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

MASTER_OUTPUT = MasterOutput(
    analyst_gaps="analysts identified inversion directionally",
    key_transformations="every cell: 0→1, 1→0",
    reasoning="no variable part; rule applies uniformly to all cells",
    unified_hypothesis="for each cell, output = 1 - input",
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
    mock_chain.ainvoke.return_value = MASTER_OUTPUT

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    assert isinstance(result, MasterOutput)
    assert result.unified_hypothesis == "for each cell, output = 1 - input"


async def test_synthesize_passes_all_train_pairs():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_OUTPUT

    await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "Pair 1:" in kwargs["train_pairs"]
    assert "Pair 2:" in kwargs["train_pairs"]


async def test_synthesize_includes_analyst_summaries():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_OUTPUT

    await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "Analyst 1" in kwargs["analyst_summaries"]
    assert "Analyst 2" in kwargs["analyst_summaries"]
    assert "invert all bits." in kwargs["analyst_summaries"]  # refined hypothesis
    assert "correction_feedback" in kwargs
    assert "test_inputs" not in kwargs  # test inputs handled by generator agent


async def test_synthesize_called_once():
    """Chain should be called exactly once (no retry loop)."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MASTER_OUTPUT

    await synthesize_hypotheses(TASK, [LOO_INITIAL, LOO_REFINED], chain=mock_chain)

    assert mock_chain.ainvoke.call_count == 1


async def test_synthesize_fallback_on_error():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("fail")

    result = await synthesize_hypotheses(TASK, [LOO_INITIAL], chain=mock_chain)

    assert result.unified_hypothesis == ""
