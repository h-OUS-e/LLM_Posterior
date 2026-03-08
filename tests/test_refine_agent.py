"""Tests for refine_agent module."""
from unittest.mock import AsyncMock

from src.hypothesis_agent import HypothesisOutput
from src.refine_agent import _make_diff_grid_str, refine_hypothesis


# --- _make_diff_grid_str ---

def test_make_diff_grid_str_all_correct():
    result = _make_diff_grid_str([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    assert result == "1 2\n3 4"


def test_make_diff_grid_str_all_wrong():
    result = _make_diff_grid_str([[1, 2]], [[3, 4]])
    assert result == "x x"


def test_make_diff_grid_str_mixed():
    result = _make_diff_grid_str([[1, 9]], [[1, 2]])
    assert result == "1 x"


def test_make_diff_grid_str_size_mismatch():
    result = _make_diff_grid_str([[1, 2]], [[1, 2, 3]])
    assert "size mismatch" in result


# --- refine_hypothesis ---

INITIAL = HypothesisOutput(
    hypothesis="flip", reasoning="test", predicted_output=[[1, 0]], confidence=0.5
)
REFINED = HypothesisOutput(
    hypothesis="rotate", reasoning="better", predicted_output=[[0, 1]], confidence=0.8
)


async def test_refine_hypothesis_returns_refined():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = REFINED

    result = await refine_hypothesis(
        demo_pairs_str="Pair 1:\nInput:\n0 1\n\nOutput:\n1 0",
        test_input=[[1, 0]],
        history=[INITIAL],
        predicted=[[1, 0]],
        actual=[[0, 1]],
        chain=mock_chain,
    )
    assert result.predicted_output == [[0, 1]]
    assert result.confidence == 0.8


async def test_refine_hypothesis_passes_ground_truth():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = REFINED

    await refine_hypothesis(
        demo_pairs_str="Pair 1:\nInput:\n0 1\n\nOutput:\n1 0",
        test_input=[[1, 0]],
        history=[INITIAL],
        predicted=[[1, 0]],
        actual=[[0, 1]],
        chain=mock_chain,
    )
    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "actual_grid" in kwargs
    assert "diff_grid" in kwargs
    assert "original_hypothesis" in kwargs
    assert "predicted_grid" in kwargs
    assert "cell_accuracy_pct" in kwargs
    assert kwargs["original_hypothesis"] == "flip"  # INITIAL.hypothesis
    assert "x" in kwargs["diff_grid"]              # both cells wrong → all x


async def test_refine_hypothesis_fallback_on_error():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("fail")

    result = await refine_hypothesis(
        demo_pairs_str="",
        test_input=[[0]],
        history=[INITIAL],
        predicted=[[0]],
        actual=[[1]],
        chain=mock_chain,
    )
    assert result.predicted_output == []
    assert result.confidence == 0.0
