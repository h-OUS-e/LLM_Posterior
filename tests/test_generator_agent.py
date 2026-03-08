"""Tests for generator_agent module."""
from unittest.mock import AsyncMock

from src.data_loader import IOPair
from src.generator_agent import GeneratorResult, generate_from_hypothesis


TRAIN_PAIRS = [
    IOPair(input=[[0, 1]], output=[[1, 0]]),
    IOPair(input=[[1, 1]], output=[[0, 0]]),
]
TEST_INPUT = [[0, 0]]

RESULT = GeneratorResult(
    step_by_step_trace="flip each bit: 0→1, 0→1",
    predicted_output=[[1, 1]],
)


async def test_generator_returns_result():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = RESULT

    result = await generate_from_hypothesis(
        hypothesis="invert all bits",
        key_insight="each 0 becomes 1, each 1 becomes 0",
        train_pairs=TRAIN_PAIRS,
        test_input=TEST_INPUT,
        chain=mock_chain,
    )

    assert isinstance(result, GeneratorResult)
    assert result.predicted_output == [[1, 1]]
    assert "flip" in result.step_by_step_trace


async def test_generator_passes_hypothesis():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = RESULT

    await generate_from_hypothesis(
        hypothesis="invert all bits",
        key_insight="each 0 becomes 1",
        train_pairs=TRAIN_PAIRS,
        test_input=TEST_INPUT,
        chain=mock_chain,
    )

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert kwargs["unified_hypothesis"] == "invert all bits"
    assert kwargs["key_insight"] == "each 0 becomes 1"


async def test_generator_passes_train_pairs():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = RESULT

    await generate_from_hypothesis(
        hypothesis="invert",
        key_insight="flip",
        train_pairs=TRAIN_PAIRS,
        test_input=TEST_INPUT,
        chain=mock_chain,
    )

    kwargs = mock_chain.ainvoke.call_args[0][0]
    assert "Pair 1:" in kwargs["train_pairs"]
    assert "Pair 2:" in kwargs["train_pairs"]


async def test_generator_fallback_on_error():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("fail")

    result = await generate_from_hypothesis(
        hypothesis="invert",
        key_insight="flip",
        train_pairs=TRAIN_PAIRS,
        test_input=TEST_INPUT,
        chain=mock_chain,
    )

    assert result.predicted_output == []
    assert result.step_by_step_trace == ""
