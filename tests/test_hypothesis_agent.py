"""Tests for hypothesis_agent module."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.data_loader import IOPair
from src.hypothesis_agent import HypothesisOutput, build_prompt, generate_hypothesis


DEMO_PAIRS = [
    IOPair(input=[[0, 1], [1, 0]], output=[[1, 0], [0, 1]]),
    IOPair(input=[[0, 0], [1, 1]], output=[[1, 1], [0, 0]]),
]
TEST_INPUT = [[0, 1, 0], [1, 0, 1]]

MOCK_RESPONSE = json.dumps({
    "hypothesis": "flip rows",
    "reasoning": "rows are inverted",
    "predicted_output": [[1, 0, 1], [0, 1, 0]],
    "confidence": 0.85,
})


def test_build_prompt_contains_demo_data():
    prompt = build_prompt(DEMO_PAIRS, TEST_INPUT)
    assert "Pair 1:" in prompt
    assert "Pair 2:" in prompt
    assert "0 1" in prompt
    assert "1 0" in prompt


def test_build_prompt_contains_test_input():
    prompt = build_prompt(DEMO_PAIRS, TEST_INPUT)
    assert "0 1 0" in prompt


def test_hypothesis_output_model():
    h = HypothesisOutput(
        hypothesis="flip horizontally",
        reasoning="each row reversed",
        predicted_output=[[0, 1, 0]],
        confidence=0.9,
    )
    assert h.confidence == 0.9
    assert h.predicted_output == [[0, 1, 0]]


async def test_generate_hypothesis_parses_response():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=MOCK_RESPONSE)

    result = await generate_hypothesis(DEMO_PAIRS, TEST_INPUT, llm=mock_llm)
    assert isinstance(result, HypothesisOutput)
    assert result.confidence == 0.85
    assert result.predicted_output == [[1, 0, 1], [0, 1, 0]]


async def test_generate_hypothesis_retries_on_bad_json():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = [
        MagicMock(content="not json {{{{"),
        MagicMock(content=MOCK_RESPONSE),
    ]

    result = await generate_hypothesis(DEMO_PAIRS, TEST_INPUT, llm=mock_llm)
    assert mock_llm.ainvoke.call_count == 2
    assert result.confidence == 0.85


async def test_generate_hypothesis_returns_empty_on_double_failure():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="totally invalid")

    result = await generate_hypothesis(DEMO_PAIRS, TEST_INPUT, llm=mock_llm)
    assert result.predicted_output == []
    assert result.confidence == 0.0
