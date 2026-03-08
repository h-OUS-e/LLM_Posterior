"""Tests for hypothesis_agent module."""
import pytest
from unittest.mock import AsyncMock
from src.data_loader import IOPair
from src.hypothesis_agent import HypothesisOutput, build_prompt, build_chain, generate_hypothesis


DEMO_PAIRS = [
    IOPair(input=[[0, 1], [1, 0]], output=[[1, 0], [0, 1]]),
    IOPair(input=[[0, 0], [1, 1]], output=[[1, 1], [0, 0]]),
]
TEST_INPUT = [[0, 1, 0], [1, 0, 1]]

MOCK_OUTPUT = HypothesisOutput(
    hypothesis="flip rows",
    reasoning="rows are inverted",
    predicted_output=[[1, 0, 1], [0, 1, 0]],
    confidence=0.85,
)


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
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = MOCK_OUTPUT

    result = await generate_hypothesis(DEMO_PAIRS, TEST_INPUT, chain=mock_chain)
    assert isinstance(result, HypothesisOutput)
    assert result.confidence == 0.85
    assert result.predicted_output == [[1, 0, 1], [0, 1, 0]]


async def test_generate_hypothesis_returns_empty_on_failure():
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("API error")

    result = await generate_hypothesis(DEMO_PAIRS, TEST_INPUT, chain=mock_chain)
    assert result.predicted_output == []
    assert result.confidence == 0.0


def test_build_chain_returns_runnable():
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = build_chain(llm)
    assert isinstance(chain, Runnable)
