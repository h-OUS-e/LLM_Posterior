"""LangChain-based hypothesis generation for ARC-AGI tasks.

Uses a simple prompt chain (no function calling, no tools). The LLM is
instructed to return raw JSON. On parse failure, retries once, then
returns a zero-confidence result.
"""
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.data_loader import IOPair
from src.grid_utils import grid_to_str

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are solving an ARC-AGI puzzle. You will be shown input/output grid pairs that demonstrate a transformation rule. Grids use integers 0-9 representing colors.

## Demonstration pairs:
{demo_pairs}

## Task:
1. Analyze the demonstration pairs carefully.
2. Formulate a clear, concise hypothesis describing the transformation rule.
3. Apply your hypothesis to the following test input.
4. Return your answer as JSON.

## Test input:
{test_input}

Return ONLY valid JSON in this exact format, no markdown fences:
{{
  "hypothesis": "description of the transformation rule",
  "reasoning": "step-by-step reasoning for how you applied the rule",
  "predicted_output": [[row1], [row2], ...],
  "confidence": 0.0-1.0
}}\
"""


class HypothesisOutput(BaseModel):
    """Parsed LLM response for a single hypothesis generation call."""

    hypothesis: str
    reasoning: str
    predicted_output: list[list[int]]
    confidence: float


def _format_demo_pairs(demo_pairs: list[IOPair]) -> str:
    """Format demo pairs as numbered sections for the prompt."""
    sections = []
    for i, pair in enumerate(demo_pairs, 1):
        sections.append(
            f"Pair {i}:\nInput:\n{grid_to_str(pair.input)}\n\nOutput:\n{grid_to_str(pair.output)}"
        )
    return "\n\n".join(sections)


def build_prompt(demo_pairs: list[IOPair], test_input: list[list[int]]) -> str:
    """Build the full prompt string for a hypothesis generation call.

    Args:
        demo_pairs: Demonstration input/output pairs.
        test_input: The grid to predict the output for.

    Returns:
        Formatted prompt string.
    """
    return PROMPT_TEMPLATE.format(
        demo_pairs=_format_demo_pairs(demo_pairs),
        test_input=grid_to_str(test_input),
    )


def _parse_response(content: str) -> HypothesisOutput | None:
    """Attempt to parse LLM JSON response into HypothesisOutput.

    Returns None on any parse error.
    """
    try:
        data = json.loads(content.strip())
        return HypothesisOutput(**data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.warning("JSON parse failed: %s | content: %.200s", e, content)
        return None


async def generate_hypothesis(
    demo_pairs: list[IOPair],
    test_input: list[list[int]],
    llm: Any = None,
    model: str = "gpt-4o",
) -> HypothesisOutput:
    """Generate a hypothesis for the given demo pairs and test input.

    Args:
        demo_pairs: Demonstration pairs used as few-shot examples.
        test_input: The grid the model must predict the output for.
        llm: Optional pre-configured LLM (for testing). If None, creates ChatOpenAI.
        model: OpenAI model name (used only when llm is None).

    Returns:
        HypothesisOutput. On double failure, returns empty output with confidence=0.
    """
    if llm is None:
        llm = ChatOpenAI(model=model, temperature=0)

    prompt = build_prompt(demo_pairs, test_input)

    for attempt in range(2):
        try:
            response = await llm.ainvoke(prompt)
            result = _parse_response(response.content)
            if result is not None:
                return result
            logger.warning("Parse attempt %d failed, retrying...", attempt + 1)
        except Exception as e:
            logger.error("LLM call failed on attempt %d: %s", attempt + 1, e)

    logger.error("Both attempts failed. Returning empty HypothesisOutput.")
    return HypothesisOutput(
        hypothesis="",
        reasoning="",
        predicted_output=[],
        confidence=0.0,
    )
