"""Generator agent: executes a known hypothesis on a test input to produce a grid.

Receives the master's unified hypothesis + key insight + all train pairs as worked
examples, then applies the rule precisely to a single test input.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.data_loader import IOPair
from src.grid_utils import grid_to_str
from src.hypothesis_agent import format_demo_pairs

logger = logging.getLogger(__name__)


GENERATOR_SYSTEM = """\
You are executing a transformation rule on an ARC-AGI puzzle. You have been given the \
correct rule and examples of it applied. Your ONLY job is to apply the rule precisely to \
the test input and produce the correct output grid. Focus entirely on accuracy — check every cell.\
"""

GENERATOR_HUMAN_TEMPLATE = """\
## The transformation rule:
{unified_hypothesis}

## Key insight:
{key_insight}

## Examples of the rule applied (input → output):
{train_pairs}

## Apply the rule to this test input:
{test_input}

Mentally trace the rule step by step on the test input. Verify each cell before outputting.\
"""

_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM),
    ("human", GENERATOR_HUMAN_TEMPLATE),
])


class GeneratorResult(BaseModel):
    """Structured output for the generator agent."""

    step_by_step_trace: str
    predicted_output: list[list[int]]


_FALLBACK = GeneratorResult(step_by_step_trace="", predicted_output=[])


def build_generator_chain(llm: ChatOpenAI) -> Runnable:
    """Construct the LCEL generator chain: prompt | structured_llm."""
    return (_GENERATOR_PROMPT | llm.with_structured_output(GeneratorResult)).with_retry(
        stop_after_attempt=2
    )


async def generate_from_hypothesis(
    hypothesis: str,
    key_insight: str,
    train_pairs: list[IOPair],
    test_input: list[list[int]],
    chain: Runnable | None = None,
    model: str = "gpt-4.1-mini",
) -> GeneratorResult:
    """Apply a known hypothesis to a single test input to produce a predicted grid.

    Args:
        hypothesis: The unified rule from the synthesis agent.
        key_insight: The key insight about what property of the input controls the output.
        train_pairs: All training pairs, used as worked examples.
        test_input: The single test grid to predict the output for.
        chain: Optional pre-built LCEL chain (for testing). If None, builds from model.
        model: OpenAI model name (used only when chain is None).

    Returns:
        GeneratorResult with step_by_step_trace and predicted_output.
        Returns fallback on failure.
    """
    if chain is None:
        chain = build_generator_chain(ChatOpenAI(model=model, temperature=0))

    try:
        return await chain.ainvoke({
            "unified_hypothesis": hypothesis,
            "key_insight":        key_insight,
            "train_pairs":        format_demo_pairs(train_pairs),
            "test_input":         grid_to_str(test_input),
        })
    except Exception as e:
        logger.error("Generator chain failed: %s", e)
        return _FALLBACK
