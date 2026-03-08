"""LangChain hypothesis generation for ARC-AGI tasks.

Uses LCEL chain: ChatPromptTemplate | llm.with_structured_output(HypothesisOutput).
Retry is handled declaratively via .with_retry(stop_after_attempt=2).
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.data_loader import IOPair
from src.grid_utils import grid_to_str

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are solving an ARC-AGI puzzle. You will be shown input/output grid pairs \
that demonstrate a transformation rule. Grids use integers 0-9 representing colors.

Analyze the demonstration pairs carefully, formulate a clear hypothesis, \
apply it to the test input, and return your answer as structured JSON.\
"""

HUMAN_TEMPLATE = """\
## Demonstration pairs:
{demo_pairs}

## Test input:
{test_input}\
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


class HypothesisOutput(BaseModel):
    """Structured LLM response for a single hypothesis generation call."""

    hypothesis: str
    reasoning: str
    predicted_output: list[list[int]]
    confidence: float


def format_demo_pairs(demo_pairs: list[IOPair]) -> str:
    """Format demo pairs as numbered sections for the prompt."""
    sections = []
    for i, pair in enumerate(demo_pairs, 1):
        sections.append(
            f"Pair {i}:\nInput:\n{grid_to_str(pair.input)}\n\nOutput:\n{grid_to_str(pair.output)}"
        )
    return "\n\n".join(sections)


def build_prompt(demo_pairs: list[IOPair], test_input: list[list[int]]) -> str:
    """Return the formatted human message string (used by dry-run and notebook).

    Args:
        demo_pairs: Demonstration input/output pairs.
        test_input: The grid to predict the output for.

    Returns:
        Formatted prompt string.
    """
    return HUMAN_TEMPLATE.format(
        demo_pairs=format_demo_pairs(demo_pairs),
        test_input=grid_to_str(test_input),
    )


def build_chain(llm: ChatOpenAI) -> Runnable:
    """Construct the LCEL hypothesis chain: prompt | structured_llm.

    Args:
        llm: Configured ChatOpenAI instance.

    Returns:
        Runnable that accepts {demo_pairs, test_input} and returns HypothesisOutput.
    """
    structured_llm = llm.with_structured_output(HypothesisOutput)
    return (_PROMPT | structured_llm).with_retry(stop_after_attempt=2)


_FALLBACK = HypothesisOutput(hypothesis="", reasoning="", predicted_output=[], confidence=0.0)


async def generate_hypothesis(
    demo_pairs: list[IOPair],
    test_input: list[list[int]],
    chain: Runnable | None = None,
    model: str = "gpt-4o",
) -> HypothesisOutput:
    """Generate a hypothesis for the given demo pairs and test input.

    Args:
        demo_pairs: Demonstration pairs used as few-shot examples.
        test_input: The grid the model must predict the output for.
        chain: Optional pre-built LCEL chain (for testing). If None, builds from model name.
        model: OpenAI model name (used only when chain is None).

    Returns:
        HypothesisOutput. On failure after retries, returns empty output with confidence=0.
    """
    if chain is None:
        chain = build_chain(ChatOpenAI(model=model, temperature=0))

    try:
        result = await chain.ainvoke({
            "demo_pairs": format_demo_pairs(demo_pairs),
            "test_input": grid_to_str(test_input),
        })
        return result
    except Exception as e:
        logger.error("Chain failed after retries: %s", e)
        return _FALLBACK
