"""Hypothesis synthesis agent for ARC-AGI LOO evaluation.

After all N LOO splits complete (with optional refinement), this agent receives:
  - All N train pairs (ground truth)
  - Compressed summaries of all N refined analyst hypotheses

It synthesises a single unified hypothesis and verifies it against all train pairs.
If any train pair is wrong, one retry is made with cell-level correction feedback.
Test input execution is delegated to the separate generator_agent.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.compress import compress_hypothesis
from src.data_loader import Task
from src.grid_utils import grid_diff, grid_to_str, grids_match
from src.hypothesis_agent import format_demo_pairs

logger = logging.getLogger(__name__)


SYNTH_SYSTEM = """\
You are synthesizing multiple analyst hypotheses about an ARC-AGI puzzle into one correct rule.

Each analyst saw a different subset of examples and proposed a hypothesis. They often \
overfit — memorizing specific block positions or patterns from their subset rather than \
finding the general rule.

CRITICAL: When analysts disagree about specifics (e.g., WHICH positions, WHICH cells), \
that disagreement means those specifics are NOT fixed — they must DEPEND on the input. \
Your job is to find what property of the input determines the variable parts.

You must output a predicted grid for EVERY train pair. Your train_predictions will be \
checked programmatically. If any train pair prediction is wrong, your hypothesis is wrong.\
"""

SYNTH_HUMAN_TEMPLATE = """\
## All training pairs:
{train_pairs}

## Analyst summaries:
{analyst_summaries}

## Instructions:
1. State what the analysts agree on
2. State where they disagree and explain WHY — they saw different inputs that produced \
different outputs, so they may have overfit to their specific subset
3. Identify what property of the INPUT controls the variable part (key insight)
4. State your unified hypothesis as one clear rule
5. Apply your rule to EACH training pair's input — include the predicted output grid

{correction_feedback}\
"""

_SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTH_SYSTEM),
    ("human", SYNTH_HUMAN_TEMPLATE),
])


class MasterOutput(BaseModel):
    """Structured output for the synthesis agent."""

    agreements: str
    disagreements: str
    key_insight: str
    unified_hypothesis: str
    train_predictions: list[list[list[int]]]  # one grid per train pair
    # Populated by synthesize_hypotheses after programmatic verification:
    train_exact_matches: list[bool] = []
    master_train_accuracy: float = 0.0


_FALLBACK = MasterOutput(
    agreements="",
    disagreements="",
    key_insight="",
    unified_hypothesis="",
    train_predictions=[],
)


def _format_analyst_summaries(loo_results: list) -> str:
    """Format all LOO results as compressed analyst summaries."""
    return "\n\n".join(compress_hypothesis(i, r) for i, r in enumerate(loo_results, 1))


def _format_correction_feedback(
    result: MasterOutput,
    task: Task,
    train_exact: list[bool],
) -> str:
    """Build a correction prompt section listing which train pairs were wrong."""
    lines = ["Your hypothesis failed verification against the training data:"]
    for i, (ok, pred, pair) in enumerate(
        zip(train_exact, result.train_predictions, task.train)
    ):
        if ok:
            lines.append(f"  train[{i}]: ✓ exact match")
        else:
            diff = grid_diff(pred, pair.output)
            acc = round(diff["cell_accuracy"] * 100, 1)
            lines.append(
                f"  train[{i}]: ✗ ({acc}% cell accuracy)"
                f"\n    Your prediction:\n{grid_to_str(pred)}"
                f"\n    Actual output:\n{grid_to_str(pair.output)}"
            )
    lines.append(
        "\nFix your hypothesis so it correctly predicts ALL training pairs. "
        "Return the same JSON format with corrected predictions."
    )
    return "\n".join(lines)


def build_synthesis_chain(llm: ChatOpenAI) -> Runnable:
    """Construct the LCEL synthesis chain: prompt | structured_llm."""
    return (_SYNTH_PROMPT | llm.with_structured_output(MasterOutput)).with_retry(
        stop_after_attempt=2
    )


async def synthesize_hypotheses(
    task: Task,
    loo_results: list,
    chain: Runnable | None = None,
    model: str = "gpt-4.1-mini",
) -> MasterOutput:
    """Synthesise a unified hypothesis from all LOO refined hypotheses.

    Verifies train_predictions against ground truth and retries once with
    cell-level correction feedback if any train pair is wrong. Returns the
    attempt with higher mean train accuracy.

    Args:
        task: The ARC task (provides all train pairs).
        loo_results: All HypothesisResult objects from the LOO pass (with optional
            refinement fields populated).
        chain: Optional pre-built LCEL chain (for testing). If None, builds from model.
        model: OpenAI model name (used only when chain is None).

    Returns:
        MasterOutput with train_exact_matches and master_train_accuracy set.
        Returns fallback on failure.
    """
    if chain is None:
        chain = build_synthesis_chain(ChatOpenAI(model=model, temperature=0))

    train_pairs_str = format_demo_pairs(task.train)
    analyst_summaries = _format_analyst_summaries(loo_results)
    base_input = {
        "train_pairs":       train_pairs_str,
        "analyst_summaries": analyst_summaries,
        "correction_feedback": "",
    }

    def _verify(result: MasterOutput) -> list[bool]:
        if not result.train_predictions:
            return []
        return [
            grids_match(pred, pair.output)
            for pred, pair in zip(result.train_predictions, task.train)
        ]

    def _accuracy(exact: list[bool]) -> float:
        return sum(exact) / len(exact) if exact else 0.0

    try:
        result = await chain.ainvoke(base_input)
        train_exact = _verify(result)

        if all(train_exact):
            return result.model_copy(update={
                "train_exact_matches": train_exact,
                "master_train_accuracy": 1.0,
            })

        # One retry with correction feedback
        correction = _format_correction_feedback(result, task, train_exact)
        result2 = await chain.ainvoke({**base_input, "correction_feedback": correction})
        train_exact2 = _verify(result2)

        # Keep whichever attempt has higher mean train accuracy
        if _accuracy(train_exact2) >= _accuracy(train_exact):
            best, best_exact = result2, train_exact2
        else:
            best, best_exact = result, train_exact

        return best.model_copy(update={
            "train_exact_matches": best_exact,
            "master_train_accuracy": _accuracy(best_exact),
        })

    except Exception as e:
        logger.error("Synthesis chain failed: %s", e)
        return _FALLBACK
