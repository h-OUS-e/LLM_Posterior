"""Self-correction refinement agent for ARC-AGI LOO predictions.

Given an initial (wrong) prediction plus the actual correct output, produces a
revised hypothesis + corrected prediction. The LLM sees: its previous hypothesis,
its predicted grid, a diff grid (x = wrong cell), the actual output, and cell
accuracy. It is asked to diagnose what was wrong and correct its hypothesis.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.hypothesis_agent import HypothesisOutput
from src.grid_utils import grid_to_str, grid_diff

logger = logging.getLogger(__name__)

REFINE_SYSTEM = """\
You are re-analyzing an ARC-AGI puzzle after seeing where your first attempt \
went wrong. You will be shown demonstration pairs, your previous hypothesis, \
your predicted output, and the actual correct output. Use this to diagnose \
exactly where your hypothesis was wrong, then produce a corrected hypothesis \
and prediction. Your corrected hypothesis MUST explain ALL demo pairs, not \
just the test input. Note: output grid dimensions may differ from input \
grid dimensions.\
"""

REFINE_HUMAN_TEMPLATE = """\
## Original demonstration pairs:
{demo_pairs}

## Test input:
{test_input}

## Your previous hypothesis:
{original_hypothesis}

## Your predicted output:
{predicted_grid}

## Cells you got wrong (0-indexed; 'x' = wrong cell):
{diff_grid}

## The actual correct output:
{actual_grid}

Your prediction had {cell_accuracy_pct}% cell accuracy.

Compare your prediction to the actual output cell-by-cell. Then:
1. State specifically what was wrong with your original hypothesis
2. Formulate a corrected hypothesis that explains the transformation rule
3. Mentally apply your corrected hypothesis to EACH demonstration pair's input \
and verify it produces the correct output shown above. If it doesn't, revise \
your hypothesis until it does.
4. Apply the corrected hypothesis to the test input
5. Return updated JSON (same format as before)\
"""

_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REFINE_SYSTEM),
    ("human", REFINE_HUMAN_TEMPLATE),
])

_FALLBACK = HypothesisOutput(hypothesis="", reasoning="", predicted_output=[], confidence=0.0)


def _make_diff_grid_str(predicted: list[list[int]], actual: list[list[int]]) -> str:
    """Return predicted grid as a string with 'x' at every cell that was wrong.

    If grids differ in size, returns a size-mismatch note instead.
    """
    if len(predicted) != len(actual) or any(
        len(pr) != len(ar) for pr, ar in zip(predicted, actual)
    ):
        p_cols = len(predicted[0]) if predicted else 0
        a_cols = len(actual[0]) if actual else 0
        return f"(size mismatch: predicted {len(predicted)}\u00d7{p_cols}, actual {len(actual)}\u00d7{a_cols})"

    rows = []
    for pred_row, act_row in zip(predicted, actual):
        cells = [str(pv) if pv == av else "x" for pv, av in zip(pred_row, act_row)]
        rows.append(" ".join(cells))
    return "\n".join(rows)


def build_refinement_chain(llm: ChatOpenAI) -> Runnable:
    """Construct the LCEL refinement chain: prompt | structured_llm."""
    return (_REFINE_PROMPT | llm.with_structured_output(HypothesisOutput)).with_retry(
        stop_after_attempt=2
    )


async def refine_hypothesis(
    demo_pairs_str: str,
    test_input: list[list[int]],
    history: list[HypothesisOutput],
    predicted: list[list[int]],
    actual: list[list[int]],
    chain: Runnable | None = None,
    model: str = "gpt-4.1-mini",
) -> HypothesisOutput:
    """Call the refinement chain for one wrong LOO prediction.

    Args:
        demo_pairs_str: Pre-formatted demo pairs string (from format_demo_pairs).
        test_input: The grid being predicted.
        history: All HypothesisOutput attempts so far; last entry is the current hypothesis.
        predicted: Latest predicted output grid.
        actual: Ground-truth output — revealed to the LLM to enable diagnosis.
        chain: Optional pre-built chain (for testing).
        model: Model name used only when chain is None.

    Returns:
        Refined HypothesisOutput, or fallback on failure.
    """
    if chain is None:
        chain = build_refinement_chain(ChatOpenAI(model=model, temperature=0))

    original_hypothesis = history[-1].hypothesis if history else ""
    cell_acc_pct = round(grid_diff(predicted, actual)["cell_accuracy"] * 100, 1)

    try:
        return await chain.ainvoke({
            "demo_pairs":          demo_pairs_str,
            "test_input":          grid_to_str(test_input),
            "original_hypothesis": original_hypothesis,
            "predicted_grid":      grid_to_str(predicted),
            "diff_grid":           _make_diff_grid_str(predicted, actual),
            "actual_grid":         grid_to_str(actual),
            "cell_accuracy_pct":   cell_acc_pct,
        })
    except Exception as e:
        logger.error("Refinement chain failed: %s", e)
        return _FALLBACK
