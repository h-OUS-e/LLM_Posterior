"""Hypothesis synthesis agent for ARC-AGI LOO evaluation.

After all N LOO splits complete (with optional refinement), this agent receives:
  - All N train pairs (ground truth)
  - Compressed summaries of all N refined analyst hypotheses

It synthesises a single unified hypothesis.
Test input execution is delegated to the separate generator_agent.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.compress import compress_hypothesis
from src.data_loader import Task
from src.hypothesis_agent import format_demo_pairs

logger = logging.getLogger(__name__)


SYNTH_SYSTEM = """\
You are an ARC-AGI puzzle solver. Your job: find ONE general rule that transforms \
every input grid into its output grid.

## How to think
1. OBSERVE: Compare each input→output pair cell by cell. Note what changed, what stayed.
2. ABSTRACT: Find the single rule that explains ALL pairs. The rule must be a function \
of the INPUT only — it cannot reference the output.
3. EXECUTE: Apply your rule to each input mechanically, row by row. Show your work.
4. VERIFY: Check your predicted output against the ground truth. Every cell must match.

## Critical rules
- Your hypothesis must be an EXECUTABLE PROCEDURE: given any input grid, someone \
could follow your steps and produce the output without seeing it first.
- If your rule says "fill color X here" — you must specify EXACTLY which cells, \
using positions relative to the input (e.g., "row i, col j where i+j < distance \
from corner").
- DO NOT describe the output ("there are stripes"). DESCRIBE THE PROCESS that \
generates it ("for each colored corner, fill cells where manhattan distance to \
corner equals k on alternating rows").
- NEVER copy output grids from memory. Derive every cell from your rule + the input.
- If two colors behave differently, explain what INPUT property controls each.

## About the analyst summaries
- Each analyst saw N-1 training pairs and predicted the held-out pair.
- Analysts were then shown the ground truth and refined. They may have patched \
their rule to fit one specific pair without finding the general pattern.
- Use analyst summaries as HINTS, not answers. The training pairs are ground truth.
- LEARN FROM ANALYST MISTAKES: If an analyst claims 100% accuracy but their rule \
is vague (e.g., "alternating stripes"), that means they memorized the output \
without understanding the generative rule. Your job is to find what they missed. \
Ask: what SPECIFIC spatial logic would a programmer need to reproduce this output? \
The analysts' failure to be precise is signal — it tells you the rule has geometric \
detail they couldn't articulate.
- If ALL analysts converge on similar vague language, they are all describing the \
EFFECT, not the CAUSE. Dig deeper into the grids.
        
## If you receive correction feedback (retry)
You are seeing this because your previous hypothesis failed on some training pairs. \
DO NOT just patch the wrong cells — that means your rule is wrong, not your execution.
- First, diagnose WHY your previous rule produced the wrong cells. What case did it \
miss? What assumption was incorrect?
- Then fix the RULE itself so the correct output follows naturally from the updated rule.
- Common failure modes: rule works for one grid size but not another, rule handles \
one anchor position but not a different one, rule describes the pattern but not \
the generative process (e.g., "stripes" vs "fill row k if distance to anchor is even").

"""

SYNTH_HUMAN_TEMPLATE = """\
## All training pairs (ground truth):
{train_pairs}

## Analyst summaries (treat as hints — may be overfit):
{analyst_summaries}

## Your task — follow these steps exactly:

### Step 1: Analyst audit → `analyst_gaps`
Read each analyst summary. For each one, ask: is this rule EXECUTABLE — could \
a programmer implement it from this description alone? If not, note what is \
missing (which cells exactly? what condition? relative to what?). If all analysts \
say something similar but vague (e.g., "alternating stripes"), they found the \
EFFECT but not the CAUSE. Summarize what the analysts got right directionally \
and what specific spatial detail they failed to articulate.

### Step 2: Cell-level observation → `key_transformations`
For each training pair, list the concrete cell-level changes between input and \
output. What cells changed? What pattern do the changed cells form? Reference \
specific row/col indices and colors. This is your evidence — be precise.

### Step 3: Cross-pair comparison → `reasoning`
What is CONSTANT across all pairs? What VARIES? For the parts that vary, \
what input property controls them (grid size, color position, distance, etc.)? \
Explain the logic that connects input properties to the transformation.

### Step 4: Executable hypothesis → `unified_hypothesis`
State your rule as a step-by-step executable procedure. Someone with only the \
input grid and your procedure must be able to produce the output. Include:
- How to locate anchor points (colored cells in input)
- How to determine which cells get filled with which color
- The exact geometric/arithmetic condition for each cell
Do NOT describe what the output looks like — describe the PROCESS that generates it.

{correction_feedback}\
"""

_SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYNTH_SYSTEM),
    ("human", SYNTH_HUMAN_TEMPLATE),
])


class MasterOutput(BaseModel):
    """Structured output for the synthesis agent."""

    analyst_gaps: str = Field(
        description="What the analysts got right directionally, and what "
        "specific spatial/geometric detail they failed to articulate."
    )
    key_transformations: str = Field(
        description="Concrete cell-level changes observed across train pairs. "
        "Reference specific row/col indices and colors."
    )
    reasoning: str = Field(
        description="What is constant vs variable across pairs, and what "
        "input property controls the variable parts."
    )
    unified_hypothesis: str = Field(
        description="Step-by-step executable procedure: anchor detection, "
        "fill conditions, geometric/arithmetic rules per cell."
    )


_FALLBACK = MasterOutput(
    analyst_gaps="",
    key_transformations="",
    reasoning="",
    unified_hypothesis="",
)


def _format_analyst_summaries(loo_results: list) -> str:
    """Format all LOO results as compressed (4-line) analyst summaries."""
    return "\n\n".join(compress_hypothesis(i, r) for i, r in enumerate(loo_results, 1))


def _format_analyst_summaries_full(loo_results: list) -> str:
    """Format all LOO results with complete hypothesis text and reasoning."""
    parts = []
    for i, r in enumerate(loo_results, 1):
        lines = [
            f"Analyst {i} (held out train[{r.held_out_index}]):",
            f"  Hypothesis: {r.hypothesis}",
            f"  Reasoning:  {r.reasoning}",
            f"  Exact match: {'yes' if r.exact_match else 'no'} ({r.cell_accuracy:.0%})",
        ]
        if r.refined_hypothesis is not None:
            lines += [
                f"  → Refined hypothesis: {r.refined_hypothesis}",
                f"  → Refined reasoning:  {r.refined_reasoning}",
                f"  → Refined exact match: {'yes' if r.refined_exact_match else 'no'} ({(r.refined_cell_accuracy or 0):.0%})",
            ]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)



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
    compress: bool = True,
) -> MasterOutput:
    """Synthesise a unified hypothesis from all LOO refined hypotheses.

    Args:
        task: The ARC task (provides all train pairs).
        loo_results: All HypothesisResult objects from the LOO pass (with optional
            refinement fields populated).
        chain: Optional pre-built LCEL chain (for testing). If None, builds from model.
        model: OpenAI model name (used only when chain is None).

    Returns:
        MasterOutput with the unified hypothesis. Returns fallback on failure.
    """
    if chain is None:
        chain = build_synthesis_chain(ChatOpenAI(model=model, temperature=0))

    train_pairs_str = format_demo_pairs(task.train)
    fmt = _format_analyst_summaries if compress else _format_analyst_summaries_full
    analyst_summaries = fmt(loo_results)

    try:
        return await chain.ainvoke({
            "train_pairs":       train_pairs_str,
            "analyst_summaries": analyst_summaries,
            "correction_feedback": "",
        })
    except Exception as e:
        logger.error("Synthesis chain failed: %s", e)
        return _FALLBACK
