"""Compact analyst report formatter for the synthesis agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.leave_one_out import HypothesisResult


def _first_sentence(text: str) -> str:
    """Return only the first sentence (up to the first '.', '!', or '?')."""
    for delim in ".!?":
        idx = text.find(delim)
        if idx != -1:
            return text[: idx + 1].strip()
    return text.strip()


def compress_hypothesis(i: int, result: HypothesisResult) -> str:
    """Return a compact ~4-line analyst summary.

    Args:
        i: 1-based analyst index.
        result: HypothesisResult with optional refined_* fields.

    Returns:
        Multi-line string: label, original rule, refined rule, exact-match status.
    """
    original = _first_sentence(result.hypothesis)
    refined_hyp = result.refined_hypothesis or result.hypothesis
    refined = _first_sentence(refined_hyp)
    match = (
        result.refined_exact_match
        if result.refined_exact_match is not None
        else result.exact_match
    )
    acc = round(
        (
            result.refined_cell_accuracy
            if result.refined_cell_accuracy is not None
            else result.cell_accuracy
        )
        * 100,
        1,
    )
    lines = [
        f"Analyst {i} (held out train[{result.held_out_index}], refined acc: {acc}%):",
        f"  Original rule: {original}",
        f"  Refined rule:  {refined}",
        f"  Exact match after refinement: {'yes' if match else 'no'}",
    ]
    return "\n".join(lines)
