"""Tests for compress module."""
from src.compress import _first_sentence, compress_hypothesis
from src.leave_one_out import HypothesisResult


def _make_result(
    hypothesis="rule A. extra sentence.",
    reasoning="some reasoning",
    refined_hypothesis=None,
    refined_reasoning=None,
    refined_exact_match=None,
    refined_cell_accuracy=None,
    exact_match=True,
    cell_accuracy=1.0,
    held_out_index=0,
):
    return HypothesisResult(
        hypothesis=hypothesis,
        reasoning=reasoning,
        predicted_output=[[1, 0]],
        confidence=0.9,
        held_out_index=held_out_index,
        is_test=False,
        actual_output=[[1, 0]],
        exact_match=exact_match,
        cell_accuracy=cell_accuracy,
        refined_hypothesis=refined_hypothesis,
        refined_reasoning=refined_reasoning,
        refined_exact_match=refined_exact_match,
        refined_cell_accuracy=refined_cell_accuracy,
    )


# --- _first_sentence ---

def test_first_sentence_stops_at_period():
    assert _first_sentence("flip bits. more detail here.") == "flip bits."


def test_first_sentence_stops_at_exclamation():
    assert _first_sentence("invert! extra") == "invert!"


def test_first_sentence_stops_at_question():
    assert _first_sentence("is it inverted? yes") == "is it inverted?"


def test_first_sentence_no_punctuation_returns_full():
    assert _first_sentence("no punctuation here") == "no punctuation here"


def test_first_sentence_empty():
    assert _first_sentence("") == ""


# --- compress_hypothesis ---

def test_compress_contains_analyst_label():
    r = _make_result()
    out = compress_hypothesis(1, r)
    assert "Analyst 1" in out
    assert "train[0]" in out


def test_compress_uses_refined_hypothesis_when_available():
    r = _make_result(
        hypothesis="wrong rule. ignore.",
        refined_hypothesis="correct rule. also ignore.",
    )
    out = compress_hypothesis(2, r)
    assert "correct rule." in out
    assert "wrong rule." in out  # original is also shown


def test_compress_uses_original_when_no_refinement():
    r = _make_result(hypothesis="only rule. rest ignored.")
    out = compress_hypothesis(1, r)
    assert out.count("only rule.") == 2  # same text for both original and refined


def test_compress_shows_correct_accuracy_refined():
    r = _make_result(refined_cell_accuracy=0.875, cell_accuracy=0.5)
    out = compress_hypothesis(1, r)
    assert "87.5%" in out


def test_compress_shows_cell_accuracy_when_no_refinement():
    r = _make_result(cell_accuracy=0.5)
    out = compress_hypothesis(1, r)
    assert "50.0%" in out


def test_compress_shows_exact_match_yes():
    r = _make_result(refined_exact_match=True)
    out = compress_hypothesis(1, r)
    assert "yes" in out


def test_compress_shows_exact_match_no():
    r = _make_result(exact_match=False, refined_exact_match=None)
    out = compress_hypothesis(1, r)
    assert "no" in out
