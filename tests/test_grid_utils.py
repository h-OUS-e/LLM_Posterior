"""Tests for grid_utils module."""
import pytest
from src.grid_utils import grid_to_str, grids_match, grid_diff


def test_grid_to_str_basic():
    grid = [[0, 1, 2], [3, 0, 1]]
    result = grid_to_str(grid)
    assert result == "0 1 2\n3 0 1"


def test_grid_to_str_single_row():
    assert grid_to_str([[5, 3]]) == "5 3"


def test_grids_match_equal():
    g = [[1, 2], [3, 4]]
    assert grids_match(g, g) is True


def test_grids_match_different_values():
    assert grids_match([[1, 0]], [[0, 1]]) is False


def test_grids_match_different_sizes():
    assert grids_match([[1, 2]], [[1, 2, 3]]) is False


def test_grid_diff_perfect_match():
    g = [[1, 2], [3, 4]]
    result = grid_diff(g, g)
    assert result["cell_accuracy"] == 1.0
    assert result["mismatches"] == []
    assert result["size_match"] is True


def test_grid_diff_partial_mismatch():
    predicted = [[1, 0], [3, 4]]
    actual    = [[1, 2], [3, 4]]
    result = grid_diff(predicted, actual)
    assert result["cell_accuracy"] == pytest.approx(0.75)
    assert result["mismatches"] == [(0, 1)]
    assert result["size_match"] is True


def test_grid_diff_size_mismatch():
    result = grid_diff([[1, 2]], [[1, 2, 3]])
    assert result["size_match"] is False
    assert result["cell_accuracy"] == 0.0
