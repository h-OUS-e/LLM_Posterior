"""Grid utilities for ARC-AGI: display, comparison, diff, and visualization."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ARC_COLORS = {
    0: "#000000", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40",
    4: "#FFDC00", 5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B",
    8: "#7FDBFF", 9: "#870C25",
}


def grid_to_str(grid: list[list[int]]) -> str:
    """Convert a grid to a space-separated string, one row per line.

    Args:
        grid: 2D list of integers 0-9.

    Returns:
        Multi-line string representation.
    """
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def grids_match(a: list[list[int]], b: list[list[int]]) -> bool:
    """Exact equality check between two grids, handling size mismatches gracefully.

    Args:
        a: First grid.
        b: Second grid.

    Returns:
        True if grids are identical in size and values.
    """
    if len(a) != len(b):
        return False
    if any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        return False
    return all(ca == cb for ra, rb in zip(a, b) for ca, cb in zip(ra, rb))


def grid_diff(predicted: list[list[int]], actual: list[list[int]]) -> dict:
    """Compute per-cell accuracy and mismatch positions between predicted and actual grids.

    Args:
        predicted: Predicted grid from LLM.
        actual: Ground-truth grid.

    Returns:
        Dict with keys:
            - size_match (bool): Whether grids have same dimensions.
            - cell_accuracy (float): Fraction of cells that match (0.0 if size mismatch).
            - mismatches (list[tuple[int,int]]): (row, col) positions of mismatched cells.
    """
    rows_p, rows_a = len(predicted), len(actual)
    cols_p = len(predicted[0]) if predicted else 0
    cols_a = len(actual[0]) if actual else 0

    if rows_p != rows_a or cols_p != cols_a:
        return {"size_match": False, "cell_accuracy": 0.0, "mismatches": []}

    mismatches = [
        (r, c)
        for r in range(rows_p)
        for c in range(cols_p)
        if predicted[r][c] != actual[r][c]
    ]
    total = rows_p * cols_p
    accuracy = (total - len(mismatches)) / total if total > 0 else 0.0
    return {"size_match": True, "cell_accuracy": accuracy, "mismatches": mismatches}


def visualize_grid(grid: list[list[int]], title: str = "") -> None:
    """Display a grid using ARC's standard 10-color palette.

    Args:
        grid: 2D list of integers 0-9.
        title: Optional plot title.
    """
    rows = len(grid)
    cols = max(len(row) for row in grid) if grid else 0
    fig, ax = plt.subplots(figsize=(cols * 0.6 + 1, rows * 0.6 + 1))
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = ARC_COLORS.get(val, "#000000")
            rect = mpatches.FancyBboxPatch(
                (c, rows - r - 1), 1, 1,
                boxstyle="square,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=0.5,
            )
            ax.add_patch(rect)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
