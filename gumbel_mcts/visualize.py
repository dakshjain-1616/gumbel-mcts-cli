"""ASCII tree visualization for MCTS search trees.

Usage::

    from gumbel_mcts.visualize import print_tree, format_action_table
    action, root, elapsed = gumbel.search_with_stats(env, state)
    print(print_tree(root, max_depth=3))
    print(format_action_table(root))
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

from .node import MCTSNode


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def print_tree(
    root: MCTSNode,
    max_depth: int = int(os.getenv("VIZ_MAX_DEPTH", "3")),
    max_children: int = int(os.getenv("VIZ_MAX_CHILDREN", "5")),
    min_visits: int = 0,
    value_lo: float = -2.0,
    value_hi: float = 2.0,
) -> str:
    """Return an ASCII representation of the MCTS search tree.

    Parameters
    ----------
    root : MCTSNode
        Root of the tree returned by ``search_with_stats``.
    max_depth : int
        Maximum depth to render (env var ``VIZ_MAX_DEPTH``, default 3).
    max_children : int
        Maximum number of children to show per node (by visit count).
    min_visits : int
        Skip children with fewer than this many visits.
    value_lo / value_hi : float
        Range used to scale the value bar (default –2 … +2).

    Returns
    -------
    str
        Multi-line ASCII tree string.
    """
    bar_width = int(os.getenv("VIZ_BAR_WIDTH", "12"))
    lines: list[str] = []

    total_visits = sum(c.visit_count for c in root.children.values())
    root_q = f"Q={root.value:.3f}" if root.visit_count > 0 else "Q=n/a"
    lines.append(
        f"ROOT  total_child_visits={total_visits}  {root_q}"
        f"  children={len(root.children)}"
    )

    top_children = _top_children(root, max_children, min_visits)
    for i, child in enumerate(top_children):
        is_last = i == len(top_children) - 1
        _render_node(
            child,
            prefix="",
            is_last=is_last,
            depth=1,
            max_depth=max_depth,
            max_children=max_children,
            min_visits=min_visits,
            bar_width=bar_width,
            value_lo=value_lo,
            value_hi=value_hi,
            lines=lines,
        )

    return "\n".join(lines)


def format_action_table(
    root: MCTSNode,
    action_names: Optional[Dict] = None,
) -> str:
    """Format a table of all root actions with their statistics.

    Parameters
    ----------
    root : MCTSNode
        Root returned by ``search_with_stats``.
    action_names : dict, optional
        Mapping from action id to display string (e.g. ``{0: "UP", 1: "DOWN"}``).

    Returns
    -------
    str
        Formatted table string.
    """
    if not root.children:
        return "  (no children to display)"

    total_visits = sum(c.visit_count for c in root.children.values()) or 1

    header = (
        f"  {'Action':>8}  {'Visits':>7}  {'Share%':>7}  "
        f"{'Q-value':>9}  {'±StdErr':>8}  {'Prior':>7}"
    )
    sep = "  " + "─" * (len(header) - 2)
    lines = [header, sep]

    for action, child in sorted(
        root.children.items(), key=lambda kv: -kv[1].visit_count
    ):
        name = (
            action_names.get(action, str(action)) if action_names else str(action)
        )
        share = 100.0 * child.visit_count / total_visits
        if child.visit_count > 1:
            stderr = 1.0 / math.sqrt(child.visit_count)
            stderr_str = f"{stderr:.4f}"
        else:
            stderr_str = "  —   "
        lines.append(
            f"  {name:>8}  {child.visit_count:>7}  {share:>6.1f}%  "
            f"{child.value:>9.4f}  {stderr_str:>8}  {child.prior:>7.4f}"
        )

    lines.append(sep)
    lines.append(f"  {'TOTAL':>8}  {total_visits:>7}")
    return "\n".join(lines)


def format_confidence_summary(
    values: list[float],
    label: str = "Values",
    confidence: float = 0.95,
) -> str:
    """Return a one-line summary with mean ± margin for a list of floats.

    Uses normal approximation (z=1.96 for 95% CI).
    """
    import numpy as np

    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return f"{label}: (no data)"
    mean = float(np.mean(arr))
    if n == 1:
        return f"{label}: {mean:.4f}  (n=1, CI unavailable)"
    sem = float(np.std(arr, ddof=1) / math.sqrt(n))
    z = 1.96  # 95% CI
    margin = z * sem
    return (
        f"{label}: {mean:.4f} ± {margin:.4f}  "
        f"(95% CI [{mean - margin:.4f}, {mean + margin:.4f}], n={n})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_children(
    node: MCTSNode, max_children: int, min_visits: int
) -> list[MCTSNode]:
    """Return up to max_children children sorted by visit count descending."""
    children = [
        c for c in node.children.values() if c.visit_count >= min_visits
    ]
    children.sort(key=lambda c: c.visit_count, reverse=True)
    return children[:max_children]


def _make_bar(value: float, lo: float, hi: float, width: int) -> str:
    """Render a Unicode block progress bar scaled between lo and hi."""
    span = hi - lo
    if span <= 0:
        norm = 0.5
    else:
        norm = min(1.0, max(0.0, (value - lo) / span))
    filled = int(round(norm * width))
    return "█" * filled + "░" * (width - filled)


def _render_node(
    node: MCTSNode,
    prefix: str,
    is_last: bool,
    depth: int,
    max_depth: int,
    max_children: int,
    min_visits: int,
    bar_width: int,
    value_lo: float,
    value_hi: float,
    lines: list[str],
) -> None:
    """Recursively append one tree node line (and its subtree) to lines."""
    connector = "└── " if is_last else "├── "
    bar = _make_bar(node.value, value_lo, value_hi, bar_width)

    action_str = f"a={node.action}" if node.action is not None else "root"
    if node.visit_count > 1:
        stderr = 1.0 / math.sqrt(node.visit_count)
        ci_str = f" ±{stderr:.3f}"
    else:
        ci_str = ""

    terminal_tag = " [T]" if node.is_terminal else ""
    line = (
        f"{prefix}{connector}[{bar}] {action_str}"
        f"  n={node.visit_count}  Q={node.value:.3f}{ci_str}{terminal_tag}"
    )
    lines.append(line)

    if depth >= max_depth or not node.children:
        return

    child_prefix = prefix + ("    " if is_last else "│   ")
    top = _top_children(node, max_children, min_visits)
    for i, child in enumerate(top):
        _render_node(
            child,
            prefix=child_prefix,
            is_last=(i == len(top) - 1),
            depth=depth + 1,
            max_depth=max_depth,
            max_children=max_children,
            min_visits=min_visits,
            bar_width=bar_width,
            value_lo=value_lo,
            value_hi=value_hi,
            lines=lines,
        )
