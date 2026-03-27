"""MCTS tree node — shared by both Vanilla and Gumbel implementations."""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any, List


class MCTSNode:
    """A node in the MCTS tree.

    Attributes
    ----------
    state : Any
        Environment-specific state representation.
    parent : Optional[MCTSNode]
        Parent node; None for the root.
    action : Optional[Any]
        The action that led to this node from its parent.
    prior : float
        Prior probability of selecting this node (used by Gumbel-MCTS).
    children : Dict[Any, MCTSNode]
        Map from action to child node.
    visit_count : int
        Number of times this node has been visited.
    value_sum : float
        Sum of backup values accumulated through this node.
    is_terminal : bool
        Whether the state is terminal.
    """

    __slots__ = (
        "state",
        "parent",
        "action",
        "prior",
        "children",
        "visit_count",
        "value_sum",
        "is_terminal",
        "is_expanded",
        "_completed_q_value",
    )

    def __init__(
        self,
        state: Any,
        parent: Optional[MCTSNode] = None,
        action: Optional[Any] = None,
        prior: float = 1.0,
        is_terminal: bool = False,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[Any, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.is_terminal: bool = is_terminal
        self.is_expanded: bool = False
        # Gumbel-MCTS uses a "completed-Q" estimate
        self._completed_q_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Value accessors
    # ------------------------------------------------------------------

    @property
    def value(self) -> float:
        """Mean value estimate (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def completed_q(self, default: float = 0.0) -> float:
        """Completed-Q value used by Gumbel-MCTS backups."""
        if self._completed_q_value is not None:
            return self._completed_q_value
        return self.value if self.visit_count > 0 else default

    # ------------------------------------------------------------------
    # UCB score (vanilla MCTS)
    # ------------------------------------------------------------------

    def ucb_score(self, parent_visit_count: int, c_puct: float = 1.414) -> float:
        """UCB1-PUCT score used by vanilla MCTS selection."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.value
        exploration = c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Tree utilities
    # ------------------------------------------------------------------

    def best_child(self, c_puct: float = 1.414) -> MCTSNode:
        """Return the child with the highest UCB score."""
        return max(self.children.values(), key=lambda n: n.ucb_score(self.visit_count, c_puct))

    def most_visited_child(self) -> MCTSNode:
        """Return the child with the highest visit count (play-out action)."""
        return max(self.children.values(), key=lambda n: n.visit_count)

    def child_list(self) -> List[MCTSNode]:
        """Return all children as an ordered list."""
        return list(self.children.values())

    def action_list(self) -> List[Any]:
        """Return the list of actions leading to each child."""
        return list(self.children.keys())

    def depth(self) -> int:
        """Return the depth of this node in the tree (root = 0)."""
        d = 0
        node = self
        while node.parent is not None:
            node = node.parent
            d += 1
        return d

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, visits={self.visit_count}, "
            f"value={self.value:.4f}, children={len(self.children)})"
        )
