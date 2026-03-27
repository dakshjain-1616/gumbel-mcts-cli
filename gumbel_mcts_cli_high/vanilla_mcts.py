"""Vanilla UCT-based Monte Carlo Tree Search.

Uses standard UCB1-PUCT selection, random rollouts for evaluation, and
standard backup. This is the baseline against which Gumbel-MCTS is compared.
"""
from __future__ import annotations
import os
import time
import numpy as np
from typing import Any, Optional, List

from .node import MCTSNode


class VanillaMCTS:
    """Standard Monte Carlo Tree Search (UCT variant).

    Parameters
    ----------
    n_simulations : int
        Number of search simulations to run per call to ``search()``.
    c_puct : float
        Exploration constant for UCB score.
    rollout_depth : int
        Maximum depth for random rollouts.
    seed : Optional[int]
        Random seed.
    """

    def __init__(
        self,
        n_simulations: int = int(os.getenv("MCTS_N_SIM", "200")),
        c_puct: float = float(os.getenv("MCTS_C_PUCT", "1.414")),
        rollout_depth: int = int(os.getenv("ROLLOUT_DEPTH", "50")),
        seed: Optional[int] = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, env: Any, state: Any) -> Any:
        """Run MCTS from *state* and return the best action.

        Parameters
        ----------
        env :
            Environment object — must expose ``.clone()``, ``.actions()``,
            ``.step(action)``, ``.is_terminal(state)``, ``.random_rollout(env)``.
        state :
            The root state to search from.
        """
        root = MCTSNode(state=state)

        for _ in range(self.n_simulations):
            node, sim_env = self._select(root, env)
            value = self._evaluate(node, sim_env)
            self._backup(node, value)

        # Return action of most-visited child
        if not root.children:
            return None
        return root.most_visited_child().action

    def search_with_stats(self, env: Any, state: Any):
        """Like search() but returns (action, root_node, elapsed_seconds)."""
        root = MCTSNode(state=state)
        t0 = time.perf_counter()

        for _ in range(self.n_simulations):
            node, sim_env = self._select(root, env)
            value = self._evaluate(node, sim_env)
            self._backup(node, value)

        elapsed = time.perf_counter() - t0
        best_action = root.most_visited_child().action if root.children else None
        return best_action, root, elapsed

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode, env: Any):
        """Selection: descend using UCB until an unvisited / leaf node."""
        node = root
        sim_env = env.clone()

        while node.is_expanded and not node.is_terminal:
            node = node.best_child(self.c_puct)
            sim_env.step(node.action)

        return node, sim_env

    def _expand(self, node: MCTSNode, env: Any) -> None:
        """Expansion: add all legal children to *node*."""
        actions = env.actions()
        if not actions:
            node.is_terminal = True
            return
        for action in actions:
            child_env = env.clone()
            child_state, _, terminal = child_env.step(action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=action,
                prior=1.0 / len(actions),
                is_terminal=terminal,
            )
            node.children[action] = child
        node.is_expanded = True

    def _evaluate(self, node: MCTSNode, env: Any) -> float:
        """Expand node then run a random rollout from a random child."""
        if node.is_terminal:
            return env.reward(node.state) if hasattr(env, "reward") else 0.0

        if not node.is_expanded:
            self._expand(node, env)

        if not node.children:
            return 0.0

        # Pick a random unexplored child for rollout
        unvisited = [c for c in node.children.values() if c.visit_count == 0]
        if unvisited:
            child = unvisited[int(self.rng.integers(0, len(unvisited)))]
        else:
            child = node.best_child(self.c_puct)

        rollout_env = env.clone()
        rollout_env.step(child.action)
        return env.random_rollout(rollout_env, self.rollout_depth)

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Backup: propagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent
