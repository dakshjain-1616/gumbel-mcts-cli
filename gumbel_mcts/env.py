"""Search environments for benchmarking MCTS variants.

Three environments:
  GridWorldEnv   — 2-D grid navigation to a goal (continuous reward)
  MaxTreeEnv     — synthetic binary tree where the agent must find the max leaf
  SequenceEnv    — combinatorial sequence optimization (beam-search style)
"""
from __future__ import annotations
import os
import numpy as np
from typing import Tuple, List, Any, Optional


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a seeded NumPy default Generator."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

class GridWorldEnv:
    """NxN grid. Agent starts at (0,0), goal at (N-1, N-1).

    Actions: 0=up, 1=down, 2=left, 3=right
    Reward: -0.01 per step, +1.0 on reaching goal, -0.5 on hitting a wall.
    Episode terminates when goal is reached or max_steps exceeded.
    """

    GRID_SIZE_DEFAULT = int(os.getenv("GRID_SIZE", "8"))
    MAX_STEPS_DEFAULT = int(os.getenv("MAX_STEPS", "200"))

    def __init__(
        self,
        size: int = GRID_SIZE_DEFAULT,
        max_steps: int = MAX_STEPS_DEFAULT,
        seed: Optional[int] = None,
        obstacle_density: float = 0.0,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.rng = _rng(seed)
        # Optionally place obstacles (cells that block movement)
        self.obstacles: set = set()
        if obstacle_density > 0:
            n_obs = int(obstacle_density * size * size)
            for _ in range(n_obs):
                r, c = self.rng.integers(0, size, size=2)
                if (r, c) not in {(0, 0), (size - 1, size - 1)}:
                    self.obstacles.add((r, c))
        self.goal = (size - 1, size - 1)
        self._state: Tuple[int, int] = (0, 0)
        self._steps: int = 0

    # State encoding
    @property
    def state(self) -> Tuple[int, int]:
        return (self._state[0], self._state[1], self._steps)

    def reset(self) -> Tuple[int, int, int]:
        self._state = (0, 0)
        self._steps = 0
        return self.state

    def actions(self) -> List[int]:
        return [0, 1, 2, 3]

    def step(self, action: int) -> Tuple[Any, float, bool]:
        r, c = self._state
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r + 1, c
        elif action == 2:
            nr, nc = r, c - 1
        else:
            nr, nc = r, c + 1

        self._steps += 1
        # Boundary / obstacle check
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or (nr, nc) in self.obstacles:
            reward = -0.1
        else:
            self._state = (nr, nc)
            if self._state == self.goal:
                reward = 1.0
            else:
                reward = -0.01

        terminal = (self._state == self.goal) or (self._steps >= self.max_steps)
        return self.state, reward, terminal

    def clone(self) -> GridWorldEnv:
        """Return a deep copy for tree simulation."""
        env = GridWorldEnv.__new__(GridWorldEnv)
        env.size = self.size
        env.max_steps = self.max_steps
        env.rng = np.random.default_rng()
        env.obstacles = self.obstacles  # immutable for benchmarks
        env.goal = self.goal
        env._state = self._state
        env._steps = self._steps
        return env

    def is_terminal(self, state: Any) -> bool:
        r, c, steps = state
        return (r, c) == self.goal or steps >= self.max_steps

    def reward(self, state: Any) -> float:
        r, c, _ = state
        return 1.0 if (r, c) == self.goal else 0.0

    def random_rollout(self, env: GridWorldEnv, depth: int = 30) -> float:
        """Fast random rollout from current env state."""
        total = 0.0
        discount = 1.0
        gamma = float(os.getenv("ROLLOUT_GAMMA", "0.99"))
        for _ in range(depth):
            action = int(env.rng.integers(0, 4))
            _, r, done = env.step(action)
            total += discount * r
            discount *= gamma
            if done:
                break
        return total


# ---------------------------------------------------------------------------
# MaxTree — synthetic evaluation environment
# ---------------------------------------------------------------------------

class MaxTreeEnv:
    """Binary tree where each leaf has a score. Agent navigates root→leaf.

    The environment is deterministic: tree values are fixed at construction.
    This is the canonical benchmark for showing Gumbel-MCTS efficiency:
    it finds the max-value leaf with far fewer simulations than vanilla UCT.
    """

    def __init__(
        self,
        depth: int = int(os.getenv("TREE_DEPTH", "6")),
        branching: int = int(os.getenv("TREE_BRANCHING", "8")),
        seed: Optional[int] = None,
    ) -> None:
        self.depth = depth
        self.branching = branching
        rng = _rng(seed)
        # Pre-generate leaf values: shape (branching^depth,)
        n_leaves = branching ** depth
        self.leaf_values: np.ndarray = rng.standard_normal(n_leaves)
        self._current_path: List[int] = []

    # State = tuple of actions taken from root
    @property
    def state(self) -> tuple:
        return tuple(self._current_path)

    def reset(self) -> tuple:
        self._current_path = []
        return self.state

    def actions(self) -> List[int]:
        if len(self._current_path) >= self.depth:
            return []
        return list(range(self.branching))

    def step(self, action: int) -> Tuple[tuple, float, bool]:
        self._current_path.append(action)
        terminal = len(self._current_path) >= self.depth
        if terminal:
            reward = float(self._leaf_value(self._current_path))
        else:
            reward = 0.0
        return self.state, reward, terminal

    def _leaf_value(self, path: List[int]) -> float:
        idx = 0
        for a in path:
            idx = idx * self.branching + a
        return float(self.leaf_values[idx % len(self.leaf_values)])

    @property
    def optimal_value(self) -> float:
        return float(np.max(self.leaf_values))

    def clone(self) -> MaxTreeEnv:
        env = MaxTreeEnv.__new__(MaxTreeEnv)
        env.depth = self.depth
        env.branching = self.branching
        env.leaf_values = self.leaf_values  # read-only, safe to share
        env._current_path = list(self._current_path)
        return env

    def is_terminal(self, state: tuple) -> bool:
        return len(state) >= self.depth

    def reward(self, state: tuple) -> float:
        """Return leaf value for a terminal state."""
        return self._leaf_value(list(state))

    def random_rollout(self, env: MaxTreeEnv, depth: int = 999) -> float:
        rng = np.random.default_rng()
        while not env.is_terminal(env.state):
            acts = env.actions()
            if not acts:
                break
            env.step(int(rng.integers(0, len(acts))))
        return env._leaf_value(list(env.state))


# ---------------------------------------------------------------------------
# SequenceEnv — combinatorial sequence optimization
# ---------------------------------------------------------------------------

class SequenceEnv:
    """Combinatorial sequence optimization environment.

    The agent builds a sequence of tokens one step at a time, then a score
    function evaluates the complete sequence.  This models problems such as:

      * Code / text generation (discrete token selection)
      * Molecular design (building SMILES strings character by character)
      * Combinatorial optimisation (ordered selection from a vocabulary)

    By default the scoring function sums a fixed random embedding for each
    (position, token) pair — deterministic given ``seed``, so optimal values
    are reproducible and comparable across algorithms.

    Parameters
    ----------
    vocab_size : int
        Vocabulary / branching factor.  Each step the agent picks one of
        0 … vocab_size−1.
    depth : int
        Target sequence length (episode length).
    score_fn : callable, optional
        ``score_fn(sequence: List[int]) -> float``.  If omitted a fixed
        positional embedding table is used.
    seed : int, optional
        Seed for the default score table.
    """

    def __init__(
        self,
        vocab_size: int = int(os.getenv("SEQ_VOCAB_SIZE", "8")),
        depth: int = int(os.getenv("SEQ_DEPTH", "5")),
        score_fn=None,
        seed: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.depth = depth
        rng = _rng(seed)
        # Fixed positional embedding: shape (depth, vocab_size)
        self._embed: np.ndarray = rng.standard_normal((depth, vocab_size))
        self._score_fn = score_fn
        self._sequence: List[int] = []

    @property
    def state(self) -> tuple:
        return tuple(self._sequence)

    def reset(self) -> tuple:
        self._sequence = []
        return self.state

    def actions(self) -> List[int]:
        if len(self._sequence) >= self.depth:
            return []
        return list(range(self.vocab_size))

    def step(self, action: int) -> Tuple[tuple, float, bool]:
        self._sequence.append(action)
        terminal = len(self._sequence) >= self.depth
        reward = float(self._score(self._sequence)) if terminal else 0.0
        return self.state, reward, terminal

    def _score(self, seq: List[int]) -> float:
        if self._score_fn is not None:
            return float(self._score_fn(seq))
        # Positional dot-product: sum embed[pos, token] over all positions
        return float(sum(self._embed[pos, tok] for pos, tok in enumerate(seq)))

    @property
    def optimal_value(self) -> float:
        """Upper bound: greedily pick the best token at each position."""
        return float(sum(float(self._embed[pos].max()) for pos in range(self.depth)))

    def clone(self) -> "SequenceEnv":
        env = SequenceEnv.__new__(SequenceEnv)
        env.vocab_size = self.vocab_size
        env.depth = self.depth
        env._embed = self._embed  # read-only, safe to share
        env._score_fn = self._score_fn
        env._sequence = list(self._sequence)
        return env

    def is_terminal(self, state: tuple) -> bool:
        return len(state) >= self.depth

    def reward(self, state: tuple) -> float:
        return self._score(list(state)) if self.is_terminal(state) else 0.0

    def random_rollout(self, env: "SequenceEnv", depth: int = 999) -> float:
        rng = np.random.default_rng()
        while not env.is_terminal(env.state):
            acts = env.actions()
            if not acts:
                break
            env.step(int(rng.integers(0, len(acts))))
        return env._score(list(env.state))
