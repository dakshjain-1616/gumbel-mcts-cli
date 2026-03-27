"""Test 1 — Search problem input → Gumbel MCTS result."""
import pytest
import numpy as np
from gumbel_mcts_cli_high import GumbelMCTS, GridWorldEnv, MaxTreeEnv
from gumbel_mcts_cli_high.node import MCTSNode


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gridworld_env():
    return GridWorldEnv(size=6, max_steps=100, seed=0)


@pytest.fixture
def maxtree_env():
    return MaxTreeEnv(depth=4, branching=4, seed=0)


@pytest.fixture
def gumbel():
    return GumbelMCTS(n_simulations=80, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Search returns a valid action on GridWorld
# ─────────────────────────────────────────────────────────────────────────────

class TestGumbelMCTSSearch:

    def test_returns_action(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        action = gumbel.search(gridworld_env, state)
        assert action is not None, "search() must return an action"

    def test_action_is_legal(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        action = gumbel.search(gridworld_env, state)
        assert action in gridworld_env.actions(), f"action {action} not in legal set"

    def test_search_with_stats_returns_triple(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        result = gumbel.search_with_stats(gridworld_env, state)
        assert len(result) == 3, "search_with_stats must return (action, root, elapsed)"

    def test_elapsed_is_positive(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        _, _, elapsed = gumbel.search_with_stats(gridworld_env, state)
        assert elapsed > 0, "elapsed time must be positive"

    def test_root_has_children(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        _, root, _ = gumbel.search_with_stats(gridworld_env, state)
        assert len(root.children) > 0, "root must have children after search"

    def test_root_visit_count(self, gridworld_env, gumbel):
        state = gridworld_env.reset()
        _, root, _ = gumbel.search_with_stats(gridworld_env, state)
        # Root itself is not visited during simulation, but children are
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits > 0, "at least one child must be visited"

    def test_maxtree_action_is_legal(self, maxtree_env, gumbel):
        state = maxtree_env.reset()
        action = gumbel.search(maxtree_env, state)
        assert action in maxtree_env.actions(), f"action {action} not legal in MaxTree"

    def test_maxtree_root_value_is_finite(self, maxtree_env, gumbel):
        state = maxtree_env.reset()
        _, root, _ = gumbel.search_with_stats(maxtree_env, state)
        assert np.isfinite(root.value), "root value must be finite"

    def test_deterministic_with_same_seed(self, gridworld_env):
        """Same seed should yield the same action."""
        g1 = GumbelMCTS(n_simulations=60, seed=1)
        g2 = GumbelMCTS(n_simulations=60, seed=1)
        state = gridworld_env.reset()
        a1 = g1.search(gridworld_env, state)
        env2 = GridWorldEnv(size=6, max_steps=100, seed=0)
        state2 = env2.reset()
        a2 = g2.search(env2, state2)
        assert a1 == a2, "same seed must yield same action"

    def test_more_sims_increases_root_value(self):
        """More simulations should generally improve the value estimate."""
        env_few  = MaxTreeEnv(depth=4, branching=4, seed=9)
        env_many = MaxTreeEnv(depth=4, branching=4, seed=9)
        g_few  = GumbelMCTS(n_simulations=20,  seed=42)
        g_many = GumbelMCTS(n_simulations=200, seed=42)
        _, root_few,  _ = g_few.search_with_stats(env_few,  env_few.reset())
        _, root_many, _ = g_many.search_with_stats(env_many, env_many.reset())
        # More sims → more children visited
        few_visits  = sum(c.visit_count for c in root_few.children.values())
        many_visits = sum(c.visit_count for c in root_many.children.values())
        assert many_visits > few_visits, "more sims must produce more visits"
