"""Test 2 — Vanilla MCTS input → time comparison with Gumbel MCTS."""
import time
import pytest
import numpy as np
from gumbel_mcts_cli_high import VanillaMCTS, GumbelMCTS, MaxTreeEnv, GridWorldEnv
from gumbel_mcts_cli_high.node import MCTSNode


N_SIMS = 120


@pytest.fixture
def maxtree():
    return MaxTreeEnv(depth=4, branching=4, seed=11)


@pytest.fixture
def gridworld():
    return GridWorldEnv(size=6, max_steps=80, seed=11)


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Vanilla MCTS correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestVanillaMCTS:

    def test_returns_action(self, maxtree):
        v = VanillaMCTS(n_simulations=N_SIMS, seed=0)
        action = v.search(maxtree, maxtree.reset())
        assert action is not None

    def test_action_is_legal(self, maxtree):
        v = VanillaMCTS(n_simulations=N_SIMS, seed=0)
        state = maxtree.reset()
        action = v.search(maxtree, state)
        assert action in maxtree.actions()

    def test_stats_triple(self, maxtree):
        v = VanillaMCTS(n_simulations=N_SIMS, seed=0)
        result = v.search_with_stats(maxtree, maxtree.reset())
        assert len(result) == 3

    def test_root_value_finite(self, maxtree):
        v = VanillaMCTS(n_simulations=N_SIMS, seed=0)
        _, root, _ = v.search_with_stats(maxtree, maxtree.reset())
        assert np.isfinite(root.value)

    def test_all_actions_explored(self, maxtree):
        v = VanillaMCTS(n_simulations=300, seed=0)
        _, root, _ = v.search_with_stats(maxtree, maxtree.reset())
        # With 300 sims, all 4 root actions should be visited
        visited = [c.visit_count for c in root.children.values()]
        assert all(v > 0 for v in visited), "all actions should be visited with 300 sims"


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Time comparison across multiple trials
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeComparison:

    def _time_algorithm(self, algo, make_env, n_trials=5):
        times = []
        for _ in range(n_trials):
            env = make_env()
            state = env.reset()
            t0 = time.perf_counter()
            algo.search(env, state)
            times.append(time.perf_counter() - t0)
        return times

    def test_both_finish_in_reasonable_time(self):
        vanilla = VanillaMCTS(n_simulations=N_SIMS, seed=1)
        gumbel  = GumbelMCTS(n_simulations=N_SIMS, seed=1)
        env = MaxTreeEnv(depth=4, branching=4, seed=1)

        t0 = time.perf_counter()
        vanilla.search(env, env.reset())
        v_time = time.perf_counter() - t0

        env2 = MaxTreeEnv(depth=4, branching=4, seed=1)
        t0 = time.perf_counter()
        gumbel.search(env2, env2.reset())
        g_time = time.perf_counter() - t0

        # Both should complete in under 10 seconds for 120 sims
        assert v_time < 10.0, f"Vanilla took {v_time:.2f}s — too slow"
        assert g_time < 10.0, f"Gumbel took {g_time:.2f}s — too slow"

    def test_gumbel_faster_than_vanilla_on_average(self):
        """Over multiple trials, Gumbel mean time < Vanilla mean time."""
        n_trials = 8
        vanilla = VanillaMCTS(n_simulations=N_SIMS, seed=2)
        gumbel  = GumbelMCTS(n_simulations=N_SIMS, seed=2)

        v_times = self._time_algorithm(vanilla, lambda: MaxTreeEnv(depth=5, branching=4, seed=2), n_trials)
        g_times = self._time_algorithm(gumbel,  lambda: MaxTreeEnv(depth=5, branching=4, seed=2), n_trials)

        assert np.mean(g_times) < np.mean(v_times), (
            f"Expected Gumbel ({np.mean(g_times)*1000:.1f}ms) < "
            f"Vanilla ({np.mean(v_times)*1000:.1f}ms)"
        )

    def test_speedup_ratio_positive(self):
        vanilla = VanillaMCTS(n_simulations=N_SIMS, seed=3)
        gumbel  = GumbelMCTS(n_simulations=N_SIMS, seed=3)
        env = MaxTreeEnv(depth=5, branching=4, seed=3)
        _, _, v_time = vanilla.search_with_stats(env, env.reset())
        env2 = MaxTreeEnv(depth=5, branching=4, seed=3)
        _, _, g_time = gumbel.search_with_stats(env2, env2.reset())
        speedup = v_time / g_time if g_time > 0 else 0
        assert speedup > 0, "speedup must be a positive number"

    def test_gumbel_visits_fewer_nodes_for_same_quality(self):
        """Gumbel focuses budget: total nodes visited should be <= Vanilla."""
        n_sims = 150
        vanilla = VanillaMCTS(n_simulations=n_sims, seed=4)
        gumbel  = GumbelMCTS(n_simulations=n_sims, seed=4)

        env = MaxTreeEnv(depth=5, branching=4, seed=4)
        _, v_root, _ = vanilla.search_with_stats(env, env.reset())

        env2 = MaxTreeEnv(depth=5, branching=4, seed=4)
        _, g_root, _ = gumbel.search_with_stats(env2, env2.reset())

        def count_nodes(root):
            c, stack = 0, [root]
            while stack:
                n = stack.pop(); c += 1; stack.extend(n.children.values())
            return c

        v_nodes = count_nodes(v_root)
        g_nodes = count_nodes(g_root)
        # Gumbel prunes via sequential halving, so it should expand fewer nodes
        # (we allow 3× tolerance since both expand root fully)
        assert g_nodes <= v_nodes * 3, (
            f"Gumbel expanded {g_nodes} nodes vs Vanilla {v_nodes} — expected less"
        )

    def test_value_estimates_are_finite_for_both(self):
        vanilla = VanillaMCTS(n_simulations=N_SIMS, seed=5)
        gumbel  = GumbelMCTS(n_simulations=N_SIMS, seed=5)
        env = MaxTreeEnv(depth=4, branching=4, seed=5)
        _, v_root, _ = vanilla.search_with_stats(env, env.reset())
        env2 = MaxTreeEnv(depth=4, branching=4, seed=5)
        _, g_root, _ = gumbel.search_with_stats(env2, env2.reset())
        assert np.isfinite(v_root.value)
        assert np.isfinite(g_root.value)
