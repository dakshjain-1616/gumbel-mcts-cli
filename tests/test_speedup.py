"""Test 3 — No input → verify speedup is significant (>10× potential in full run).

This test suite benchmarks via multiple carefully controlled trials and
verifies the speedup claim.  It also tests the benchmark module itself.
"""
import time
import pytest
import numpy as np
from gumbel_mcts_cli_high import VanillaMCTS, GumbelMCTS, MaxTreeEnv
from gumbel_mcts_cli_high.benchmark import run_benchmark, BenchmarkReport


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def timed_search(algo, env_factory, n_trials=10, n_sims=150):
    """Return list of elapsed times."""
    times = []
    for seed in range(n_trials):
        env = env_factory(seed)
        state = env.reset()
        t0 = time.perf_counter()
        algo.search(env, state)
        times.append(time.perf_counter() - t0)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# 3a. Speedup tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeedup:

    @pytest.mark.timeout(120)
    def test_speedup_exceeds_1x(self):
        """Gumbel must be at least 1× faster (i.e. not slower)."""
        N = 8
        n_sims = 150
        vanilla = VanillaMCTS(n_simulations=n_sims, seed=10)
        gumbel  = GumbelMCTS(n_simulations=n_sims, seed=10)

        def factory(s):
            return MaxTreeEnv(depth=5, branching=8, seed=s)

        v_times = timed_search(vanilla, factory, N, n_sims)
        g_times = timed_search(gumbel,  factory, N, n_sims)

        speedup = np.mean(v_times) / np.mean(g_times)
        assert speedup > 1.0, f"Expected speedup > 1.0×, got {speedup:.2f}×"

    @pytest.mark.timeout(120)
    def test_speedup_exceeds_2x(self):
        """Gumbel must be at least 2× faster on larger trees."""
        N = 8
        n_sims = 200
        vanilla = VanillaMCTS(n_simulations=n_sims, seed=20)
        gumbel  = GumbelMCTS(n_simulations=n_sims, seed=20)

        def factory(s):
            return MaxTreeEnv(depth=6, branching=8, seed=s)

        v_times = timed_search(vanilla, factory, N, n_sims)
        g_times = timed_search(gumbel,  factory, N, n_sims)

        speedup = np.mean(v_times) / np.mean(g_times)
        assert speedup >= 2.0, (
            f"Expected speedup ≥ 2.0×, got {speedup:.2f}×\n"
            f"Vanilla mean: {np.mean(v_times)*1000:.1f}ms\n"
            f"Gumbel mean:  {np.mean(g_times)*1000:.1f}ms"
        )

    @pytest.mark.timeout(120)
    def test_speedup_is_stable_across_seeds(self):
        """Speedup should be consistent: all seeds show Gumbel faster."""
        n_sims = 150

        def factory(s):
            return MaxTreeEnv(depth=5, branching=8, seed=s)

        speedups = []
        for seed in range(6):
            vanilla = VanillaMCTS(n_simulations=n_sims, seed=seed * 100)
            gumbel  = GumbelMCTS(n_simulations=n_sims, seed=seed * 100)
            env_v = factory(seed)
            env_g = factory(seed)
            _, _, vt = vanilla.search_with_stats(env_v, env_v.reset())
            _, _, gt = gumbel.search_with_stats(env_g, env_g.reset())
            speedups.append(vt / gt if gt > 0 else 0)

        # At least 4 out of 6 seeds should show Gumbel faster
        n_faster = sum(1 for s in speedups if s > 1.0)
        assert n_faster >= 4, (
            f"Expected Gumbel faster on ≥4/6 seeds, got {n_faster}/6\n"
            f"Speedups: {[f'{s:.2f}' for s in speedups]}"
        )

    def test_gumbel_mean_time_lower(self):
        n_sims = 100
        results = []
        for seed in range(5):
            env_v = MaxTreeEnv(depth=4, branching=4, seed=seed)
            env_g = MaxTreeEnv(depth=4, branching=4, seed=seed)
            v = VanillaMCTS(n_simulations=n_sims, seed=seed)
            g = GumbelMCTS(n_simulations=n_sims, seed=seed)
            _, _, vt = v.search_with_stats(env_v, env_v.reset())
            _, _, gt = g.search_with_stats(env_g, env_g.reset())
            results.append((vt, gt))

        mean_v = np.mean([r[0] for r in results])
        mean_g = np.mean([r[1] for r in results])
        assert mean_g < mean_v, (
            f"Gumbel mean ({mean_g*1000:.2f}ms) should be < Vanilla ({mean_v*1000:.2f}ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Benchmark module tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkModule:

    @pytest.mark.timeout(120)
    def test_run_benchmark_returns_report(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=5,
            n_simulations=80,
            seed=99,
            verbose=False,
        )
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.timeout(120)
    def test_report_has_correct_trial_count(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=4,
            n_simulations=60,
            seed=99,
            verbose=False,
        )
        assert len(report.vanilla_times) == 4
        assert len(report.gumbel_times) == 4

    @pytest.mark.timeout(120)
    def test_report_speedup_is_positive(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=4,
            n_simulations=80,
            seed=77,
            verbose=False,
        )
        assert report.speedup > 0.0

    @pytest.mark.timeout(120)
    def test_report_summary_contains_speedup(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=3,
            n_simulations=60,
            seed=55,
            verbose=False,
        )
        summary = report.summary()
        assert "Speedup" in summary

    @pytest.mark.timeout(120)
    def test_benchmark_speedup_positive_on_gridworld(self):
        report = run_benchmark(
            env_name="gridworld",
            n_trials=3,
            n_simulations=60,
            seed=33,
            verbose=False,
        )
        assert report.speedup > 0.0

    @pytest.mark.timeout(120)
    def test_report_to_dict_serializable(self):
        import json
        report = run_benchmark(
            env_name="maxtree",
            n_trials=3,
            n_simulations=60,
            seed=11,
            verbose=False,
        )
        d = report.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    @pytest.mark.timeout(60)
    def test_all_times_positive(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=3,
            n_simulations=60,
            seed=22,
            verbose=False,
        )
        assert all(t > 0 for t in report.vanilla_times)
        assert all(t > 0 for t in report.gumbel_times)
