"""Tests for new features added in v1.1.0.

Covers:
  - SequenceEnv correctness
  - GumbelMCTS.search_anytime generator
  - gumbel_mcts.visualize (print_tree, format_action_table)
  - BenchmarkReport confidence intervals
  - benchmark_mcts.py CLI flags (--dry-run, --format, --compare-sims, --tree)
  - run_benchmark with env_name="sequence"
"""
from __future__ import annotations

import json
import os
import sys
import math
import tempfile
import pytest
import numpy as np

# ── ensure project root is importable when running from any directory ─────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from gumbel_mcts_cli_high import GumbelMCTS, VanillaMCTS, MaxTreeEnv, GridWorldEnv, SequenceEnv
from gumbel_mcts_cli_high import print_tree, format_action_table
from gumbel_mcts_cli_high.benchmark import run_benchmark, BenchmarkReport, ConfidenceInterval
from gumbel_mcts_cli_high.visualize import format_confidence_summary


# ─────────────────────────────────────────────────────────────────────────────
# SequenceEnv
# ─────────────────────────────────────────────────────────────────────────────

class TestSequenceEnv:

    def test_reset_returns_empty_tuple(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        state = env.reset()
        assert state == ()

    def test_actions_at_start(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        env.reset()
        assert env.actions() == [0, 1, 2, 3]

    def test_actions_empty_at_depth(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        env.reset()
        for a in [0, 1, 2]:
            env.step(a)
        assert env.actions() == []

    def test_step_returns_correct_types(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        env.reset()
        state, reward, terminal = env.step(0)
        assert isinstance(state, tuple)
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)

    def test_terminal_at_depth(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        env.reset()
        _, _, t1 = env.step(0)
        _, _, t2 = env.step(1)
        _, _, t3 = env.step(2)
        assert not t1
        assert not t2
        assert t3, "must be terminal after depth steps"

    def test_reward_only_at_terminal(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        env.reset()
        _, r1, _ = env.step(0)
        _, r2, _ = env.step(1)
        _, r_terminal, _ = env.step(2)
        assert r1 == 0.0
        assert r2 == 0.0
        # terminal reward depends on embedding; just check it's a finite float
        assert np.isfinite(r_terminal)

    def test_clone_is_independent(self):
        env = SequenceEnv(vocab_size=4, depth=4, seed=1)
        env.reset()
        env.step(0)
        clone = env.clone()
        clone.step(1)
        # Original should not be affected
        assert len(env._sequence) == 1
        assert len(clone._sequence) == 2

    def test_optimal_value_gte_random(self):
        env = SequenceEnv(vocab_size=8, depth=5, seed=7)
        env.reset()
        opt = env.optimal_value
        # Random policy expected value is sum of per-position means ≈ 0
        # Optimal must be ≥ the mean (sum of row maxima ≥ sum of row means)
        row_means = float(env._embed.mean())
        assert opt >= row_means * env.depth - 1e-9

    def test_is_terminal(self):
        env = SequenceEnv(vocab_size=4, depth=2, seed=0)
        assert not env.is_terminal(())
        assert not env.is_terminal((0,))
        assert env.is_terminal((0, 1))

    def test_reward_method(self):
        env = SequenceEnv(vocab_size=4, depth=2, seed=0)
        env.reset()
        # reward for non-terminal state should be 0
        assert env.reward(()) == 0.0
        assert env.reward((0,)) == 0.0
        # terminal state
        r = env.reward((0, 1))
        assert np.isfinite(r)

    def test_random_rollout_returns_float(self):
        env = SequenceEnv(vocab_size=4, depth=4, seed=2)
        env.reset()
        clone = env.clone()
        val = env.random_rollout(clone)
        assert np.isfinite(val)

    def test_gumbel_search_on_sequence_env(self):
        env = SequenceEnv(vocab_size=4, depth=4, seed=3)
        gumbel = GumbelMCTS(n_simulations=60, seed=0)
        state = env.reset()
        action = gumbel.search(env, state)
        assert action in env.actions()

    def test_custom_score_fn(self):
        """Custom score_fn is called at terminal states."""
        called = []

        def my_score(seq):
            called.append(list(seq))
            return float(sum(seq))

        env = SequenceEnv(vocab_size=3, depth=2, score_fn=my_score, seed=0)
        env.reset()
        env.step(1)
        _, r, terminal = env.step(2)
        assert terminal
        assert r == 3.0  # 1 + 2
        assert len(called) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# GumbelMCTS.search_anytime
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchAnytime:

    @pytest.fixture
    def env(self):
        return MaxTreeEnv(depth=4, branching=4, seed=42)

    @pytest.fixture
    def gumbel(self):
        return GumbelMCTS(n_simulations=80, seed=1)

    def test_yields_at_least_one_snapshot(self, env, gumbel):
        snapshots = list(gumbel.search_anytime(env, env.reset()))
        assert len(snapshots) >= 1

    def test_snapshot_keys(self, env, gumbel):
        snap = next(iter(gumbel.search_anytime(env, env.reset())))
        for key in ("action", "round", "n_candidates", "elapsed_sec", "root", "scores"):
            assert key in snap, f"Missing key: {key}"

    def test_action_is_legal(self, env, gumbel):
        for snap in gumbel.search_anytime(env, env.reset()):
            assert snap["action"] in env.actions()

    def test_elapsed_increases(self, env, gumbel):
        times = [s["elapsed_sec"] for s in gumbel.search_anytime(env, env.reset())]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], "Elapsed must be non-decreasing"

    def test_candidates_halve(self, env, gumbel):
        counts = [s["n_candidates"] for s in gumbel.search_anytime(env, env.reset())]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], "n_candidates must be non-increasing"

    def test_early_stop_gives_valid_action(self, env, gumbel):
        """Stopping after the first round still yields a valid action."""
        gen = gumbel.search_anytime(env, env.reset())
        snap = next(gen)
        assert snap["action"] in env.actions()

    def test_round_index_increments(self, env, gumbel):
        rounds = [s["round"] for s in gumbel.search_anytime(env, env.reset())]
        assert rounds == list(range(len(rounds)))

    def test_scores_list_matches_n_candidates(self, env, gumbel):
        for snap in gumbel.search_anytime(env, env.reset()):
            assert len(snap["scores"]) == snap["n_candidates"]

    def test_works_on_gridworld(self):
        env = GridWorldEnv(size=5, max_steps=50, seed=0)
        gumbel = GumbelMCTS(n_simulations=50, seed=0)
        snaps = list(gumbel.search_anytime(env, env.reset()))
        assert len(snaps) >= 1
        assert snaps[-1]["action"] in env.actions()

    def test_works_on_sequence_env(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        gumbel = GumbelMCTS(n_simulations=50, seed=0)
        snaps = list(gumbel.search_anytime(env, env.reset()))
        assert len(snaps) >= 1
        assert snaps[-1]["action"] in env.actions()


# ─────────────────────────────────────────────────────────────────────────────
# Tree visualization
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualize:

    @pytest.fixture
    def root(self):
        env = MaxTreeEnv(depth=4, branching=4, seed=0)
        gumbel = GumbelMCTS(n_simulations=80, seed=0)
        _, root, _ = gumbel.search_with_stats(env, env.reset())
        return root

    def test_print_tree_returns_string(self, root):
        result = print_tree(root)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_print_tree_contains_root(self, root):
        result = print_tree(root)
        assert "ROOT" in result

    def test_print_tree_respects_max_depth(self, root):
        shallow = print_tree(root, max_depth=1)
        deep = print_tree(root, max_depth=3)
        assert len(deep) >= len(shallow)

    def test_print_tree_contains_q_value(self, root):
        result = print_tree(root)
        assert "Q=" in result

    def test_format_action_table_returns_string(self, root):
        result = format_action_table(root)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_action_table_has_header(self, root):
        result = format_action_table(root)
        assert "Action" in result
        assert "Visits" in result
        assert "Q-value" in result

    def test_format_action_table_with_action_names(self, root):
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        result = format_action_table(root, action_names=action_names)
        assert any(name in result for name in action_names.values())

    def test_format_action_table_no_children(self):
        from gumbel_mcts_cli_high.node import MCTSNode
        empty_root = MCTSNode(state=None)
        result = format_action_table(empty_root)
        assert "no children" in result.lower()

    def test_format_confidence_summary(self):
        values = [0.5, 0.6, 0.55, 0.52, 0.58]
        summary = format_confidence_summary(values, label="TestMetric")
        assert "TestMetric" in summary
        assert "95%" in summary
        assert "CI" in summary

    def test_format_confidence_summary_single_value(self):
        summary = format_confidence_summary([1.0], label="X")
        assert "X" in summary
        assert "n=1" in summary

    def test_visualize_on_sequence_env(self):
        env = SequenceEnv(vocab_size=4, depth=3, seed=0)
        gumbel = GumbelMCTS(n_simulations=60, seed=0)
        _, root, _ = gumbel.search_with_stats(env, env.reset())
        result = print_tree(root, max_depth=2)
        assert "ROOT" in result


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkReport enhancements
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkEnhancements:

    @pytest.fixture(scope="class")
    def report(self):
        return run_benchmark(
            env_name="maxtree",
            n_trials=5,
            n_simulations=60,
            seed=0,
            verbose=False,
        )

    def test_confidence_intervals_exist(self, report):
        assert report.vanilla_time_ci is not None
        assert report.gumbel_time_ci is not None
        assert report.vanilla_value_ci is not None
        assert report.gumbel_value_ci is not None

    def test_ci_mean_matches_list_mean(self, report):
        expected = float(np.mean(report.vanilla_times))
        assert abs(report.vanilla_time_ci.mean - expected) < 1e-12

    def test_ci_margin_positive(self, report):
        assert report.vanilla_time_ci.margin >= 0
        assert report.gumbel_time_ci.margin >= 0

    def test_ci_low_lt_high(self, report):
        ci = report.vanilla_time_ci
        assert ci.low <= ci.high

    def test_summary_contains_ci(self, report):
        summary = report.summary()
        assert "95% CI" in summary

    def test_summary_contains_percentiles(self, report):
        summary = report.summary()
        assert "pct" in summary.lower() or "10th" in summary or "percentile" in summary.lower()

    def test_to_dict_includes_ci(self, report):
        d = report.to_dict()
        assert "vanilla_time_ci" in d
        assert d["vanilla_time_ci"] is not None
        assert "mean" in d["vanilla_time_ci"]

    def test_to_csv_rows(self, report):
        rows = report.to_csv_rows()
        assert isinstance(rows, list)
        assert rows[0].startswith("algorithm")  # header
        # header + 2 algorithms × n_trials rows
        assert len(rows) == 1 + 2 * report.n_trials

    def test_trial_timestamps_present(self, report):
        for trial in report.trials:
            assert hasattr(trial, "timestamp")
            assert len(trial.timestamp) > 0

    def test_benchmark_sequence_env(self):
        report = run_benchmark(
            env_name="sequence",
            n_trials=4,
            n_simulations=50,
            seed=1,
            verbose=False,
        )
        assert report.speedup > 0
        assert isinstance(report.vanilla_time_ci, ConfidenceInterval)

    @pytest.mark.timeout(30)
    def test_dry_run_uses_reduced_budget(self):
        report = run_benchmark(
            env_name="maxtree",
            n_trials=20,   # will be clamped to 5
            n_simulations=500,  # will be clamped to 40
            seed=0,
            verbose=False,
            dry_run=True,
        )
        assert report.n_trials <= 5
        assert report.n_simulations <= 40


# ─────────────────────────────────────────────────────────────────────────────
# CLI flags
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    """Tests for benchmark_mcts.py CLI entry point."""

    def _run(self, argv):
        """Import and call main() with argv; return exit code."""
        import benchmark_mcts
        return benchmark_mcts.main(argv)

    def test_dry_run_exits_zero(self, tmp_path):
        os.chdir(tmp_path)
        # Need gumbel_mcts importable
        code = self._run(["--dry-run", "--env", "maxtree", "--quiet"])
        assert code == 0

    def test_json_format_writes_valid_json(self, tmp_path):
        os.chdir(tmp_path)
        out = str(tmp_path / "out.json")
        code = self._run([
            "--dry-run", "--env", "maxtree", "--quiet",
            "--format", "json", "--output", out,
        ])
        assert code == 0
        with open(out) as f:
            data = json.load(f)
        assert "speedup" in data

    def test_csv_format_writes_csv(self, tmp_path):
        os.chdir(tmp_path)
        out = str(tmp_path / "out.csv")
        code = self._run([
            "--dry-run", "--env", "maxtree", "--quiet",
            "--format", "csv", "--output", out,
        ])
        assert code == 0
        with open(out) as f:
            lines = f.readlines()
        assert lines[0].startswith("algorithm")

    def test_sequence_env_via_cli(self, tmp_path):
        os.chdir(tmp_path)
        code = self._run(["--dry-run", "--env", "sequence", "--quiet"])
        assert code == 0

    def test_gridworld_env_via_cli(self, tmp_path):
        os.chdir(tmp_path)
        code = self._run(["--dry-run", "--env", "gridworld", "--quiet"])
        assert code == 0

    def test_compare_sims_exits_zero(self, tmp_path):
        os.chdir(tmp_path)
        code = self._run([
            "--compare-sims", "40,60",
            "--trials", "3",
            "--env", "maxtree",
            "--quiet",
        ])
        assert code == 0

    def test_tree_mode_exits_zero(self, tmp_path):
        os.chdir(tmp_path)
        code = self._run(["--tree", "--sims", "50", "--env", "maxtree"])
        assert code == 0

    def test_invalid_compare_sims(self, tmp_path):
        os.chdir(tmp_path)
        code = self._run(["--compare-sims", "not,a,number"])
        assert code == 1

    def test_custom_output_path(self, tmp_path):
        os.chdir(tmp_path)
        out = str(tmp_path / "my_report.txt")
        code = self._run([
            "--dry-run", "--env", "maxtree", "--quiet",
            "--format", "text", "--output", out,
        ])
        assert code == 0
        assert os.path.exists(out)


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceInterval
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceInterval:

    def test_from_samples_basic(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = ConfidenceInterval.from_samples(samples)
        assert abs(ci.mean - 3.0) < 1e-10
        assert ci.margin > 0
        assert ci.low < ci.mean < ci.high
        assert ci.n == 5

    def test_single_sample(self):
        ci = ConfidenceInterval.from_samples([42.0])
        assert ci.mean == 42.0
        assert ci.margin == 0.0
        assert ci.n == 1

    def test_str_output(self):
        ci = ConfidenceInterval.from_samples([0.1, 0.2, 0.3])
        s = str(ci)
        assert "CI" in s
        assert "±" in s
