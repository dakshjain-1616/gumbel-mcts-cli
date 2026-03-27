"""Benchmarking utilities: run head-to-head comparison of Vanilla vs Gumbel MCTS."""
from __future__ import annotations
import os
import time
import json
import math
import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .vanilla_mcts import VanillaMCTS
from .gumbel_mcts import GumbelMCTS
from .env import MaxTreeEnv, GridWorldEnv, SequenceEnv


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    algorithm: str
    trial: int
    n_simulations: int
    elapsed_sec: float
    root_value: float
    best_action: Any
    n_nodes: int
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


@dataclass
class ConfidenceInterval:
    """95% CI for a list of float samples using the normal approximation."""
    mean: float
    std: float
    margin: float   # ±margin (z=1.96, two-tailed 95%)
    low: float
    high: float
    n: int

    @classmethod
    def from_samples(cls, samples: List[float]) -> "ConfidenceInterval":
        arr = np.array(samples, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        margin = 1.96 * sem
        return cls(mean=mean, std=std, margin=margin, low=mean - margin, high=mean + margin, n=n)

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.margin:.4f} (95% CI [{self.low:.4f}, {self.high:.4f}])"


@dataclass
class BenchmarkReport:
    env_name: str
    n_trials: int
    n_simulations: int
    vanilla_times: List[float]
    gumbel_times: List[float]
    vanilla_values: List[float]
    gumbel_values: List[float]
    speedup: float
    quality_ratio: float  # gumbel_quality / vanilla_quality (>1 = Gumbel wins on value)
    timestamp: str
    trials: List[TrialResult]
    # Confidence intervals (populated by run_benchmark)
    vanilla_time_ci: Optional[ConfidenceInterval] = field(default=None)
    gumbel_time_ci: Optional[ConfidenceInterval] = field(default=None)
    vanilla_value_ci: Optional[ConfidenceInterval] = field(default=None)
    gumbel_value_ci: Optional[ConfidenceInterval] = field(default=None)

    def summary(self) -> str:
        v_ci = self.vanilla_time_ci
        g_ci = self.gumbel_time_ci
        vv_ci = self.vanilla_value_ci
        gv_ci = self.gumbel_value_ci

        def _ms(ci: Optional[ConfidenceInterval], times: List[float]) -> str:
            if ci:
                return (
                    f"{ci.mean*1000:.2f} ms ± {ci.margin*1000:.2f} ms  "
                    f"(95% CI [{ci.low*1000:.2f}, {ci.high*1000:.2f}] ms)"
                )
            return f"{np.mean(times)*1000:.2f} ms"

        def _val(ci: Optional[ConfidenceInterval], vals: List[float]) -> str:
            if ci:
                return (
                    f"{ci.mean:.4f} ± {ci.margin:.4f}  "
                    f"(95% CI [{ci.low:.4f}, {ci.high:.4f}])"
                )
            return f"{np.mean(vals):.4f}"

        v_mean_ms = np.mean(self.vanilla_times) * 1000
        g_mean_ms = np.mean(self.gumbel_times) * 1000
        v_std_ms = np.std(self.vanilla_times) * 1000
        g_std_ms = np.std(self.gumbel_times) * 1000

        lines = [
            "=" * 64,
            f"  Benchmark : {self.env_name}",
            f"  Timestamp : {self.timestamp}",
            "=" * 64,
            f"  Simulations per call : {self.n_simulations}",
            f"  Trials               : {self.n_trials}",
            "",
            "  Vanilla MCTS",
            f"    Time  : {_ms(v_ci, self.vanilla_times)}",
            f"    StdDev: {v_std_ms:.2f} ms",
            f"    Value : {_val(vv_ci, self.vanilla_values)}",
            "",
            "  Gumbel MCTS",
            f"    Time  : {_ms(g_ci, self.gumbel_times)}",
            f"    StdDev: {g_std_ms:.2f} ms",
            f"    Value : {_val(gv_ci, self.gumbel_values)}",
            "",
            f"  {'─'*56}",
            f"  Speedup  (Vanilla / Gumbel) : {self.speedup:.2f}×",
            f"  Quality ratio (Gumbel/Van.) : {self.quality_ratio:.4f}",
            f"  {'─'*56}",
            f"  Vanilla 10th/50th/90th pct  : "
            f"{np.percentile(self.vanilla_times, 10)*1000:.2f} / "
            f"{np.percentile(self.vanilla_times, 50)*1000:.2f} / "
            f"{np.percentile(self.vanilla_times, 90)*1000:.2f} ms",
            f"  Gumbel  10th/50th/90th pct  : "
            f"{np.percentile(self.gumbel_times, 10)*1000:.2f} / "
            f"{np.percentile(self.gumbel_times, 50)*1000:.2f} / "
            f"{np.percentile(self.gumbel_times, 90)*1000:.2f} ms",
            "=" * 64,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trials"] = [asdict(t) for t in self.trials]
        # Convert CI objects
        for key in ("vanilla_time_ci", "gumbel_time_ci", "vanilla_value_ci", "gumbel_value_ci"):
            ci = getattr(self, key)
            d[key] = asdict(ci) if ci is not None else None
        return d

    def to_csv_rows(self) -> List[str]:
        """Return CSV rows (header + one row per trial) for machine consumption."""
        header = "algorithm,trial,n_simulations,elapsed_ms,root_value,n_nodes,timestamp"
        rows = [header]
        for t in self.trials:
            rows.append(
                f"{t.algorithm},{t.trial},{t.n_simulations},"
                f"{t.elapsed_sec*1000:.4f},{t.root_value:.6f},"
                f"{t.n_nodes},{t.timestamp}"
            )
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_nodes(root) -> int:
    """Count total nodes in the MCTS tree via BFS."""
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children.values())
    return count


def _make_env(env_name: str, rng: np.random.Generator):
    """Construct a fresh environment instance for the given env name."""
    seed = int(rng.integers(0, 2**31))
    if env_name == "maxtree":
        return MaxTreeEnv(
            depth=int(os.getenv("TREE_DEPTH", "6")),
            branching=int(os.getenv("TREE_BRANCHING", "8")),
            seed=seed,
        )
    elif env_name == "gridworld":
        return GridWorldEnv(
            size=int(os.getenv("GRID_SIZE", "8")),
            max_steps=int(os.getenv("MAX_STEPS", "200")),
            seed=seed,
        )
    elif env_name == "sequence":
        return SequenceEnv(
            vocab_size=int(os.getenv("SEQ_VOCAB_SIZE", "8")),
            depth=int(os.getenv("SEQ_DEPTH", "5")),
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown env '{env_name}'. Choose from: maxtree, gridworld, sequence"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    env_name: str = os.getenv("BENCH_ENV", "maxtree"),
    n_trials: int = int(os.getenv("BENCH_N_TRIALS", "20")),
    n_simulations: int = int(os.getenv("MCTS_N_SIM", "200")),
    seed: int = int(os.getenv("BENCH_SEED", "42")),
    verbose: bool = True,
    dry_run: bool = False,
    on_trial_complete: Optional[Callable] = None,
) -> BenchmarkReport:
    """Run head-to-head benchmark and return a ``BenchmarkReport``.

    Parameters
    ----------
    env_name : str
        One of ``"maxtree"``, ``"gridworld"``, ``"sequence"``.
    n_trials : int
        How many independent trials to average.
    n_simulations : int
        MCTS simulation budget per search call.
    seed : int
        Global RNG seed for reproducibility.
    verbose : bool
        Print per-trial progress and the final summary.
    dry_run : bool
        If True, use a tiny budget (5 trials, 40 sims) — useful for CI / demos.
    on_trial_complete : callable, optional
        Called after each trial pair as
        ``on_trial_complete(trial_num, total, v_elapsed, v_value, g_elapsed, g_value)``.
        Suppresses the built-in verbose print for that trial when provided.
    """
    if dry_run:
        n_trials = min(n_trials, 5)
        n_simulations = min(n_simulations, 40)
        if verbose:
            print("[dry-run] Using n_trials=5, n_simulations=40")

    rng = np.random.default_rng(seed)
    vanilla = VanillaMCTS(n_simulations=n_simulations, seed=int(rng.integers(0, 2**31)))
    gumbel = GumbelMCTS(n_simulations=n_simulations, seed=int(rng.integers(0, 2**31)))

    vanilla_times: List[float] = []
    gumbel_times: List[float] = []
    vanilla_values: List[float] = []
    gumbel_values: List[float] = []
    trials: List[TrialResult] = []

    if verbose:
        print(
            f"\nRunning benchmark: {env_name.upper()}"
            f" | {n_trials} trials | {n_simulations} sims"
        )
        print("-" * 64)

    for i in range(n_trials):
        # Fresh env for each algorithm on the same trial (deterministic seed)
        env_v = _make_env(env_name, rng)
        state_v = env_v.reset()

        # --- Vanilla MCTS ---
        t0 = time.perf_counter()
        v_action, v_root, _ = vanilla.search_with_stats(env_v, state_v)
        v_elapsed = time.perf_counter() - t0
        v_value = v_root.value
        vanilla_times.append(v_elapsed)
        vanilla_values.append(v_value)
        trials.append(TrialResult(
            algorithm="vanilla",
            trial=i,
            n_simulations=n_simulations,
            elapsed_sec=v_elapsed,
            root_value=v_value,
            best_action=str(v_action),
            n_nodes=_count_nodes(v_root),
        ))

        # --- Gumbel MCTS ---
        env_g = _make_env(env_name, rng)
        state_g = env_g.reset()

        t0 = time.perf_counter()
        g_action, g_root, _ = gumbel.search_with_stats(env_g, state_g)
        g_elapsed = time.perf_counter() - t0
        g_value = g_root.value
        gumbel_times.append(g_elapsed)
        gumbel_values.append(g_value)
        trials.append(TrialResult(
            algorithm="gumbel",
            trial=i,
            n_simulations=n_simulations,
            elapsed_sec=g_elapsed,
            root_value=g_value,
            best_action=str(g_action),
            n_nodes=_count_nodes(g_root),
        ))

        if on_trial_complete is not None:
            on_trial_complete(i + 1, n_trials, v_elapsed, v_value, g_elapsed, g_value)
        elif verbose and (i % 5 == 0 or i == n_trials - 1):
            print(
                f"  Trial {i+1:>3}/{n_trials} | "
                f"Vanilla: {v_elapsed*1000:6.2f}ms val={v_value:.4f} | "
                f"Gumbel:  {g_elapsed*1000:6.2f}ms val={g_value:.4f}"
            )

    mean_v = float(np.mean(vanilla_times))
    mean_g = float(np.mean(gumbel_times))
    speedup = mean_v / mean_g if mean_g > 0 else 0.0

    mean_vv = float(np.mean(vanilla_values))
    mean_gv = float(np.mean(gumbel_values))
    quality_ratio = mean_gv / mean_vv if abs(mean_vv) > 1e-10 else 1.0

    report = BenchmarkReport(
        env_name=env_name,
        n_trials=n_trials,
        n_simulations=n_simulations,
        vanilla_times=vanilla_times,
        gumbel_times=gumbel_times,
        vanilla_values=vanilla_values,
        gumbel_values=gumbel_values,
        speedup=speedup,
        quality_ratio=quality_ratio,
        timestamp=datetime.datetime.now().isoformat(),
        trials=trials,
        vanilla_time_ci=ConfidenceInterval.from_samples(vanilla_times),
        gumbel_time_ci=ConfidenceInterval.from_samples(gumbel_times),
        vanilla_value_ci=ConfidenceInterval.from_samples(vanilla_values),
        gumbel_value_ci=ConfidenceInterval.from_samples(gumbel_values),
    )

    if verbose:
        print(report.summary())

    return report
